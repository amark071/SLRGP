#!/usr/bin/env python3
"""Minimal OpenAI-compatible Qwen server using Transformers + bitsandbytes.

This is a fallback for 2x32GB machines where BF16 vLLM cannot materialize
Qwen3-32B weights. It implements only the endpoints used by S4b.
"""
from __future__ import annotations

import argparse
import re
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int = 1024
    temperature: float = 0.3
    seed: int | None = None


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def create_app(model_path: str, served_model_name: str, gpu_max_memory: str, cpu_max_memory: str) -> FastAPI:
    app = FastAPI()
    print(f"[qwen-bnb] loading tokenizer: {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"[qwen-bnb] loading model 4-bit: {model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: gpu_max_memory, 1: gpu_max_memory, "cpu": cpu_max_memory},
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("[qwen-bnb] ready", flush=True)

    @app.get("/v1/models")
    def models() -> dict:
        return {"object": "list", "data": [{"id": served_model_name, "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest) -> dict:
        if req.seed is not None:
            torch.manual_seed(req.seed)
            torch.cuda.manual_seed_all(req.seed)
        try:
            encoded = tokenizer.apply_chat_template(
                req.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                req.messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
        if hasattr(encoded, "to") and hasattr(encoded, "data") and "input_ids" in encoded:
            encoded = encoded.to(model.device)
            prompt_len = int(encoded["input_ids"].shape[-1])
            generate_kwargs = dict(encoded)
        else:
            input_ids = encoded.to(model.device)
            prompt_len = int(input_ids.shape[-1])
            generate_kwargs = {"input_ids": input_ids}
        do_sample = req.temperature > 0
        with torch.inference_mode():
            output = model.generate(
                **generate_kwargs,
                max_new_tokens=req.max_tokens,
                do_sample=do_sample,
                temperature=req.temperature if do_sample else None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0, prompt_len:]
        text = strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return {
            "id": f"chatcmpl-s4b-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "model": served_model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(new_tokens.shape[-1]),
                "total_tokens": int(prompt_len + new_tokens.shape[-1]),
            },
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model-name", default="qwen3-32b")
    parser.add_argument("--gpu-max-memory", default="28GiB")
    parser.add_argument("--cpu-max-memory", default="96GiB")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model_path, args.served_model_name, args.gpu_max_memory, args.cpu_max_memory),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
