"""
对接 vLLM OpenAI-compatible 端点的最小客户端。
不依赖 openai SDK，避免额外依赖；纯标准库 urllib 实现。
"""
import json
import re
import time
import urllib.error
import urllib.request


class LLMClient:
    def __init__(self, base_url="http://localhost:8000/v1", model="qwen3-32b", api_key=None,
                 send_thinking_kwarg=True, default_timeout=1800, verbose=False):
        """api_key:端点需要 Bearer 鉴权时传入,自建 vLLM 留空即可。
        send_thinking_kwarg:`chat_template_kwargs.enable_thinking`是 vLLM+Qwen3 专用字段,
        其他 OpenAI 兼容端点不认识这个字段,为避免被严格 schema 校验拒绝,非 vLLM 场景应传 False。
        default_timeout:自建 vLLM 排队严重时单次请求可能等待较久,默认给 1800s;
        经网络中转的端点正常响应是秒级到几十秒,长时间无响应大概率是连接异常导致的假死,
        应快速失败进入重试,这类场景建议构造时传更短的 default_timeout(如 120s)。"""
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.send_thinking_kwarg = send_thinking_kwarg
        self.default_timeout = default_timeout
        self.verbose = verbose  # 打开后每次调用打印一行耗时/token日志,用于定位延迟环节
        self.n_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0  # 按 token 计费的端点可用来事后核算成本

    def chat(self, messages, max_tokens=1024, temperature=0.3, enable_thinking=False, timeout=None,
              seed=None, network_retries=3):
        """timeout 默认 1800s:多条实验链并发共享同一个 vLLM 实例时,队列深的情况下单个请求的
        排队等待可能超过几百秒,超时不代表请求本身有问题,而是资源共享下的正常排队延迟。
        network_retries 专门兜底这种超时/连接类瞬时错误(JSON 格式错误由 chat_json 的 retries
        处理),避免整条实验链因为一次排队超时就写入失败结果、再也不会被重试。"""
        if timeout is None:
            timeout = self.default_timeout
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.send_thinking_kwarg:
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if seed is not None:
            payload["seed"] = seed  # 固定采样种子，保证严格可复现（vLLM OpenAI兼容端点支持）
        last_err = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        for attempt in range(network_retries + 1):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
            )
            t0 = time.time()
            if self.verbose:
                print(f"    [llm] 发起调用 #{self.n_calls + 1} prompt_chars={prompt_chars} "
                      f"max_tokens={max_tokens} attempt={attempt} ...", flush=True)
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage") or {}
                self.n_calls += 1
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                if self.verbose:
                    print(f"    [llm] 调用完成 耗时={time.time()-t0:.1f}s "
                          f"prompt_tok={usage.get('prompt_tokens')} completion_tok={usage.get('completion_tokens')}",
                          flush=True)
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = e
                if self.verbose:
                    print(f"    [llm] 调用失败({type(e).__name__}: {e}) 耗时={time.time()-t0:.1f}s "
                          f"attempt={attempt}/{network_retries}", flush=True)
                if attempt < network_retries:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
        raise last_err

    def chat_json(self, messages, max_tokens=800, temperature=0.2, retries=2, seed=None, **kwargs):
        last_err = None
        for attempt in range(retries + 1):
            call_seed = None if seed is None else seed + attempt  # 重试时换种子，避免死循环重复同一坏输出
            text = self.chat(messages, max_tokens=max_tokens, temperature=temperature, seed=call_seed, **kwargs)
            try:
                return _extract_json(text)
            except Exception as e:
                last_err = e
                # 追加更严格的指令重试一次
                messages = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "你上面的输出不是合法JSON。请只输出合法JSON，不要任何其他文字、不要markdown代码块。"},
                ]
        raise RuntimeError(f"LLM 未能返回合法JSON（{retries+1}次尝试后放弃）: {last_err}")


def _extract_json(text):
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
        if starts:
            start = min(starts)
            end = max(text.rfind("}"), text.rfind("]"))
            if end > start:
                text = text[start:end + 1]
    return json.loads(text)
