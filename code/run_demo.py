"""
端到端演示:用 30 篇真实论文(强化学习主题,data/common/sample_rl_papers.json)跑通完整
SLRGP 管线 E -> Solve(L/F/R/O/V + 回溯) -> Render(P/W/C 递归)。
LLM 走 OpenAI-compatible 端点,默认本机 vLLM(http://localhost:8000/v1,模型 qwen3-32b);
可用环境变量切换端点,无需改代码:
  SLRGP_LLM_BASE_URL  端点地址(默认 http://localhost:8000/v1)
  SLRGP_LLM_MODEL     模型名(默认 qwen3-32b)
  SLRGP_LLM_API_KEY   Bearer 密钥(自建 vLLM 留空)
非 vLLM 端点(不认识 enable_thinking 字段)请置 SLRGP_LLM_VLLM=0。

自建端点启动示例(vLLM):
  python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-32B --port 8000

用法:
  python3 run_demo.py
"""
import json
import os

from slrgp.state import SLRGPState, Paper
from slrgp.llm_client import LLMClient
from slrgp.operators import op_E, op_C
from slrgp.control import solve, render


def load_corpus(path):
    papers = []
    for item in json.load(open(path, encoding="utf-8")):
        doc_id = item.get("work_id") or item.get("arxivid") or item["title"][:30]
        papers.append(Paper(
            doc_id=doc_id,
            title=item["title"],
            abstract=item["abstract"],
            authors=item.get("authors", []),
            year=item.get("year"),
            tier="T2",  # 演示语料统一置中性档(真实管线中 tier 由 OpenAlex 关联的正式发表 venue 决定)
        ))
    return papers


def main():
    corpus = load_corpus("data/common/sample_rl_papers.json")
    corpus_by_id = {p.doc_id: p for p in corpus}
    print(f"语料库加载完成：{len(corpus)} 篇论文\n")

    llm = LLMClient(
        base_url=os.environ.get("SLRGP_LLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("SLRGP_LLM_MODEL", "qwen3-32b"),
        api_key=os.environ.get("SLRGP_LLM_API_KEY") or None,
        send_thinking_kwarg=os.environ.get("SLRGP_LLM_VLLM", "1") == "1",
    )

    state = SLRGPState(
        q={"seed": "Reinforcement learning: fairness, efficiency, and behavior priors", "terms": []},
        D=[], Gamma={}, kappa={}, x={},
        meta={"top_n": 20, "min_abs_len": 200},
    )

    print("=" * 60)
    print("Step E (expand)：查询扩展")
    print("=" * 60)
    state = op_E(state, llm)
    print("扩展词表:", state.q["terms"], "\n")

    print("=" * 60)
    print("Solve：L(定位) -> F(筛选) -> R(排序) -> O(组织) -> V(验证)，不满足则回溯")
    print("=" * 60)
    state = solve(state, corpus, llm)
    print(f"候选文献数(D): {len(state.D)}")
    print(f"求解轮数: {state.meta.get('solve_rounds')}  是否放弃兜底: {state.meta.get('solve_gave_up', False)}")
    print(f"组织维度(Gamma.dimension): {state.Gamma.get('dimension')}")
    for g in state.Gamma.get("groups", []):
        print(f"  - [{g['name']}] {len(g['doc_ids'])} 篇: {g['doc_ids']}")
    print(f"验证结果(kappa): {state.kappa}\n")

    print("=" * 60)
    print("Render：递归展开 Descend -> Solve -> Render，叶子节点 P(格式化) -> W(写作)")
    print("=" * 60)
    text = render(state, corpus, corpus_by_id, llm, depth=0, d_max=1)
    state.x["text"] = text

    print("=" * 60)
    print("Step C (reffilter)：引用一致性核对")
    print("=" * 60)
    state = op_C(state, corpus_by_id)
    print(f"正文实际引用并保留的文献数: {len(state.x['refs'])}\n")

    out_path = "output_demo_review.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {state.q['seed']}\n\n")
        f.write(f"*组织维度: {state.Gamma.get('dimension')}*\n\n")
        f.write(text)
        f.write("\n\n## References\n")
        for rid in state.x["refs"]:
            p = corpus_by_id.get(rid)
            if p:
                authors = ", ".join(p.authors[:3])
                f.write(f"- [{rid}] {authors}. \"{p.title}\". ({p.year})\n")

    print(f"完整综述已写入 {out_path}")
    print("\n" + "=" * 60)
    print("正文预览（前1500字符）")
    print("=" * 60)
    print(text[:1500])


if __name__ == "__main__":
    main()
