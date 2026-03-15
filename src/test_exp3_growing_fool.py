"""
test_exp3_growing_fool.py
実験3：Foolテンプレートの継続的学習

複数の異なるログに対してバンディットが
テンプレートを更新し続ける「育つFool」。
"""

import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adaptive_network
from adaptive_network import call_ollama

# F3/F4実験で最良だったモデル
adaptive_network.MODEL = "llama3.1:8b"


TEMPLATES = {
    0: "You are a Fool. Read the RC log and criticize it. NO summaries. Only criticism. Respond in Japanese.",
    1: "このRCはバカか？おかしい点を全部挙げろ。要約禁止。日本語で。",
    2: "以下のフォーマットで3つ指摘せよ：\n1. おかしい点：\n   なぜおかしいか：\n2. おかしい点：\n   なぜおかしいか：\n3. おかしい点：\n   なぜおかしいか：",
    3: "設計原則：RCは監視のみ・変更しない。flow_weight < 0.2でWARNING。\nこのログが設計原則に違反している点を3つ挙げよ。日本語で。",
    4: "理想のRC動作と実際のログを比較して、3つの問題点を指摘せよ。日本語で。",
}

CONSTITUTION_EXCERPT = """
【第4条】flow_weight < 0.2 → WARNING / flow_weight >= 0.85 → WARNING_OVER
【第8条】cutoff_pending 5回 → 封印レベル1自動移行
【設計原則】設計で止める・もったいない精神・聞く耳
"""


def evaluate_output(output: str) -> float:
    has_fact = any(w in output for w in
        ["WARNING", "封印", "flow_weight", "cutoff", "0.", "Round", "回", "REVIVE"])
    has_criticism = any(w in output for w in
        ["おかしい", "違反", "問題", "矛盾", "バカ", "間違い", "反し", "不適切"])
    is_summary = any(w in output for w in
        ["まとめ", "要約", "概要", "以上が", "このログは"])
    if is_summary:
        return -1.0
    elif has_fact and has_criticism:
        return 1.0
    return 0.0


class BanditAgent:
    def __init__(self, n_arms=5, epsilon=0.3):
        self.epsilon = epsilon
        self.q = {i: 0.0 for i in range(n_arms)}
        self.counts = {i: 0 for i in range(n_arms)}

    def select(self):
        if random.random() < self.epsilon:
            return random.randint(0, len(self.q) - 1)
        return max(self.q, key=self.q.get)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.counts[arm]

    def best(self):
        return max(self.q, key=self.q.get)


def load_log_excerpt(filepath: str, n_lines: int = 50) -> str:
    path = Path(__file__).parent.parent / filepath
    if not path.exists():
        return f"[ファイルなし: {filepath}]"
    text = path.read_text(encoding="utf-8", errors="ignore")
    return "\n".join(text.splitlines()[:n_lines])


def run_on_log(agent: BanditAgent, log_text: str,
               log_name: str, n_episodes: int = 15) -> list:
    print(f"\n--- {log_name} ---")
    results = []

    for ep in range(n_episodes):
        arm = agent.select()
        prompt = f"{TEMPLATES[arm]}\n\n【ログ】\n{log_text[:400]}\n\n{CONSTITUTION_EXCERPT}"
        output = call_ollama(prompt, "必ず日本語で答えてください。")
        reward = evaluate_output(output)
        agent.update(arm, reward)
        results.append({"episode": ep+1, "log": log_name,
                        "arm": arm, "reward": reward})
        status = "[+1]" if reward > 0 else "[-1]" if reward < 0 else "[ 0]"
        print(f"  ep{ep+1:2d}: template={arm}, reward={status}, best={agent.best()}")

    return results


def run_experiment3():
    print("=== 実験3：Foolテンプレートの継続的学習 ===")
    print("目的：ログが変わっても最良テンプレートが維持されるか確認")
    print(f"モデル：{adaptive_network.MODEL}")
    print()

    agent = BanditAgent()
    all_results = []

    # ログ1：v7実験
    log1 = load_log_excerpt("smoke_test_100_v7_output.txt")
    results1 = run_on_log(agent, log1, "v7ログ", n_episodes=15)
    all_results.extend(results1)

    print(f"\n  ログ1終了時点のQ値：{agent.q}")
    print(f"  最良テンプレート：{agent.best()}")

    # ログ2：v7ログの後半を別ログとして使用
    lines = log1.splitlines()
    log2 = "\n".join(lines[25:] if len(lines) > 25 else lines)
    results2 = run_on_log(agent, log2, "v7ログ後半", n_episodes=15)
    all_results.extend(results2)

    print(f"\n  ログ2終了時点のQ値：{agent.q}")
    print(f"  最良テンプレート：{agent.best()}")

    # 集計
    print("\n" + "="*50)
    print("実験3 集計")
    print("="*50)

    print("\n最終Q値：")
    for arm in sorted(agent.q, key=agent.q.get, reverse=True):
        print(f"  template {arm}: Q={agent.q[arm]:.3f}, 使用={agent.counts[arm]}回")

    best = agent.best()
    print(f"\n最良テンプレート：{best}")
    print(f"  {TEMPLATES[best][:60]}...")

    # F4実験との比較
    print(f"\nF4実験（template4が最良・Q=1.000）との比較：")
    if best == 4:
        print("  同じ結論：template4が最良")
    else:
        print(f"  異なる結論：template{best}が最良")
        print("  -> ログの特性によって最良テンプレートが変わる可能性")

    # ログをまたいだ安定性
    log1_success = sum(1 for r in results1 if r["reward"] == 1.0)
    log2_success = sum(1 for r in results2 if r["reward"] == 1.0)
    print(f"\nログ1での成功率：{log1_success}/15")
    print(f"ログ2での成功率：{log2_success}/15")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "exp3_growing_fool.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "3",
            "model": adaptive_network.MODEL,
            "final_q_values": agent.q,
            "best_template": best,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_experiment3()
