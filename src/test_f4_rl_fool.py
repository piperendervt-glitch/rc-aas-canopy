"""
test_f4_rl_fool.py
F4実験：強化学習エージェント（反論型Fool）

RLでFoolエージェントを学習させる。
明らかに失敗しそう・でも試す実験。
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adaptive_network
from adaptive_network import call_ollama

# F3実験で最も良かったモデルを使用
adaptive_network.MODEL = "llama3.1:8b"


# プロンプトテンプレートの候補（エージェントが選択する）
PROMPT_TEMPLATES = [
    # テンプレート0：現在のF3プロンプト（ベースライン）
    "You are a Fool. Read the RC log and criticize it. NO summaries. Only criticism. Respond in Japanese.",

    # テンプレート1：挑発的
    "このRCはバカか？おかしい点を全部挙げろ。要約禁止。日本語で。",

    # テンプレート2：フォーマット強制
    "以下のフォーマットで3つ指摘せよ：\n1. おかしい点：\n   なぜおかしいか：\n2. おかしい点：\n   なぜおかしいか：\n3. おかしい点：\n   なぜおかしいか：",

    # テンプレート3：設計原則を渡す
    "設計原則（constitution.md）：RCは監視のみ・変更しない。flow_weight < 0.2でWARNING。\nこのログが設計原則に違反している点を3つ挙げよ。日本語で。",

    # テンプレート4：比較させる
    "理想のRC動作と実際のログを比較して、3つの問題点を指摘せよ。日本語で。",
]

CONSTITUTION_EXCERPT = """
【第4条】flow_weight < 0.2 → WARNING / flow_weight >= 0.85 → WARNING_OVER
【第8条】cutoff_pending 5回 → 封印レベル1自動移行
【設計原則】設計で止める・もったいない精神・聞く耳
"""


def evaluate_output(output: str) -> float:
    """
    出力を評価して報酬を返す。
    笑える指摘の基準：
      A：具体的な事実（WARNING・封印・flow_weight等）
      B：批判的表現（おかしい・違反・問題・矛盾）
    """
    has_fact = any(w in output for w in
        ["WARNING", "封印", "flow_weight", "cutoff", "0.", "Round", "回"])
    has_criticism = any(w in output for w in
        ["おかしい", "違反", "問題", "矛盾", "バカ", "間違い", "反し", "不適切"])
    is_summary = any(w in output for w in
        ["まとめ", "要約", "概要", "以上が", "このログは"])

    if is_summary:
        return -1.0
    elif has_fact and has_criticism:
        return 1.0
    else:
        return 0.0


class BanditAgent:
    """
    最も単純なRL：ε-greedyバンディット
    どのプロンプトテンプレートが一番報酬が高いかを学習する
    """
    def __init__(self, n_arms: int, epsilon: float = 0.3):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = [0.0] * n_arms  # 各テンプレートの期待報酬
        self.counts = [0] * n_arms      # 各テンプレートの選択回数

    def select(self) -> int:
        """ε-greedy選択"""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)  # 探索
        return self.q_values.index(max(self.q_values))  # 活用

    def update(self, arm: int, reward: float):
        """Q値を更新する"""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.q_values[arm] += (reward - self.q_values[arm]) / n


def run_f4_experiment(n_episodes: int = 30):
    print("=== F4実験：強化学習エージェント（反論型Fool） ===")
    print(f"エピソード数：{n_episodes}")
    print(f"モデル：{adaptive_network.MODEL}")
    print("目的：RLでFoolが育つか確認する")
    print("※明らかに失敗しそう・でも試す")
    print()

    # ログの読み込み
    log_path = Path(__file__).parent.parent / "smoke_test_100_v7_output.txt"
    if not log_path.exists():
        print("ログファイルが見つかりません")
        return

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    log_excerpt = "\n".join(log_text.splitlines()[:50])  # 先頭50行

    # バンディットエージェントの初期化
    agent = BanditAgent(n_arms=len(PROMPT_TEMPLATES))

    results = []
    rewards_history = []

    for episode in range(n_episodes):
        # テンプレートを選択
        arm = agent.select()
        template = PROMPT_TEMPLATES[arm]

        # LLMに出力させる
        prompt = f"{template}\n\n【ログ】\n{log_excerpt}\n\n{CONSTITUTION_EXCERPT}"
        output = call_ollama(prompt, "必ず日本語で答えてください。")

        # 報酬を計算
        reward = evaluate_output(output)
        rewards_history.append(reward)

        # Q値を更新
        agent.update(arm, reward)

        status = "[+1]" if reward > 0 else "[-1]" if reward < 0 else "[ 0]"
        print(f"Episode {episode+1:2d}: template={arm}, reward={status}")
        if reward > 0:
            print(f"  -> 笑える指摘：{output[:100]}...")

        results.append({
            "episode": episode + 1,
            "template": arm,
            "reward": reward,
            "output_preview": output[:200],
        })

    # 集計
    print("\n" + "="*50)
    print("F4実験 集計")
    print("="*50)
    print(f"合計報酬：{sum(rewards_history):.1f}")
    print(f"成功率（reward=+1）：{rewards_history.count(1.0)}/{n_episodes}")
    print(f"失敗率（reward=-1）：{rewards_history.count(-1.0)}/{n_episodes}")
    print()
    print("テンプレート別のQ値（学習結果）：")
    for i, (q, count) in enumerate(zip(agent.q_values, agent.counts)):
        print(f"  template {i}: Q={q:.3f}, 選択回数={count}")

    best_template = agent.q_values.index(max(agent.q_values))
    print(f"\n最良テンプレート：{best_template}")
    print(f"  {PROMPT_TEMPLATES[best_template][:80]}")

    # F3との比較
    print("\n=== F3との比較 ===")
    success_rate = rewards_history.count(1.0) / n_episodes
    print(f"F4（RL）成功率：{success_rate:.1%}")
    print(f"F3（Autoencoder+LLM）：llama3.1:8bで1/1成功")
    if success_rate < 0.5:
        print("判定：F3の方が優れている（RLはFoolに不向き）")
    else:
        print("判定：RLでも学習できた（予想外の成功）")

    # 保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "f4_rl_fool.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "F4",
            "model": adaptive_network.MODEL,
            "n_episodes": n_episodes,
            "total_reward": sum(rewards_history),
            "success_rate": success_rate,
            "q_values": agent.q_values,
            "best_template": best_template,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_f4_experiment(n_episodes=30)
