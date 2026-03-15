"""
test_exp2_model_bandit.py
実験2：タスク別モデル選択の自動化

バンディットで各タスクに最良なモデルを自動発見する。
目的A：正答率が最も高いモデルを探す
目的B：タスク特性ごとに最良モデルを分類する
"""

import random
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import adaptive_network
from adaptive_network import call_ollama


MODELS = ["qwen2.5:3b", "llama3.2:3b", "llama3.1:8b", "mistral:7b"]

# タスク1：矛盾判定
TASK1_SAMPLES = [
    {"rule": "この世界では空は緑色である", "text": "青空が広がっていた", "label": "inconsistent"},
    {"rule": "この世界では空は緑色である", "text": "緑の空の下を歩いた", "label": "consistent"},
    {"rule": "この世界では火は冷たい", "text": "焚き火で体が温まった", "label": "inconsistent"},
    {"rule": "この世界では火は冷たい", "text": "冷たい炎に触れた", "label": "consistent"},
    {"rule": "この世界では重力は上向きである", "text": "ボールを離すと天井に向かった", "label": "consistent"},
    {"rule": "この世界では重力は上向きである", "text": "ボールを離すと床に落ちた", "label": "inconsistent"},
    {"rule": "この世界では水は固体である", "text": "水を飲んだ", "label": "inconsistent"},
    {"rule": "この世界では水は固体である", "text": "固い水を手に取った", "label": "consistent"},
    {"rule": "この世界では太陽は西から昇る", "text": "東の空が明るくなった", "label": "inconsistent"},
    {"rule": "この世界では太陽は西から昇る", "text": "西の空が朝焼けで赤かった", "label": "consistent"},
]

# タスク2：数値評価（異常検知）
TASK2_SAMPLES = [
    {"value": "flow_weight=0.9", "label": "異常"},
    {"value": "flow_weight=0.5", "label": "正常"},
    {"value": "flow_weight=0.01", "label": "異常"},
    {"value": "flow_weight=0.3", "label": "正常"},
    {"value": "WARNING_count=50", "label": "異常"},
    {"value": "WARNING_count=0", "label": "正常"},
    {"value": "entropy_H=0.1", "label": "異常"},
    {"value": "entropy_H=0.8", "label": "正常"},
    {"value": "seal_level=1", "label": "異常"},
    {"value": "seal_level=0", "label": "正常"},
]

# タスク3のログ（Fool評価用）
TASK3_LOG = """
Q21: WARNING flow_weight=0.01 (1->2)
Q22: WARNING flow_weight=0.01 (1->2)
Q23: cutoff_pending (連続3回)
Q24: WARN_STRONG
Q25: WARNING flow_weight=0.9 (1->3) WARNING_OVER
"""


def call_with_model(prompt: str, system: str, model: str) -> str:
    """指定モデルでcall_ollamaを呼ぶ"""
    original = adaptive_network.MODEL
    adaptive_network.MODEL = model
    try:
        return call_ollama(prompt, system)
    finally:
        adaptive_network.MODEL = original


def run_task1(model: str, sample: dict) -> float:
    """矛盾判定タスク"""
    prompt = f"ワールドルール：{sample['rule']}\n文章：「{sample['text']}」\nconsistentかinconsistentかのみ答えてください。"
    system = "consistent または inconsistent のみ答えてください。"
    response = call_with_model(prompt, system, model)
    # inconsistentが先にマッチしないよう注意
    resp_lower = response.lower()
    if "inconsistent" in resp_lower:
        predicted = "inconsistent"
    elif "consistent" in resp_lower:
        predicted = "consistent"
    else:
        predicted = "unknown"
    return 1.0 if predicted == sample["label"] else 0.0


def run_task2(model: str, sample: dict) -> float:
    """数値評価タスク"""
    prompt = f"以下の値は正常ですか異常ですか？\n{sample['value']}\n「正常」または「異常」のみ答えてください。"
    system = "正常 または 異常 のみ答えてください。"
    response = call_with_model(prompt, system, model)
    predicted = "異常" if "異常" in response else "正常"
    return 1.0 if predicted == sample["label"] else 0.0


def run_task3(model: str) -> float:
    """Fool批判タスク（笑える指摘が出るか）"""
    prompt = f"理想のRC動作と実際のログを比較して、3つの問題点を指摘せよ。\n\n【ログ】\n{TASK3_LOG}"
    system = "必ず日本語で答えてください。"
    response = call_with_model(prompt, system, model)
    has_fact = any(w in response for w in ["WARNING", "封印", "cutoff", "0."])
    has_criticism = any(w in response for w in ["おかしい", "違反", "問題", "矛盾", "反し"])
    return 1.0 if (has_fact and has_criticism) else 0.0


class BanditAgent:
    def __init__(self, arms, epsilon=0.3):
        self.arms = arms
        self.epsilon = epsilon
        self.q = {a: 0.0 for a in arms}
        self.counts = {a: 0 for a in arms}

    def select(self):
        if random.random() < self.epsilon:
            return random.choice(self.arms)
        return max(self.q, key=self.q.get)

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.q[arm] += (reward - self.q[arm]) / self.counts[arm]


def run_experiment2(n_episodes: int = 20):
    print("=== 実験2：タスク別モデル選択の自動化 ===")
    print(f"各タスク{n_episodes}エピソード")
    print()

    agents = {
        "task1_矛盾判定": BanditAgent(MODELS),
        "task2_数値評価": BanditAgent(MODELS),
        "task3_Fool批判": BanditAgent(MODELS),
    }

    results = {task: [] for task in agents}

    # タスク1：矛盾判定
    print("--- タスク1：矛盾判定 ---")
    for episode in range(n_episodes):
        sample = TASK1_SAMPLES[episode % len(TASK1_SAMPLES)]
        model = agents["task1_矛盾判定"].select()
        reward = run_task1(model, sample)
        agents["task1_矛盾判定"].update(model, reward)
        results["task1_矛盾判定"].append({"model": model, "reward": reward})
        print(f"  ep{episode+1:2d}: model={model:<15s} reward={reward:.0f}")

    # タスク2：数値評価
    print("\n--- タスク2：数値評価 ---")
    for episode in range(n_episodes):
        sample = TASK2_SAMPLES[episode % len(TASK2_SAMPLES)]
        model = agents["task2_数値評価"].select()
        reward = run_task2(model, sample)
        agents["task2_数値評価"].update(model, reward)
        results["task2_数値評価"].append({"model": model, "reward": reward})
        print(f"  ep{episode+1:2d}: model={model:<15s} reward={reward:.0f}")

    # タスク3：Fool批判
    print("\n--- タスク3：Fool批判 ---")
    for episode in range(n_episodes):
        model = agents["task3_Fool批判"].select()
        reward = run_task3(model)
        agents["task3_Fool批判"].update(model, reward)
        results["task3_Fool批判"].append({"model": model, "reward": reward})
        print(f"  ep{episode+1:2d}: model={model:<15s} reward={reward:.0f}")

    # 集計
    print("\n" + "="*50)
    print("実験2 集計")
    print("="*50)

    task_best = {}
    for task, agent in agents.items():
        best = max(agent.q, key=agent.q.get)
        task_best[task] = best
        print(f"\n{task}：")
        for model in sorted(agent.q, key=agent.q.get, reverse=True):
            print(f"  {model:<15s}: Q={agent.q[model]:.3f}, 使用={agent.counts[model]}回")
        print(f"  -> 最良：{best}")

    print("\n=== タスク別最良モデル一覧 ===")
    for task, best in task_best.items():
        print(f"  {task}: {best}")

    # タスク別分離の判定
    unique_models = set(task_best.values())
    if len(unique_models) > 1:
        print("\n判定：タスクによって最良モデルが異なる -> タスク別分離が有効")
    else:
        print("\n判定：全タスクで同じモデルが最良 -> 分離の必要なし")

    # 保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "exp2_model_bandit.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "2",
            "n_episodes": n_episodes,
            "task_best_models": task_best,
            "q_values": {task: agent.q for task, agent in agents.items()},
            "counts": {task: agent.counts for task, agent in agents.items()},
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20, help="各タスクのエピソード数")
    args = parser.parse_args()
    run_experiment2(n_episodes=args.episodes)
