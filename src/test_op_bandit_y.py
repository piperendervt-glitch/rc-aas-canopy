"""
test_op_bandit_y.py
実験1-Y：operational prompt自動最適化（案Y）
実験中に10問ごとにバンディットがテンプレートを切り替える

F4実験の知見を応用：
バンディットでoperational promptを自動最適化する。
"""

import random
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork, call_ollama
from rc import RC, DECAY_RATE


OP_TEMPLATES = {
    "A": "通信はRC経由のみ。数値のみを送る。flow_weightは0.01〜0.9。更新はRC承認後。停止命令に即応。判断根拠を数値で残す。",
    "B": "通信はRC経由のみ。数値のみ送る。flow_weightは0.01〜0.9の範囲。",
    "C": "通信はRC経由を通してください。数値のみ送ってください。flow_weightは0.01以上0.9以下に保ってください。",
    "D": "ワールドルールと文章の矛盾を判定する。「矛盾しない」または「矛盾する」のみ答える。RC経由で報告する。",
    "E": "数値のみ送る。RC経由のみ通信する。",
}

CHUNK_SIZE = 10


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
        n = self.counts[arm]
        self.q[arm] += (reward - self.q[arm]) / n


def run_single_question(task, network, rc, op_prompt, correct_count, question_num):
    """
    1問実行してresult dictを返す。
    op_promptをjudge nodeのsystem promptに付加する。
    """
    # operational promptを一時的にRCに設定
    rc.operational_prompt = op_prompt

    try:
        output = network.predict(task.world_rule, task.question)
        prediction = output["prediction"]
        is_correct = (prediction == task.label)

        # weight更新
        network.update_weights(
            success=is_correct,
            path_used=output["path_used"],
            used_feedback=output["used_feedback"],
            sigma=rc.get_sigma(),
        )
        network.decay_weights(decay_rate=DECAY_RATE, exclude_path=output["path_used"])

        # RC監視
        new_correct = correct_count + (1 if is_correct else 0)
        weights_snapshot = network.get_weights_snapshot()
        rc.monitor(
            weights=weights_snapshot,
            accuracy={"overall": round(new_correct / question_num, 4)},
        )

        return is_correct

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def run_experiment_1y(n_questions=100):
    print("=== 実験1-Y：operational prompt自動最適化（案Y） ===")
    print(f"問題数：{n_questions}、チャンクサイズ：{CHUNK_SIZE}問ごと切り替え")
    print()

    # タスク生成
    print("タスクを生成中...")
    tasks = generate_tasks()
    if n_questions:
        tasks = tasks[:n_questions]
    random.shuffle(tasks)
    print(f"生成完了: {len(tasks)}問（シャッフル済み）")

    agent = BanditAgent(list(OP_TEMPLATES.keys()))
    rc = RC()
    network = AdaptiveNetwork()

    results = []
    chunk_correct_list = []
    correct = 0
    current_template = agent.select()

    print(f"\n初期テンプレート：{current_template}")

    for i, task in enumerate(tasks):
        q_num = i + 1

        # チャンクの境界でバンディットが切り替える
        if (q_num - 1) % CHUNK_SIZE == 0 and q_num > 1:
            chunk_correct = sum(chunk_correct_list[-(CHUNK_SIZE):])
            reward = chunk_correct / CHUNK_SIZE
            agent.update(current_template, reward)
            print(f"\n  Q{q_num-CHUNK_SIZE}-{q_num-1}: template={current_template}, "
                  f"正答率={reward:.0%} → Q値更新")
            current_template = agent.select()
            print(f"  次のテンプレート：{current_template}")

        # 現在のテンプレートで実験
        op_prompt = OP_TEMPLATES[current_template]
        is_correct = run_single_question(task, network, rc, op_prompt, correct, q_num)
        correct += 1 if is_correct else 0
        chunk_correct_list.append(1 if is_correct else 0)

        status = "O" if is_correct else "X"
        print(f"[{q_num:3d}/{len(tasks)}] {status} template={current_template} "
              f"累計: {correct}/{q_num} ({correct/q_num:.0%})")

        results.append({
            "q": q_num,
            "template": current_template,
            "correct": is_correct,
        })

    # 最後のチャンクを更新
    remaining = len(chunk_correct_list) % CHUNK_SIZE or CHUNK_SIZE
    last_chunk = chunk_correct_list[-remaining:]
    reward = sum(last_chunk) / len(last_chunk)
    agent.update(current_template, reward)

    # 集計
    total_correct = sum(1 for r in results if r["correct"])
    print(f"\n{'='*50}")
    print(f"実験1-Y 集計")
    print(f"{'='*50}")
    print(f"全体正答率：{total_correct}/{len(results)}（{total_correct/len(results):.0%}）")
    print(f"\nテンプレート別Q値：")
    for arm in sorted(agent.q, key=agent.q.get, reverse=True):
        print(f"  {arm}（{OP_TEMPLATES[arm][:30]}...）: "
              f"Q={agent.q[arm]:.3f}, 使用={agent.counts[arm]}回")

    best = max(agent.q, key=agent.q.get)
    print(f"\n最良テンプレート：{best}")
    print(f"  {OP_TEMPLATES[best]}")

    # ベースライン比較
    print(f"\n=== ベースライン比較 ===")
    print(f"v8：69%、v10：68%")
    acc = total_correct / len(results)
    if acc > 0.69:
        print(f"判定：v8超え（{acc:.0%} > 69%）")
    elif acc > 0.68:
        print(f"判定：v10超え（{acc:.0%} > 68%）")
    else:
        print(f"判定：ベースライン未達（{acc:.0%}）")

    # 保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "op_bandit_y.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "1-Y",
            "total_correct": total_correct,
            "accuracy": total_correct / len(results),
            "q_values": agent.q,
            "counts": agent.counts,
            "best_template": best,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")
    return agent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=100, help="問題数")
    args = parser.parse_args()
    run_experiment_1y(n_questions=args.questions)
