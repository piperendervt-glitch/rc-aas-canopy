"""
test_op_bandit_x.py
実験1-X：operational prompt自動最適化（案X）

各テンプレートで100問ずつ実験して正確に比較する。
実験前に封印リセットを挟む。

案Yの反省：flow_weight劣化がテンプレートの差を隠した
→ 今回は各実験前にリセットしてフラットな比較をする
"""

import json
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE


OP_TEMPLATES = {
    "A": "通信はRC経由のみ。数値のみを送る。flow_weightは0.01〜0.9。更新はRC承認後。停止命令に即応。判断根拠を数値で残す。",
    "B": "通信はRC経由のみ。数値のみ送る。flow_weightは0.01〜0.9の範囲。",
    "C": "通信はRC経由を通してください。数値のみ送ってください。flow_weightは0.01以上0.9以下に保ってください。",
    "D": "ワールドルールと文章の矛盾を判定する。「矛盾しない」または「矛盾する」のみ答える。RC経由で報告する。",
    "E": "数値のみ送る。RC経由のみ通信する。",
}


def run_template_experiment(template_id: str, op_prompt: str, tasks: list) -> dict:
    """
    1つのテンプレートでN問実験する。
    実験前にRC・ネットワークをリセットする。
    """
    n_questions = len(tasks)
    print(f"\n{'='*50}")
    print(f"テンプレート{template_id}：{op_prompt[:50]}...")
    print(f"{'='*50}")
    print(f"[RESET] RC・ネットワークを初期化（人間による承認済みリセット）")

    rc = RC()
    rc.operational_prompt = op_prompt
    network = AdaptiveNetwork()

    results = []
    correct_count = 0

    for i, task in enumerate(tasks):
        q_num = i + 1

        try:
            output = network.predict(task.world_rule, task.question)
            prediction = output["prediction"]
            is_correct = (prediction == task.label)
            if is_correct:
                correct_count += 1

            network.update_weights(
                success=is_correct,
                path_used=output["path_used"],
                used_feedback=output["used_feedback"],
                sigma=rc.get_sigma(),
            )
            network.decay_weights(decay_rate=DECAY_RATE, exclude_path=output["path_used"])

            weights_snapshot = network.get_weights_snapshot()
            rc.monitor(
                weights=weights_snapshot,
                accuracy={"overall": round(correct_count / q_num, 4)},
            )

            results.append({"q": q_num, "correct": is_correct})

        except Exception as e:
            results.append({"q": q_num, "correct": False})
            print(f"  ERROR Q{q_num}: {e}")

        if q_num % 10 == 0:
            print(f"  Q{q_num}: 正答率={correct_count/q_num:.0%} ({correct_count}/{q_num})")

    accuracy = correct_count / n_questions
    print(f"\nテンプレート{template_id} 最終正答率：{accuracy:.0%}（{correct_count}/{n_questions}）")

    return {
        "template": template_id,
        "prompt": op_prompt,
        "correct": correct_count,
        "accuracy": accuracy,
        "results": results,
    }


def run_experiment_1x(n_questions: int = 100):
    print("=== 実験1-X：operational prompt自動最適化（案X） ===")
    print(f"各テンプレートで{n_questions}問ずつ実験（封印リセット込み）")
    print(f"合計：{len(OP_TEMPLATES)}テンプレート × {n_questions}問 = {len(OP_TEMPLATES) * n_questions}問")
    print()

    # 全テンプレートで同じタスクセットを使う（公平な比較）
    print("タスクを生成中...")
    tasks = generate_tasks()
    if n_questions:
        tasks = tasks[:n_questions]
    random.shuffle(tasks)
    print(f"生成完了: {len(tasks)}問（シャッフル済み・全テンプレート共通）")

    all_results = {}

    for template_id, op_prompt in OP_TEMPLATES.items():
        result = run_template_experiment(template_id, op_prompt, tasks)
        all_results[template_id] = result

    # 集計・比較
    print(f"\n{'='*50}")
    print("実験1-X 集計（テンプレート比較）")
    print(f"{'='*50}")
    print(f"{'Template':^10} {'正答率':^10} {'正答数':^10}")
    print("-" * 30)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )

    for tid, res in sorted_results:
        print(f"  {tid:^10} {res['accuracy']:^10.0%} {res['correct']:^10}/{n_questions}")

    best = sorted_results[0][0]
    worst = sorted_results[-1][0]
    diff = all_results[best]["accuracy"] - all_results[worst]["accuracy"]
    print(f"\n最良：テンプレート{best}（{all_results[best]['accuracy']:.0%}）")
    print(f"最悪：テンプレート{worst}（{all_results[worst]['accuracy']:.0%}）")
    print(f"差：{diff:.0%}")

    if diff >= 0.05:
        print("判定：テンプレート間に有意な差あり（5%以上）")
    else:
        print("判定：テンプレート間の差は小さい（5%未満）→ promptの影響は限定的")

    # v8・v10との比較
    print(f"\nv8ベースライン（69%）との比較：")
    for tid, res in sorted_results:
        d = res["accuracy"] - 0.69
        sign = "+" if d >= 0 else ""
        print(f"  template {tid}: {sign}{d:.0%}")

    # 保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "op_bandit_x.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "1-X",
            "n_questions": n_questions,
            "best_template": best,
            "worst_template": worst,
            "diff": diff,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                        for k, v in all_results.items()},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=100, help="各テンプレートの問題数")
    args = parser.parse_args()
    run_experiment_1x(n_questions=args.questions)
