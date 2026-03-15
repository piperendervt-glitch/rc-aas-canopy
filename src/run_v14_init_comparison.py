"""
run_v14_init_comparison.py
v14実験：v12と同じgenerate_tasks()でパターンA・C・Eを比較

v13ではgenerate_tasks(count=100, seed=42)を使ったためタスクセットが異なり
v12の69%と比較できなかった。v14はv12と同じgenerate_tasks()（固定100問）で再実験。
"""

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EDGE_MAP = {
    "1->2": (1, 2), "2->1": (2, 1),
    "2->3": (2, 3), "3->2": (3, 2),
    "1->3": (1, 3), "3->1": (3, 1),
}

PATTERNS = {
    "A": {
        "name": "均一（v12ベースライン）",
        "weights": {
            "1->2": 0.5, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.5, "3->1": 0.5,
        },
    },
    "C": {
        "name": "タスク特化型",
        "weights": {
            "1->2": 0.7, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.3, "3->1": 0.5,
        },
    },
    "E": {
        "name": "多様性強制",
        "weights": {
            "1->2": 0.3, "2->1": 0.5,
            "2->3": 0.5, "3->2": 0.5,
            "1->3": 0.7, "3->1": 0.5,
        },
    },
}


def set_initial_weights(network: AdaptiveNetwork, weights: dict):
    for key_str, value in weights.items():
        edge = EDGE_MAP[key_str]
        if edge in network.connections:
            network.connections[edge].flow_weight = value


def calc_entropy(weights: dict) -> float:
    values = list(weights.values())
    total = sum(values)
    if total == 0:
        return 0.0
    H = 0.0
    for w in values:
        p = w / total
        if p > 0:
            H -= p * math.log(p)
    return H


def run_trial(init_weights: dict, tasks) -> dict:
    rc = RC()
    network = AdaptiveNetwork()
    set_initial_weights(network, init_weights)

    correct = 0
    errors = 0
    rc_warnings = 0
    questions_done = 0

    for i, task in enumerate(tasks):
        if rc.is_stopped():
            print(f"    [RC] 停止中のため中断（{i}問完了）")
            break

        try:
            output = network.predict(task.world_rule, task.question)
            prediction = output["prediction"]
            is_correct = (prediction == task.label)
            correct += 1 if is_correct else 0
            questions_done += 1

            network.update_weights(
                success=is_correct,
                path_used=output["path_used"],
                used_feedback=output["used_feedback"],
                sigma=rc.get_sigma(),
            )

            network.decay_weights(
                decay_rate=DECAY_RATE,
                exclude_path=output["path_used"],
            )

            weights_snapshot = network.get_weights_snapshot()
            alerts = rc.monitor(
                weights=weights_snapshot,
                accuracy={"overall": round(correct / (i + 1), 4)},
            )
            if alerts:
                rc_warnings += len(alerts)

            if (i + 1) % 20 == 0:
                print(f"    問{i+1}: acc={correct / (i + 1):.1%}")

        except Exception as e:
            errors += 1

    final_weights = network.get_weights_snapshot()
    accuracy = correct / questions_done if questions_done > 0 else 0

    return {
        "init_weights": init_weights,
        "final_weights": final_weights,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": questions_done,
        "errors": errors,
        "rc_warnings": rc_warnings,
        "entropy_final": round(calc_entropy(final_weights), 4),
    }


def main():
    print("=== v14実験：v12同一タスクセットでの初期値比較 ===")
    print("タスクセット：generate_tasks()（v12と同一の固定100問）")
    print()

    # v12と同じ呼び出し
    tasks = generate_tasks()
    print(f"タスク数: {len(tasks)}")
    print()

    output_path = RESULTS_DIR / "v14_init_comparison.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for pattern_id in ["A", "C", "E"]:
            pattern = PATTERNS[pattern_id]
            print(f"--- パターン{pattern_id}：{pattern['name']} ---")
            print(f"  init={pattern['weights']}")

            start = time.time()
            result = run_trial(pattern["weights"], tasks)
            elapsed = time.time() - start

            result["pattern"] = pattern_id
            result["pattern_name"] = pattern["name"]
            result["elapsed_sec"] = round(elapsed, 1)

            all_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  結果: acc={result['accuracy']:.1%}  "
                  f"warnings={result['rc_warnings']}  ({elapsed:.0f}s)")
            print()

    # 最終比較
    print("=" * 60)
    print("v14実験 最終比較（v12同一タスクセット）")
    print("=" * 60)
    print(f"\n{'パターン':<12} {'名前':<30} {'正答率':<10} {'RC警告':<8}")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['pattern']:<10} {r['pattern_name']:<28} {r['accuracy']:.1%}      {r['rc_warnings']}")

    best = max(all_results, key=lambda r: r["accuracy"])
    print(f"\nベスト：パターン{best['pattern']}（{best['pattern_name']}）= {best['accuracy']:.1%}")
    print(f"v12ベースライン目標値：69%")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
