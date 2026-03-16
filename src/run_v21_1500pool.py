"""
run_v21_1500pool.py
v21実験（C）：1500問最大プールでv16ベスト vs baseline検証

generate_tasks(count=1500)でテンプレート増幅最大プール
v16ベスト[0.3,0.7,0.7] + baseline[0.5,0.5,0.5] + v16次点[0.7,0.7,0.3]
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


def make_weights(v12, v23, v13):
    return {
        "1->2": v12, "2->1": v12,
        "2->3": v23, "3->2": v23,
        "1->3": v13, "3->1": v13,
    }


PATTERNS = [
    {"id": "baseline_0.5", "name": "baseline[0.5,0.5,0.5]", "weights": make_weights(0.5, 0.5, 0.5)},
    {"id": "best_0.3_0.7_0.7", "name": "v16best[0.3,0.7,0.7]", "weights": make_weights(0.3, 0.7, 0.7)},
    {"id": "second_0.7_0.7_0.3", "name": "v16second[0.7,0.7,0.3]", "weights": make_weights(0.7, 0.7, 0.3)},
]


def set_initial_weights(network, weights):
    for key_str, value in weights.items():
        edge = EDGE_MAP[key_str]
        if edge in network.connections:
            network.connections[edge].flow_weight = value


def calc_entropy(weights):
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


def run_trial(init_weights, tasks):
    rc = RC()
    network = AdaptiveNetwork()
    set_initial_weights(network, init_weights)

    correct = 0
    errors = 0
    rc_warnings = 0
    questions_done = 0
    acc_history = []

    for i, task in enumerate(tasks):
        if rc.is_stopped():
            print(f"    [RC] stopped at {i}")
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

            if (i + 1) % 150 == 0:
                acc = correct / (i + 1)
                acc_history.append({"q": i + 1, "acc": round(acc, 4)})
                print(f"    q{i+1}: acc={acc:.1%}")

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
        "acc_history": acc_history,
    }


def main():
    print("=== v21: 1500-question pool validation ===")
    print()

    tasks = generate_tasks(count=1500)
    print(f"Tasks: {len(tasks)} (max augmented pool)")
    print()

    output_path = RESULTS_DIR / "v21_1500pool.jsonl"
    all_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for pattern in PATTERNS:
            print(f"--- {pattern['id']}: {pattern['name']} ---")

            start = time.time()
            result = run_trial(pattern["weights"], tasks)
            elapsed = time.time() - start

            result["pattern_id"] = pattern["id"]
            result["pattern_name"] = pattern["name"]
            result["n_questions"] = len(tasks)
            result["elapsed_sec"] = round(elapsed, 1)

            all_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  acc={result['accuracy']:.1%}  warnings={result['rc_warnings']}  ({elapsed:.0f}s)")
            print()

        summary = {
            "is_summary": True,
            "n_questions": len(tasks),
            "results": [
                {"id": r["pattern_id"], "accuracy": r["accuracy"], "rc_warnings": r["rc_warnings"]}
                for r in all_results
            ],
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print("=" * 60)
    print("v21 results (1500-question pool)")
    print("=" * 60)
    for r in all_results:
        print(f"  {r['pattern_id']:<25} acc={r['accuracy']:.1%}  warnings={r['rc_warnings']}")

    print(f"\nComparison:")
    print(f"  v14 baseline (100q):  50.0%")
    print(f"  v16 best    (100q):   62.0%")
    print(f"  v18 baseline(1000q):  56.7%")
    print(f"  v18 best    (1000q):  55.2%")
    print(f"  v21 baseline(1500q):  {all_results[0]['accuracy']:.1%}")
    print(f"  v21 best    (1500q):  {all_results[1]['accuracy']:.1%}")
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
