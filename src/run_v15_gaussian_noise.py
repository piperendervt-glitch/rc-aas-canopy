"""
run_v15_gaussian_noise.py
v15実験：flow_weight初期値にガウシアンノイズを加える

設計：initial = clip([0.5,0.5,0.5] + N(0,0.1), 0.01, 0.9)
v12と同じgenerate_tasks()で100問、3回試行平均
比較対象：v14パターンC（56%）
"""

import json
import math
import random
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

N_TRIALS = 3
NOISE_SIGMA = 0.1
BASE_WEIGHT = 0.5


def generate_noisy_weights(seed: int) -> dict:
    rng = random.Random(seed)
    paths = ["1->2", "2->1", "2->3", "3->2", "1->3", "3->1"]
    return {p: round(max(0.01, min(0.9, BASE_WEIGHT + rng.gauss(0, NOISE_SIGMA))), 4)
            for p in paths}


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
    print("=== v15実験：ガウシアンノイズ初期値 ===")
    print(f"設計：clip(0.5 + N(0, {NOISE_SIGMA}), 0.01, 0.9)")
    print(f"試行数：{N_TRIALS}")
    print()

    tasks = generate_tasks()
    print(f"タスク数: {len(tasks)}（v12同一）")
    print()

    output_path = RESULTS_DIR / "v15_gaussian_noise.jsonl"
    trial_results = []

    with open(output_path, "w", encoding="utf-8") as f:
        for trial in range(N_TRIALS):
            init_weights = generate_noisy_weights(seed=200 + trial)
            print(f"--- 試行{trial+1} ---")
            print(f"  init={init_weights}")

            start = time.time()
            result = run_trial(init_weights, tasks)
            elapsed = time.time() - start

            result["trial"] = trial
            result["noise_sigma"] = NOISE_SIGMA
            result["elapsed_sec"] = round(elapsed, 1)

            trial_results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  結果: acc={result['accuracy']:.1%}  "
                  f"warnings={result['rc_warnings']}  ({elapsed:.0f}s)")
            print()

        # サマリー
        avg_acc = sum(r["accuracy"] for r in trial_results) / len(trial_results)
        avg_warnings = sum(r["rc_warnings"] for r in trial_results) / len(trial_results)
        std_acc = (sum((r["accuracy"] - avg_acc) ** 2 for r in trial_results) / len(trial_results)) ** 0.5

        summary = {
            "is_summary": True,
            "n_trials": N_TRIALS,
            "noise_sigma": NOISE_SIGMA,
            "avg_accuracy": round(avg_acc, 4),
            "std_accuracy": round(std_acc, 4),
            "avg_warnings": round(avg_warnings, 1),
            "individual_accuracies": [r["accuracy"] for r in trial_results],
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # 最終表示
    print("=" * 60)
    print("v15実験 結果")
    print("=" * 60)
    for r in trial_results:
        print(f"  試行{r['trial']+1}: acc={r['accuracy']:.1%}  warnings={r['rc_warnings']}")
    print(f"\n  平均正答率: {avg_acc:.1%} ± {std_acc:.1%}")
    print(f"  平均RC警告: {avg_warnings:.0f}")

    print(f"\n=== 比較 ===")
    print(f"  v14 パターンA（均一）:     50.0%")
    print(f"  v14 パターンC（特化）:     56.0%")
    print(f"  v14 パターンE（多様性）:   52.0%")
    print(f"  v15 ガウシアンノイズ:      {avg_acc:.1%} ± {std_acc:.1%}")

    if avg_acc > 0.56:
        print(f"\n判定：ガウシアンノイズがパターンCを上回った")
    elif avg_acc > 0.50:
        print(f"\n判定：ガウシアンノイズはA以上C以下")
    else:
        print(f"\n判定：ガウシアンノイズの効果なし")

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    main()
