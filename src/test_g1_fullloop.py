"""
test_g1_fullloop.py
G1実験：基準値状態でのズレ計測（フルループ版）

S3実験はmonitor()単体でズレ=0だった。
G1実験はフルループ（タスク実行→更新→monitor()）でのズレを計測する。
Phase 4のG₂単位元数値確定の下準備。
"""

import math
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 基準値：全エッジ 0.5（AdaptiveNetwork初期値）
BASELINE_WEIGHTS = {
    "1->2": 0.5, "2->1": 0.5,
    "2->3": 0.5, "3->2": 0.5,
    "1->3": 0.5, "3->1": 0.5,
}

N_QUESTIONS = 10   # 1ループのタスク数
N_LOOPS = 5        # 何回ループするか


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


def calc_drift_from_baseline(weights: dict) -> dict:
    """基準値（全エッジ0.5）からのズレを計算する"""
    drifts = {}
    for path, w in weights.items():
        baseline = BASELINE_WEIGHTS.get(path, 0.5)
        drifts[path] = abs(w - baseline)
    return {
        "per_path": drifts,
        "mean": sum(drifts.values()) / len(drifts) if drifts else 0.0,
        "max": max(drifts.values()) if drifts else 0.0,
    }


def run_g1_experiment():
    print("=== G1実験：基準値状態でのズレ計測（フルループ版） ===")
    print(f"設定：{N_QUESTIONS}問 × {N_LOOPS}ループ")
    print(f"基準値：{BASELINE_WEIGHTS}")
    print()

    tasks = generate_tasks()
    results = []

    for loop in range(N_LOOPS):
        print(f"--- ループ{loop+1} ---")

        # 毎ループ新しいRC・ネットワークを基準値から開始
        rc = RC()
        network = AdaptiveNetwork()
        correct = 0

        # タスクを実行（フルループ：predict→update→decay→monitor）
        loop_tasks = tasks[loop * N_QUESTIONS : (loop + 1) * N_QUESTIONS]
        for i, task in enumerate(loop_tasks):
            if rc.is_stopped():
                print(f"  [RC] 停止中のため中断（{i}問完了）")
                break

            try:
                output = network.predict(task.world_rule, task.question)
                prediction = output["prediction"]
                is_correct = (prediction == task.label)
                correct += 1 if is_correct else 0

                # weight更新（RC経由σ）
                network.update_weights(
                    success=is_correct,
                    path_used=output["path_used"],
                    used_feedback=output["used_feedback"],
                    sigma=rc.get_sigma(),
                )

                # 時間減衰
                network.decay_weights(
                    decay_rate=DECAY_RATE,
                    exclude_path=output["path_used"],
                )

                # RC監視
                weights_snapshot = network.get_weights_snapshot()
                alerts = rc.monitor(
                    weights=weights_snapshot,
                    accuracy={"overall": round(correct / (i + 1), 4)},
                )
                if alerts:
                    print(f"  [RC] 通知{len(alerts)}件 (問{i+1})")

                status = "o" if is_correct else "x"
                print(f"  問{i+1}: {status}")

            except Exception as e:
                print(f"  問{i+1}: ERROR: {e}")

        # 現在のflow_weightを取得
        current_weights = network.get_weights_snapshot()

        # 基準値からのズレを計測
        drift = calc_drift_from_baseline(current_weights)
        H = calc_entropy(current_weights)

        print(f"  現在のweights: {current_weights}")
        print(f"  基準値からのズレ（平均）: {drift['mean']:.6f}")
        print(f"  基準値からのズレ（最大）: {drift['max']:.6f}")
        print(f"  エントロピーH: {H:.3f}")

        results.append({
            "loop": loop + 1,
            "weights": current_weights,
            "drift_per_path": drift["per_path"],
            "drift_mean": drift["mean"],
            "drift_max": drift["max"],
            "entropy_H": H,
            "accuracy": round(correct / len(loop_tasks), 4) if loop_tasks else 0,
        })

    # 集計
    print("\n" + "="*50)
    print("G1実験 集計")
    print("="*50)
    avg_drift = sum(r["drift_mean"] for r in results) / N_LOOPS
    max_drift = max(r["drift_max"] for r in results)
    avg_H = sum(r["entropy_H"] for r in results) / N_LOOPS

    print(f"ズレ（平均）: {avg_drift:.6f}")
    print(f"ズレ（最大）: {max_drift:.6f}")
    print(f"エントロピーH（平均）: {avg_H:.3f}")
    print()

    # 判定
    if avg_drift < 0.01:
        print("判定：○ ズレ小（単位元に近い）")
    elif avg_drift < 0.1:
        print("判定：△ ズレあり（Phase 4で詳細検証が必要）")
    else:
        print("判定：× ズレ大（基準値の見直しが必要）")

    # S3との比較
    print(f"\nS3実験（monitor()単体）との比較：")
    print(f"  S3：ズレ = 0.000000（設計上ゼロ）")
    print(f"  G1：ズレ = {avg_drift:.6f}（フルループ後）")
    print(f"  差分：{avg_drift:.6f}")

    # ループごとのズレ推移
    print(f"\nループごとのズレ推移：")
    for r in results:
        print(f"  ループ{r['loop']}: drift_mean={r['drift_mean']:.6f}  "
              f"drift_max={r['drift_max']:.6f}  H={r['entropy_H']:.3f}  "
              f"acc={r['accuracy']:.1%}")

    output_path = RESULTS_DIR / "g1_fullloop_drift.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "G1",
            "description": "基準値状態でのズレ計測（フルループ版）",
            "n_questions": N_QUESTIONS,
            "n_loops": N_LOOPS,
            "avg_drift": avg_drift,
            "max_drift": max_drift,
            "avg_entropy": avg_H,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_g1_experiment()
