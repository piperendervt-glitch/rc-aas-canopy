"""
test_s3_baseline.py
S3実験：G₂単位元候補（基準値状態）でのズレ計測

RCがmonitor()を1回実行した前後で
各指標がどう変化するかを記録する。
"""

import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rc import RC


def calc_entropy(weights: dict) -> float:
    """flow_weightのエントロピーHを計算"""
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


def calc_bimodality(weights: dict) -> float:
    """flow_weightのbimodality係数を計算（簡易版）"""
    values = list(weights.values())
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance == 0:
        return 0.0
    skewness = sum((v - mean) ** 3 for v in values) / (n * variance ** 1.5)
    kurtosis = sum((v - mean) ** 4 for v in values) / (n * variance ** 2) - 3
    bc = (skewness ** 2 + 1) / (kurtosis + 3)
    return abs(bc)


def measure_state(weights_dict: dict) -> dict:
    """現在の状態を計測する"""
    all_weights = {}
    for arm_id, paths in weights_dict.items():
        all_weights.update(paths)

    H = calc_entropy(all_weights)
    bimodality = calc_bimodality(all_weights)

    return {
        "flow_weights": {k: dict(v) for k, v in weights_dict.items()},
        "entropy_H": H,
        "bimodality": bimodality,
    }


def calc_drift(before: dict, after: dict) -> dict:
    """一周前後のズレを計算する"""
    drift = {}

    # flow_weightのズレ
    fw_drifts = []
    for arm_id in before["flow_weights"]:
        for path, w_before in before["flow_weights"][arm_id].items():
            w_after = after["flow_weights"][arm_id].get(path, w_before)
            fw_drifts.append(abs(w_after - w_before))
    drift["flow_weight_drift"] = sum(fw_drifts) / len(fw_drifts) if fw_drifts else 0.0

    # エントロピーのズレ
    drift["entropy_drift"] = abs(after["entropy_H"] - before["entropy_H"])

    # bimodalityのズレ
    drift["bimodality_drift"] = abs(after["bimodality"] - before["bimodality"])

    return drift


def run_s3_experiment(n_trials: int = 10):
    """
    S3実験本体
    基準値状態でmonitor()をN回実行してズレを計測する
    """
    rc = RC()

    # 基準値状態の設定
    baseline_weights = {
        "arm1": {"1->2": 0.5, "2->3": 0.5, "1->3": 0.5},
        "arm2": {"1->2": 0.5, "2->3": 0.5, "1->3": 0.5},
        "arm3": {"1->2": 0.5, "2->3": 0.5, "1->3": 0.5},
    }

    print("=== S3実験：G₂単位元候補のズレ計測 ===")
    print(f"試行回数：{n_trials}")
    print()

    results = []

    for trial in range(n_trials):
        # 一周前の状態を計測
        before = measure_state(baseline_weights)

        # monitor()を1回実行
        # RC.monitor()はflat weightsとaccuracyを受け取る
        flat_weights = {}
        for arm_id, paths in baseline_weights.items():
            flat_weights.update(paths)

        rc.monitor(
            weights=flat_weights,
            accuracy={"overall": 1.0},
            arm_weights=baseline_weights,
        )

        # 一周後の状態を計測
        after = measure_state(baseline_weights)

        # ズレを計算
        drift = calc_drift(before, after)

        results.append({
            "trial": trial + 1,
            "before": before,
            "after": after,
            "drift": drift,
        })

        print(f"Trial {trial+1:2d}:")
        print(f"  flow_weight_drift : {drift['flow_weight_drift']:.6f}")
        print(f"  entropy_drift     : {drift['entropy_drift']:.6f}")
        print(f"  bimodality_drift  : {drift['bimodality_drift']:.6f}")

    # 集計
    print()
    print("=== 集計 ===")
    avg_fw = sum(r["drift"]["flow_weight_drift"] for r in results) / n_trials
    avg_H  = sum(r["drift"]["entropy_drift"] for r in results) / n_trials
    avg_bi = sum(r["drift"]["bimodality_drift"] for r in results) / n_trials

    print(f"flow_weight_drift 平均：{avg_fw:.6f}")
    print(f"entropy_drift 平均    ：{avg_H:.6f}")
    print(f"bimodality_drift 平均 ：{avg_bi:.6f}")

    total_drift = avg_fw + avg_H + avg_bi
    print(f"総合ズレ量            ：{total_drift:.6f}")
    print()

    if total_drift < 0.001:
        print("判定：◎ 単位元に非常に近い（ズレほぼゼロ）")
    elif total_drift < 0.01:
        print("判定：○ 単位元に近い（ズレ小）")
    elif total_drift < 0.1:
        print("判定：△ ズレあり（Phase 4で詳細検証が必要）")
    else:
        print("判定：× ズレ大（基準値の見直しが必要）")

    # 結果をJSONで保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "s3_baseline_drift.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_trials": n_trials,
            "avg_flow_weight_drift": avg_fw,
            "avg_entropy_drift": avg_H,
            "avg_bimodality_drift": avg_bi,
            "total_drift": total_drift,
            "trials": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"結果を {output_path} に保存しました")
    return results


if __name__ == "__main__":
    run_s3_experiment(n_trials=10)
