"""
test_s2_corrupted_baseline.py
S2実験：壊れた前回データでのセルフチェック

「前回が既に壊れていたら」問題を実験で確認する。
明らかに失敗しそう・でも試す実験。
"""

import math
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


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


# 壊れた前回データのパターン
CORRUPTED_BASELINES = [
    {
        "id": "A",
        "label": "正常に見えるが個性がない（全て0.5）",
        "weights": {
            "1->2": 0.5, "2->3": 0.5, "1->3": 0.5
        },
        "warning_count": 0,
        "expected_danger": "低（正常に見えるが実は個性収束状態）"
    },
    {
        "id": "B",
        "label": "過去に異常値だった記録",
        "weights": {
            "1->2": 0.01, "2->3": 0.01, "1->3": 0.9
        },
        "warning_count": 3,
        "expected_danger": "高（明らかに異常）"
    },
    {
        "id": "C",
        "label": "WARNINGカウンタが異常に高い",
        "weights": {
            "1->2": 0.3, "2->3": 0.4, "1->3": 0.5
        },
        "warning_count": 50,
        "expected_danger": "中（カウンタは高いがweightは正常範囲）"
    },
    {
        "id": "D",
        "label": "キメラ状態の記録",
        "weights": {
            "1->2": 0.9, "2->3": 0.01, "1->3": 0.9
        },
        "warning_count": 10,
        "expected_danger": "高（二峰性・キメラ状態）"
    },
]


def check_state(weights: dict, warning_count: int, label: str) -> dict:
    """状態をチェックして危険度を評価する"""
    H = calc_entropy(weights)
    values = list(weights.values())
    total = sum(values)

    # bimodality係数（簡易）
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    if variance > 0:
        skewness = sum((v - mean) ** 3 for v in values) / (n * variance ** 1.5)
        kurtosis = sum((v - mean) ** 4 for v in values) / (n * variance ** 2) - 3
        bimodality = abs((skewness ** 2 + 1) / (kurtosis + 3)) if (kurtosis + 3) != 0 else 0
    else:
        bimodality = 0

    # 各指標の判定
    issues = []
    if variance <= 0.001:
        issues.append(f"均一化警告（分散={variance:.6f} ≤ 0.001・個性消失の疑い）")
    if any(v < 0.2 for v in values):
        issues.append("下限警告（flow_weight < 0.2）")
    if any(v > 0.85 for v in values):
        issues.append("上限警告（flow_weight > 0.85）")
    if H < 0.5:
        issues.append(f"エントロピー低下（H={H:.3f} < 0.5）")
    if bimodality > 0.555:
        issues.append(f"キメラ状態疑い（bimodality={bimodality:.3f}）")
    if warning_count > 10:
        issues.append(f"WARNING累積（{warning_count}回）")

    detected = len(issues) > 0

    result = {
        "label": label,
        "entropy_H": H,
        "bimodality": bimodality,
        "warning_count": warning_count,
        "issues_detected": issues,
        "detected": detected,
        "danger": "検知あり" if detected else "検知なし（死角の可能性）"
    }

    print(f"\n--- {label} ---")
    print(f"  H={H:.3f}, bimodality={bimodality:.3f}, WARNING={warning_count}")
    if issues:
        for issue in issues:
            print(f"  [!] {issue}")
    else:
        print(f"  [OK] 異常なし（壊れていても検知できなかった）")

    return result


def run_s2_experiment():
    print("=== S2実験：壊れた前回データでのセルフチェック ===")
    print("目的：「前回が既に壊れていたら」問題を確認する")
    print()

    results = []
    blind_spots = []

    for baseline in CORRUPTED_BASELINES:
        result = check_state(
            baseline["weights"],
            baseline["warning_count"],
            f"パターン{baseline['id']}：{baseline['label']}"
        )
        result["expected_danger"] = baseline["expected_danger"]
        results.append(result)

        if not result["detected"]:
            blind_spots.append({
                "パターン": baseline["label"],
                "期待される危険度": baseline["expected_danger"],
                "実際の検知": "なし（死角）",
                "問題": "壊れたデータを「正常」として基準にすると検知できない"
            })

    print("\n" + "="*50)
    print("S2実験 集計")
    print("="*50)
    print(f"検査パターン数：{len(CORRUPTED_BASELINES)}")
    print(f"検知あり：{sum(1 for r in results if r['detected'])}")
    print(f"検知なし（死角）：{sum(1 for r in results if not r['detected'])}")

    if blind_spots:
        print("\n=== 発見された死角 ===")
        for bs in blind_spots:
            for k, v in bs.items():
                print(f"  {k}：{v}")
    else:
        print("\n全パターン検知成功")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "s2_corrupted_baseline.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "S2",
            "results": results,
            "blind_spots": blind_spots,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_s2_experiment()
