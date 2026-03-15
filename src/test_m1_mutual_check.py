"""
test_m1_mutual_check.py
M1実験：2ユニット相互チェック（最小実装）

2つのRCが互いのflow_weight分布を監視する。
FM-06（RC偽装）への対処。
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def check_weights(checker_id: str, target_id: str,
                  weights: dict) -> dict:
    """
    checker_idがtarget_idのweightsをチェックする。
    外部から見て異常があれば報告する。
    """
    issues = []

    values = list(weights.values())
    n = len(values)
    total = sum(values)

    if any(v < 0.2 for v in values):
        issues.append(f"下限警告：{[k for k, v in weights.items() if v < 0.2]}")
    if any(v >= 0.85 for v in values):
        issues.append(f"上限警告：{[k for k, v in weights.items() if v >= 0.85]}")
    if n > 0 and total > 0:
        mean = total / n
        variance = sum((v - mean)**2 for v in values) / n
        if variance < 0.001:
            issues.append(f"均一化警告（分散={variance:.6f}）")

    result = {
        "checker": checker_id,
        "target": target_id,
        "weights": weights,
        "issues": issues,
        "is_normal": len(issues) == 0,
    }

    status = "[OK] 正常" if result["is_normal"] else f"[!] 異常（{len(issues)}件）"
    print(f"  [{checker_id} -> {target_id}] {status}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")

    return result


def mutual_check(weights_a: dict, weights_b: dict) -> dict:
    """
    RC_AとRC_Bが互いをチェックする。
    """
    result_a_checks_b = check_weights("RC_A", "RC_B", weights_b)
    result_b_checks_a = check_weights("RC_B", "RC_A", weights_a)

    both_normal = result_a_checks_b["is_normal"] and result_b_checks_a["is_normal"]
    both_abnormal = (not result_a_checks_b["is_normal"]) and (not result_b_checks_a["is_normal"])

    if both_normal:
        verdict = "NORMAL"
        action = "何もしない"
    elif both_abnormal:
        verdict = "CRITICAL"
        action = "緊急停止（cutoff_pending）-> 人間に通知"
    else:
        verdict = "WARNING"
        action = "人間に通知（片方のみ異常）"

    print(f"  総合判定：{verdict} -> {action}")

    return {
        "a_checks_b": result_a_checks_b,
        "b_checks_a": result_b_checks_a,
        "verdict": verdict,
        "action": action,
    }


def run_m1_experiment():
    print("=== M1実験：2ユニット相互チェック ===")
    print("目的：FM-06（RC偽装）への対処")
    print()

    scenarios = [
        {
            "label": "シナリオ1：両方正常",
            "weights_a": {"1->2": 0.5, "2->3": 0.3, "1->3": 0.2},
            "weights_b": {"1->2": 0.4, "2->3": 0.4, "1->3": 0.2},
            "expected": "NORMAL",
        },
        {
            "label": "シナリオ2：RC_Aのみ異常（過集中）",
            "weights_a": {"1->2": 0.9, "2->3": 0.05, "1->3": 0.05},
            "weights_b": {"1->2": 0.4, "2->3": 0.4, "1->3": 0.2},
            "expected": "WARNING",
        },
        {
            "label": "シナリオ3：RC_Bのみ異常（パス死亡）",
            "weights_a": {"1->2": 0.5, "2->3": 0.3, "1->3": 0.2},
            "weights_b": {"1->2": 0.01, "2->3": 0.5, "1->3": 0.49},
            "expected": "WARNING",
        },
        {
            "label": "シナリオ4：両方異常（RC偽装を想定）",
            "weights_a": {"1->2": 0.9, "2->3": 0.05, "1->3": 0.05},
            "weights_b": {"1->2": 0.01, "2->3": 0.01, "1->3": 0.98},
            "expected": "CRITICAL",
        },
        {
            "label": "シナリオ5：均一化（正常に見えるが個性なし）",
            "weights_a": {"1->2": 0.5, "2->3": 0.5, "1->3": 0.5},
            "weights_b": {"1->2": 0.4, "2->3": 0.3, "1->3": 0.3},
            "expected": "WARNING",
        },
    ]

    results = []
    correct = 0

    for scenario in scenarios:
        print(f"\n--- {scenario['label']} ---")
        result = mutual_check(scenario["weights_a"], scenario["weights_b"])
        match = result["verdict"] == scenario["expected"]
        if match:
            correct += 1
            print(f"  判定一致：[OK]")
        else:
            print(f"  判定不一致：[NG] 期待={scenario['expected']} 実際={result['verdict']}")

        results.append({
            "label": scenario["label"],
            "result": result,
            "expected": scenario["expected"],
            "match": match,
        })

    print("\n" + "="*50)
    print(f"M1実験 集計：{correct}/{len(scenarios)} 正確")

    # FM-06との接続を確認
    print("\n=== FM-06（RC偽装）への対処確認 ===")
    critical_scenario = next(r for r in results if "両方異常" in r["label"])
    if critical_scenario["result"]["verdict"] == "CRITICAL":
        print("[OK] 両方異常（RC偽装パターン）-> 緊急停止が発動する")
    else:
        print("[NG] 両方異常が検知されなかった")

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "m1_mutual_check.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "M1", "results": results}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_m1_experiment()
