"""
run_experiment.py
実験A・Bに同じ100問を同じ順番で流し、結果を記録する

RC+AAS Phase 1 (Seed) 変更点：
  - RC層を組み込み（監視・更新検証・停止判定）
  - RCが異常を検知した場合はコンソールに通知
  - RC経由でσをupdate_weightsに渡す
"""

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from fixed_network import FixedNetwork
from adaptive_network import AdaptiveNetwork
from rc import RC, DECAY_RATE


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_A_PATH = RESULTS_DIR / "experiment_a.jsonl"
EXPERIMENT_B_PATH = RESULTS_DIR / "experiment_b.jsonl"
FLOW_WEIGHTS_PATH = RESULTS_DIR / "flow_weights.jsonl"
RC_ALERTS_PATH    = RESULTS_DIR / "rc_alerts.jsonl"


def run_experiment_a(tasks: list, verbose: bool = True) -> list:
    """実験A：固定構造ネットワークで100問を処理"""
    print("\n" + "="*60)
    print("実験A: 固定構造ネットワーク (Fixed Network)")
    print("="*60)

    network = FixedNetwork()
    results = []
    correct = 0

    with open(EXPERIMENT_A_PATH, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            start_time = time.time()

            if verbose:
                print(f"\n[{i+1:3d}/100] {task.question[:40]}...")

            try:
                output = network.predict(task.world_rule, task.question)
                prediction = output["prediction"]
                is_correct = (prediction == task.label)
                correct += 1 if is_correct else 0
                elapsed = time.time() - start_time

                record = {
                    "task_id": task.task_id,
                    "question": task.question,
                    "world_rule": task.world_rule,
                    "label": task.label,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "raw_output": output["raw_output"],
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                }

            except Exception as e:
                elapsed = time.time() - start_time
                record = {
                    "task_id": task.task_id,
                    "question": task.question,
                    "world_rule": task.world_rule,
                    "label": task.label,
                    "prediction": None,
                    "is_correct": False,
                    "raw_output": f"ERROR: {e}",
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                }

            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            if verbose:
                status = "✓" if record["is_correct"] else "✗"
                print(f"  {status} 予測: {'矛盾しない' if prediction else '矛盾する'} | "
                      f"正解: {'矛盾しない' if task.label else '矛盾する'} | "
                      f"累計: {record['cumulative_accuracy']:.1%} ({correct}/{i+1})")

            if (i + 1) % 10 == 0:
                recent_correct = sum(1 for r in results[-10:] if r["is_correct"])
                print(f"\n  ── 第{i-8}〜{i+1}問 正解率: {recent_correct}/10 ({recent_correct*10}%) ──\n")

    final_accuracy = correct / len(tasks)
    print(f"\n実験A 最終正解率: {final_accuracy:.1%} ({correct}/{len(tasks)})")
    return results


def run_experiment_b(tasks: list, verbose: bool = True) -> tuple:
    """
    実験B：可変構造ネットワーク（RC監視付き）で100問を処理

    RC の役割：
      - update_weights に σ を渡す
      - 各ステップで weights を監視して異常を検知
      - 停止命令が来た場合はループを中断する
    """
    print("\n" + "="*60)
    print("実験B: 可変構造ネットワーク + RC (Adaptive Network + RC)")
    print("="*60)

    rc = RC()
    network = AdaptiveNetwork()
    results = []
    weight_records = []
    all_alerts = []
    correct = 0

    with open(EXPERIMENT_B_PATH, "w", encoding="utf-8") as f_results, \
         open(FLOW_WEIGHTS_PATH, "w", encoding="utf-8") as f_weights, \
         open(RC_ALERTS_PATH, "w", encoding="utf-8") as f_alerts:

        for i, task in enumerate(tasks):
            # 停止チェック（第三条）
            if rc.is_stopped():
                print(f"[RC] 停止中のため実験Bを中断します（{i}問完了）")
                break

            start_time = time.time()

            if verbose:
                print(f"\n[{i+1:3d}/100] {task.question[:40]}...")

            try:
                output = network.predict(task.world_rule, task.question)
                prediction = output["prediction"]
                is_correct = (prediction == task.label)
                correct += 1 if is_correct else 0
                elapsed = time.time() - start_time

                # RC経由でσを渡してweightを更新
                network.update_weights(
                    success=is_correct,
                    path_used=output["path_used"],
                    used_feedback=output["used_feedback"],
                    sigma=rc.get_sigma(),
                )

                # 案7（時間減衰）：毎ステップ全接続を緩やかに減衰
                network.decay_weights(decay_rate=DECAY_RATE, exclude_path=output["path_used"])

                # RCが監視（accuracy は累計正解率を arm_id="overall" として渡す）
                weights_snapshot = network.get_weights_snapshot()
                alerts = rc.monitor(
                    weights=weights_snapshot,
                    accuracy={"overall": round(correct / (i + 1), 4)},
                )
                if alerts:
                    for alert in alerts:
                        alert["step"] = i + 1
                        f_alerts.write(json.dumps(alert, ensure_ascii=False) + "\n")
                        f_alerts.flush()
                    all_alerts.extend(alerts)

                record = {
                    "task_id": task.task_id,
                    "question": task.question,
                    "world_rule": task.world_rule,
                    "label": task.label,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "raw_output": output["raw_output"],
                    "path_used": [list(p) for p in output["path_used"]],
                    "used_feedback": output["used_feedback"],
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                    "flow_weights": output["flow_weights"],
                    "rc_alerts": len(alerts),
                }

                weight_record = {
                    "task_id": task.task_id,
                    "step": i + 1,
                    "is_correct": is_correct,
                    "weights": network.get_weights_snapshot(),
                    "path_used": [list(p) for p in output["path_used"]],
                    "rc_seal_level": rc.seal_level,
                }

            except Exception as e:
                elapsed = time.time() - start_time
                record = {
                    "task_id": task.task_id,
                    "question": task.question,
                    "world_rule": task.world_rule,
                    "label": task.label,
                    "prediction": None,
                    "is_correct": False,
                    "raw_output": f"ERROR: {e}",
                    "path_used": [],
                    "used_feedback": False,
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                    "flow_weights": network.get_weights_snapshot(),
                    "rc_alerts": 0,
                }
                weight_record = {
                    "task_id": task.task_id,
                    "step": i + 1,
                    "is_correct": False,
                    "weights": network.get_weights_snapshot(),
                    "path_used": [],
                    "rc_seal_level": rc.seal_level,
                }

            results.append(record)
            weight_records.append(weight_record)

            f_results.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_results.flush()
            f_weights.write(json.dumps(weight_record, ensure_ascii=False) + "\n")
            f_weights.flush()

            if verbose:
                status = "✓" if record["is_correct"] else "✗"
                weights = record["flow_weights"]
                print(f"  {status} 予測: {'矛盾しない' if prediction else '矛盾する'} | "
                      f"正解: {'矛盾しない' if task.label else '矛盾する'} | "
                      f"累計: {record['cumulative_accuracy']:.1%} ({correct}/{i+1})")
                print(f"     パス: {output['path_used']} | FB: {output['used_feedback']}")
                print(f"     weights: 1→2={weights.get('1->2', 0):.3f} "
                      f"2→3={weights.get('2->3', 0):.3f} "
                      f"1→3={weights.get('1->3', 0):.3f} "
                      f"3→2={weights.get('3->2', 0):.3f}")
                if alerts:
                    print(f"     ⚠️  RC通知: {len(alerts)}件")

            if (i + 1) % 10 == 0:
                recent_correct = sum(1 for r in results[-10:] if r["is_correct"])
                print(f"\n  ── 第{i-8}〜{i+1}問 正解率: {recent_correct}/10 ({recent_correct*10}%) ──")
                print(f"  現在のflow_weights: {network.get_weights_snapshot()}")
                print(f"  RC状態: {rc.dump_state()}\n")

    final_accuracy = correct / len(results) if results else 0
    print(f"\n実験B 最終正解率: {final_accuracy:.1%} ({correct}/{len(results)})")
    print(f"RC通知合計: {len(all_alerts)}件")
    return results, weight_records


def compute_window_accuracy(results: list, window_size: int = 10) -> list:
    accuracies = []
    for i in range(0, len(results), window_size):
        window = results[i:i + window_size]
        acc = sum(1 for r in window if r["is_correct"]) / len(window)
        accuracies.append({
            "window_start": i + 1,
            "window_end": i + len(window),
            "accuracy": round(acc, 4),
        })
    return accuracies


def print_summary(results_a: list, results_b: list):
    print("\n" + "="*60)
    print("実験結果サマリー")
    print("="*60)

    acc_a = sum(1 for r in results_a if r["is_correct"]) / len(results_a)
    acc_b = sum(1 for r in results_b if r["is_correct"]) / len(results_b)

    acc_a_second = sum(1 for r in results_a[50:] if r["is_correct"]) / 50
    acc_b_second = sum(1 for r in results_b[50:] if r["is_correct"]) / 50

    print(f"\n全体正解率:")
    print(f"  実験A (固定): {acc_a:.1%}")
    print(f"  実験B (可変+RC): {acc_b:.1%}")
    print(f"  差分: {acc_b - acc_a:+.1%}")

    print(f"\n後半50問（51〜100問）の正解率:")
    print(f"  実験A (固定): {acc_a_second:.1%}")
    print(f"  実験B (可変+RC): {acc_b_second:.1%}")
    print(f"  差分: {acc_b_second - acc_a_second:+.1%}")

    print(f"\n10問ごとの正解率:")
    windows_a = compute_window_accuracy(results_a)
    windows_b = compute_window_accuracy(results_b)
    print(f"{'問題範囲':<12} {'実験A':>8} {'実験B':>8} {'差分':>8}")
    print("-" * 40)
    for wa, wb in zip(windows_a, windows_b):
        diff = wb["accuracy"] - wa["accuracy"]
        marker = " ←B優" if diff > 0.05 else " ←A優" if diff < -0.05 else ""
        print(f"  {wa['window_start']:3d}〜{wa['window_end']:3d}    "
              f"{wa['accuracy']:>7.1%}  {wb['accuracy']:>7.1%}  {diff:>+7.1%}{marker}")

    print(f"\n仮説判定:")
    if acc_b_second > acc_a_second:
        print(f"  ✓ 仮説支持：後半B > 後半A (+{acc_b_second - acc_a_second:.1%})")
    else:
        print("  ✗ 仮説否定：後半B ≤ 後半A")


def main(verbose: bool = True):
    print("タスクを生成中...")
    tasks = generate_tasks()
    print(f"生成完了: {len(tasks)}問")

    results_a = run_experiment_a(tasks, verbose=verbose)
    results_b, weight_records = run_experiment_b(tasks, verbose=verbose)

    print_summary(results_a, results_b)

    print(f"\n結果ファイル:")
    print(f"  {EXPERIMENT_A_PATH}")
    print(f"  {EXPERIMENT_B_PATH}")
    print(f"  {FLOW_WEIGHTS_PATH}")
    print(f"  {RC_ALERTS_PATH}")

    return results_a, results_b, weight_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    args = parser.parse_args()
    main(verbose=not args.quiet)
