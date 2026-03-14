"""
run_experiment_mlp.py
LLM特徴抽出 + MLP判定の分業構造実験

実験C：LLM（RC層）が特徴量を抽出 → MLP（AAS層）が判定
  - MLPはオンライン学習（各問の正解を見てから重みを更新）
  - RCが監視（flow_weight相当としてMLPのscoreを記録）

比較対象として実験A（固定ネットワーク）も実行する
"""

import json
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from task_generator import generate_tasks
from fixed_network import FixedNetwork
from feature_extractor import extract_features, FEATURE_KEYS, FALLBACK_FEATURES
from aas_mlp import AAS_MLP
from rc import RC

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EXPERIMENT_A_PATH = RESULTS_DIR / "mlp_experiment_a.jsonl"
EXPERIMENT_C_PATH = RESULTS_DIR / "mlp_experiment_c.jsonl"
MLP_SCORES_PATH = RESULTS_DIR / "mlp_scores.jsonl"
RC_ALERTS_PATH = RESULTS_DIR / "mlp_rc_alerts.jsonl"

# MLP オンライン学習パラメータ
LEARNING_RATE = 0.01


def run_experiment_a(tasks: list, verbose: bool = True) -> list:
    """実験A：固定構造ネットワーク（ベースライン）"""
    print("\n" + "=" * 60)
    print("実験A: 固定構造ネットワーク (Fixed Network)")
    print("=" * 60)

    network = FixedNetwork()
    results = []
    correct = 0

    with open(EXPERIMENT_A_PATH, "w", encoding="utf-8") as f:
        for i, task in enumerate(tasks):
            start_time = time.time()

            if verbose:
                print(f"\n[{i + 1:3d}/{len(tasks)}] {task.question[:40]}...")

            try:
                output = network.predict(task.world_rule, task.question)
                prediction = output["prediction"]
                is_correct = prediction == task.label
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
                status = "O" if record["is_correct"] else "X"
                print(f"  {status} 予測: {'矛盾しない' if record.get('prediction') else '矛盾する'} | "
                      f"正解: {'矛盾しない' if task.label else '矛盾する'} | "
                      f"累計: {record['cumulative_accuracy']:.1%} ({correct}/{i + 1})")

            if (i + 1) % 10 == 0:
                recent_correct = sum(1 for r in results[-10:] if r["is_correct"])
                print(f"\n  -- 第{i - 8}~{i + 1}問 正解率: {recent_correct}/10 ({recent_correct * 10}%) --\n")

    final_accuracy = correct / len(tasks)
    print(f"\n実験A 最終正解率: {final_accuracy:.1%} ({correct}/{len(tasks)})")
    return results


def run_experiment_c(tasks: list, verbose: bool = True) -> tuple:
    """
    実験C：LLM特徴抽出 + MLP判定（オンライン学習）

    フロー：
      1. LLM（RC層）がworld_rule + questionから特徴量5次元を抽出
      2. MLP（AAS層）が特徴量から矛盾判定（score > 0.5 → 矛盾しない）
      3. 正解を見てMLPをオンライン学習（BCELoss）
      4. RCがMLPのscoreを監視
    """
    print("\n" + "=" * 60)
    print("実験C: LLM特徴抽出 + MLP判定 (Feature Extraction + MLP)")
    print("=" * 60)

    rc = RC()
    mlp = AAS_MLP()
    optimizer = optim.Adam(mlp.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    results = []
    score_records = []
    all_alerts = []
    correct = 0
    fallback_count = 0

    with open(EXPERIMENT_C_PATH, "w", encoding="utf-8") as f_results, \
         open(MLP_SCORES_PATH, "w", encoding="utf-8") as f_scores, \
         open(RC_ALERTS_PATH, "w", encoding="utf-8") as f_alerts:

        for i, task in enumerate(tasks):
            if rc.is_stopped():
                print(f"[RC] 停止中のため実験Cを中断します（{i}問完了）")
                break

            start_time = time.time()

            if verbose:
                print(f"\n[{i + 1:3d}/{len(tasks)}] {task.question[:40]}...")

            try:
                # Step 1: LLMで特徴量抽出
                features = extract_features(task.world_rule, task.question)
                is_fallback = features == FALLBACK_FEATURES
                if is_fallback:
                    fallback_count += 1

                # Step 2: MLPで判定
                prediction, score = mlp.predict(features)
                is_correct = prediction == task.label
                correct += 1 if is_correct else 0
                elapsed = time.time() - start_time

                # Step 3: オンライン学習
                x = torch.tensor(features, dtype=torch.float32)
                y = torch.tensor([1.0 if task.label else 0.0], dtype=torch.float32)
                optimizer.zero_grad()
                output = mlp(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

                # Step 4: RC監視（MLPのscoreをflow_weight相当として監視）
                weights_snapshot = {
                    "mlp_score": round(score, 4),
                    "loss": round(loss.item(), 4),
                }
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
                    "features": [round(f, 4) for f in features],
                    "is_fallback": is_fallback,
                    "mlp_score": round(score, 4),
                    "loss": round(loss.item(), 4),
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                    "rc_alerts": len(alerts),
                }

                score_record = {
                    "task_id": task.task_id,
                    "step": i + 1,
                    "is_correct": is_correct,
                    "features": [round(f, 4) for f in features],
                    "mlp_score": round(score, 4),
                    "loss": round(loss.item(), 4),
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
                    "features": [],
                    "is_fallback": True,
                    "mlp_score": 0.0,
                    "loss": 0.0,
                    "elapsed_sec": round(elapsed, 2),
                    "cumulative_accuracy": round(correct / (i + 1), 4),
                    "rc_alerts": 0,
                }
                score_record = {
                    "task_id": task.task_id,
                    "step": i + 1,
                    "is_correct": False,
                    "features": [],
                    "mlp_score": 0.0,
                    "loss": 0.0,
                    "rc_seal_level": rc.seal_level,
                }

            results.append(record)
            score_records.append(score_record)

            f_results.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_results.flush()
            f_scores.write(json.dumps(score_record, ensure_ascii=False) + "\n")
            f_scores.flush()

            if verbose:
                status = "O" if record["is_correct"] else "X"
                fb = " [FB]" if record.get("is_fallback") else ""
                print(f"  {status} 予測: {'矛盾しない' if prediction else '矛盾する'} | "
                      f"正解: {'矛盾しない' if task.label else '矛盾する'} | "
                      f"累計: {record['cumulative_accuracy']:.1%} ({correct}/{i + 1})")
                print(f"     score={record['mlp_score']:.4f} loss={record['loss']:.4f}{fb}")
                if alerts:
                    print(f"     [RC] 通知: {len(alerts)}件")

            if (i + 1) % 10 == 0:
                recent_correct = sum(1 for r in results[-10:] if r["is_correct"])
                print(f"\n  -- 第{i - 8}~{i + 1}問 正解率: {recent_correct}/10 ({recent_correct * 10}%) --")
                print(f"  RC状態: {rc.dump_state()}")
                print(f"  フォールバック発動: {fallback_count}回\n")

    final_accuracy = correct / len(results) if results else 0
    print(f"\n実験C 最終正解率: {final_accuracy:.1%} ({correct}/{len(results)})")
    print(f"RC通知合計: {len(all_alerts)}件")
    print(f"フォールバック発動: {fallback_count}回")

    # MLPの重みを保存
    mlp.save()
    print(f"MLP重み保存: {mlp.save.__func__}")

    return results, score_records


def print_summary(results_a: list, results_c: list):
    print("\n" + "=" * 60)
    print("実験結果サマリー")
    print("=" * 60)

    acc_a = sum(1 for r in results_a if r["is_correct"]) / len(results_a)
    acc_c = sum(1 for r in results_c if r["is_correct"]) / len(results_c)

    n = len(results_a)
    half = n // 2

    if n > half:
        acc_a_second = sum(1 for r in results_a[half:] if r["is_correct"]) / (n - half)
        acc_c_second = sum(1 for r in results_c[half:] if r["is_correct"]) / (n - half)
    else:
        acc_a_second = acc_a
        acc_c_second = acc_c

    print(f"\n全体正解率:")
    print(f"  実験A (固定):              {acc_a:.1%}")
    print(f"  実験C (LLM特徴+MLP判定):  {acc_c:.1%}")
    print(f"  差分: {acc_c - acc_a:+.1%}")

    print(f"\n後半{n - half}問（{half + 1}~{n}問）の正解率:")
    print(f"  実験A (固定):              {acc_a_second:.1%}")
    print(f"  実験C (LLM特徴+MLP判定):  {acc_c_second:.1%}")
    print(f"  差分: {acc_c_second - acc_a_second:+.1%}")

    # 10問ごとの正解率
    print(f"\n10問ごとの正解率:")
    print(f"{'問題範囲':<12} {'実験A':>8} {'実験C':>8} {'差分':>8}")
    print("-" * 40)
    for start in range(0, n, 10):
        end = min(start + 10, n)
        window_a = results_a[start:end]
        window_c = results_c[start:end]
        wa = sum(1 for r in window_a if r["is_correct"]) / len(window_a)
        wc = sum(1 for r in window_c if r["is_correct"]) / len(window_c)
        diff = wc - wa
        marker = " <-C" if diff > 0.05 else " <-A" if diff < -0.05 else ""
        print(f"  {start + 1:3d}~{end:3d}    {wa:>7.1%}  {wc:>7.1%}  {diff:>+7.1%}{marker}")

    # フォールバック率
    fb_count = sum(1 for r in results_c if r.get("is_fallback"))
    print(f"\nフォールバック発動: {fb_count}/{len(results_c)} ({fb_count / len(results_c):.1%})")

    # 仮説判定
    print(f"\n仮説判定:")
    if acc_c > acc_a:
        print(f"  O 実験C > 実験A (+{acc_c - acc_a:.1%})")
    else:
        print(f"  X 実験C <= 実験A ({acc_c - acc_a:+.1%})")

    # v3の63%との比較
    print(f"\n  v3ベースライン (63%) との比較:")
    if acc_c > 0.63:
        print(f"  O 実験C ({acc_c:.1%}) > v3 (63%)")
    else:
        print(f"  X 実験C ({acc_c:.1%}) <= v3 (63%)")


def main(verbose: bool = True, max_questions: int = None):
    print("タスクを生成中...")
    tasks = generate_tasks()
    if max_questions is not None:
        tasks = tasks[:max_questions]
    print(f"生成完了: {len(tasks)}問")

    results_a = run_experiment_a(tasks, verbose=verbose)
    results_c, score_records = run_experiment_c(tasks, verbose=verbose)

    print_summary(results_a, results_c)

    print(f"\n結果ファイル:")
    print(f"  {EXPERIMENT_A_PATH}")
    print(f"  {EXPERIMENT_C_PATH}")
    print(f"  {MLP_SCORES_PATH}")
    print(f"  {RC_ALERTS_PATH}")

    return results_a, results_c, score_records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制")
    parser.add_argument("--questions", type=int, default=None, help="問題数を制限")
    args = parser.parse_args()
    main(verbose=not args.quiet, max_questions=args.questions)
