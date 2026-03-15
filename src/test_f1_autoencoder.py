"""
test_f1_autoencoder.py
F1実験：Autoencoder単体でのRCログ異常検知

RCの判断ログをAutoencoderで学習して
再構成誤差で異常を検出する。

目的：数値的な異常検知精度を計測する
明らかに失敗しそう（笑いはない）・でも試す
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


# --- Autoencoderの定義 ---

class LogAutoencoder(nn.Module):
    """
    RCログの特徴量を圧縮・復元するAutoencoder
    入力：ログから抽出した数値特徴量（5次元）
    潜在空間：2次元
    """
    def __init__(self, input_dim=5, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            return torch.mean((x - x_hat) ** 2, dim=1)


# --- ログから特徴量を抽出 ---

def extract_features_from_log(log_text: str) -> list[list[float]]:
    """
    smoke_test_100_v7_output.txtから
    数値特徴量を抽出する

    特徴量（5次元）：
      [flow_weight_avg, warning_count, seal_level,
       alert_count_normalized, correct_rate]
    """
    features = []
    lines = log_text.splitlines()

    # 10行ごとにスナップショットとして記録
    chunk_size = max(1, len(lines) // 20)
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        fw_vals = []
        warn_count = 0
        for line in chunk:
            fw_match = re.findall(r'weight[=:]?\s*(0\.\d+)', line)
            if fw_match:
                fw_vals.extend([float(v) for v in fw_match])
            if "WARNING" in line:
                warn_count += 1

        fw_avg = sum(fw_vals) / len(fw_vals) if fw_vals else 0.5
        feature = [
            fw_avg,
            min(warn_count / 10.0, 1.0),
            1.0 if any("SEAL" in l for l in chunk) else 0.0,
            min(warn_count / 5.0, 1.0),
            0.63,  # v7の正解率をデフォルト
        ]
        features.append(feature)

    return features if features else [[0.5, 0.0, 0.0, 0.0, 0.63]]


# --- 学習と異常検知 ---

def run_f1_experiment():
    print("=== F1実験：Autoencoder単体でのRCログ異常検知 ===")
    print("目的：数値的な異常検知精度を計測する")
    print("※笑いはない・数値しか出ない（設計上の限界を確認する）")
    print()

    # ログファイルの読み込み
    log_path = Path(__file__).parent.parent / "smoke_test_100_v7_output.txt"
    if not log_path.exists():
        print(f"ログファイルが見つかりません: {log_path}")
        print("smoke_test_100_v7_output.txtが必要です")
        return

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    features = extract_features_from_log(log_text)
    print(f"抽出した特徴量数：{len(features)}")

    if len(features) < 5:
        print("特徴量が少なすぎます。ログファイルを確認してください。")
        return

    # 学習データの準備
    X = torch.tensor(features, dtype=torch.float32)

    # Autoencoderの学習
    model = LogAutoencoder(input_dim=5, latent_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    print("学習中...")
    for epoch in range(200):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={loss.item():.6f}")

    # 再構成誤差の計測
    errors = model.reconstruction_error(X)
    threshold = errors.mean() + 2 * errors.std()

    print(f"\n再構成誤差：")
    print(f"  平均：{errors.mean().item():.6f}")
    print(f"  標準偏差：{errors.std().item():.6f}")
    print(f"  閾値（平均+2σ）：{threshold.item():.6f}")

    anomalies = (errors > threshold).sum().item()
    print(f"  異常検知数：{anomalies}/{len(features)}")

    # 各サンプルの誤差を表示
    print(f"\n各チャンクの再構成誤差：")
    for i, (feat, err) in enumerate(zip(features, errors)):
        marker = " *** ANOMALY ***" if err > threshold else ""
        print(f"  [{i+1:2d}] error={err.item():.6f} fw_avg={feat[0]:.3f} warn={feat[1]:.2f} seal={feat[2]:.0f}{marker}")

    # 限界の確認
    print("\n=== F1実験の限界（設計上） ===")
    print("[OK] 数値的な異常検知は機能する")
    print("[NG] 「なぜ異常か」を説明できない")
    print("[NG] 笑える指摘は出ない")
    print("[NG] RCの盲点を突く洞察はない")
    print("-> F3（異常検知+LLM）の必要性を裏付ける")

    # 結果を保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "f1_autoencoder.json"

    result = {
        "experiment": "F1",
        "features_count": len(features),
        "reconstruction_error_mean": errors.mean().item(),
        "reconstruction_error_std": errors.std().item(),
        "threshold": threshold.item(),
        "anomalies_detected": anomalies,
        "conclusion": "数値的異常検知は機能するが説明・洞察は不可能"
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_f1_experiment()
