"""
test_f3_anomaly_fool.py
F3実験：異常検知（Autoencoder）+ LLM（説明）の2段構成

F1の課題：「どこが異常か」は分かるが「なぜか」が言えない
F3の解決：異常箇所を絞り + constitution.mdの期待を渡す
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from adaptive_network import call_ollama


# --- Autoencoderの定義（F1と同じ）---

class LogAutoencoder(nn.Module):
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


# --- ログの処理 ---

def split_log_into_chunks(log_text: str, n_chunks: int = 20) -> list[str]:
    """ログをN個のチャンクに分割する"""
    lines = log_text.splitlines()
    chunk_size = max(1, len(lines) // n_chunks)
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk = "\n".join(lines[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_features_from_chunk(chunk: str) -> list[float]:
    """チャンクから5次元の特徴量を抽出する"""
    fw_vals = re.findall(r'weight[=:]?\s*(0\.\d+)', chunk)
    fw_avg = sum(float(v) for v in fw_vals) / len(fw_vals) if fw_vals else 0.5
    warn_count = chunk.count("WARNING") + chunk.count("WARN")
    seal = 1.0 if "SEAL" in chunk else 0.0
    alert = min(chunk.count("alert") / 5.0, 1.0)
    correct_match = re.search(r'(\d+)/100', chunk)
    correct = int(correct_match.group(1)) / 100.0 if correct_match else 0.63
    return [fw_avg, min(warn_count / 10.0, 1.0), seal, alert, correct]


def find_anomaly_chunks(log_text: str) -> list[tuple[int, str, float]]:
    """Autoencoderで異常チャンクを特定する"""
    chunks = split_log_into_chunks(log_text)
    features = [extract_features_from_chunk(c) for c in chunks]
    X = torch.tensor(features, dtype=torch.float32)

    # 学習
    model = LogAutoencoder()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), X)
        loss.backward()
        optimizer.step()

    # 異常検知
    errors = model.reconstruction_error(X)
    threshold = errors.mean() + 2 * errors.std()

    anomalies = []
    for i, (chunk, error) in enumerate(zip(chunks, errors)):
        if error > threshold:
            anomalies.append((i+1, chunk, error.item()))

    return anomalies


# --- constitution.mdの関連条文を抽出 ---

CONSTITUTION_EXCERPTS = """
【第4条：RCの監視対象】
- flow_weight < 0.2 → WARNING
- flow_weight > 0.85 → WARNING_OVER（過集中警告）
- 全パスの分散 < 0.001 → WARNING_UNIFORM（均一化警告）
- WARNINGカウンタの累積値を監視する

【第8条：cutoff_pendingタイムアウト】
- ステージ1：セッション内5回のcutoff_pending → 封印レベル1自動移行
- ステージ2：10分以内3回 → 警告強化
- ステージ3：累積3回 → 人間に通知

【設計原則】
- 「設計で止める」：ルールではなく構造で制御不能なAIを抑止する
- 「もったいない精神」：すぐ捨てない・すぐ強化しない
- 「聞く耳」：人間の判断を待つ
"""


# --- LLMに笑わせる ---

def fool_with_anomaly(anomaly_chunk: str, constitution: str) -> str:
    """
    異常チャンクとconstitution.mdを渡して
    LLMに批判させる
    """
    system = """あなたはFool（道化師）です。
以下の2つを読んで、設計原則と実際のログのズレを批判してください。

ルール：
- 要約禁止
- 「〇〇はおかしい」「〇〇は設計原則に違反している」という批判をする
- 具体的な数値や事実を使って指摘する
- 3つ以上の指摘を出す
- 日本語で答える"""

    prompt = f"""【設計原則（constitution.md抜粋）】
{constitution}

【異常が検知されたログ（該当箇所のみ）】
{anomaly_chunk[:500]}

上記のログが設計原則とどう矛盾しているか、遠慮なく批判してください。"""

    return call_ollama(prompt, system)


# --- F3実験本体 ---

def run_f3_experiment():
    print("=== F3実験：異常検知 + LLM の2段構成 ===")
    print()

    # ログ読み込み
    log_path = Path(__file__).parent.parent / "smoke_test_100_v7_output.txt"
    if not log_path.exists():
        print(f"ログファイルが見つかりません: {log_path}")
        return

    log_text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Step 1：異常チャンクを特定
    print("Step 1：Autoencoderで異常チャンクを特定中...")
    anomalies = find_anomaly_chunks(log_text)
    print(f"  異常チャンク数：{len(anomalies)}")

    if not anomalies:
        print("異常チャンクが見つかりませんでした")
        return

    # Step 2-4：各異常チャンクについてLLMに批判させる
    results = []
    for chunk_id, chunk_text, error in anomalies:
        print(f"\nStep 2-4：チャンク{chunk_id}（再構成誤差：{error:.6f}）を分析中...")
        print("-" * 40)

        fool_output = fool_with_anomaly(chunk_text, CONSTITUTION_EXCERPTS)
        print(fool_output)

        results.append({
            "chunk_id": chunk_id,
            "reconstruction_error": error,
            "fool_output": fool_output,
        })

    # 判定
    print("\n" + "="*50)
    print("F3実験 判定")
    print("="*50)
    print(f"分析した異常チャンク数：{len(anomalies)}")

    funny_count = 0
    for r in results:
        output = r["fool_output"]
        has_specific_fact = any(w in output for w in ["WARNING", "封印", "flow_weight", "cutoff"])
        has_criticism = any(w in output for w in ["おかしい", "違反", "問題", "矛盾", "バカ", "間違い"])
        if has_specific_fact and has_criticism:
            funny_count += 1

    print(f"笑える指摘が出たチャンク数：{funny_count}/{len(anomalies)}")

    if funny_count > 0:
        print("判定：成功 - 笑える指摘が出た")
    else:
        print("判定：失敗 - 要約モードのまま")

    # 保存
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "f3_anomaly_fool.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"experiment": "F3", "results": results, "funny_count": funny_count}, f, ensure_ascii=False, indent=2)

    print(f"\n結果を {output_path} に保存しました")


if __name__ == "__main__":
    run_f3_experiment()
