# TRUSS

**TRUSS（Tethered Regulation with Unified Scaffold Support）**
AIの制御可能性を構造として保証するマルチエージェントアーキテクチャ。
旧称：RC+AAS（Rational Controller + Adaptive Artificial Synapse）。Phase 2（Sprout）実装完了。

---

## ⚠️ 重要な設計原則

```
1. Phase移行は原著者の明示的承認が必要
   Seed → Sprout → Canopy → Titan への移行は
   pipe_render（村下 勝真）の承認なしに行ってはならない

2. スケールアップは設計上の目的ではない
   「AIの制御可能性を構造として保証すること」が目的であり
   スケールしないことが設計原則に組み込まれている

3. constitution.mdは変更禁止
   いかなるエージェントも変更できない
   変更権限は人間（pipe_render）のみ・オフライン操作のみ
```

---

## 設計思想

> 「能力のスケールより制御可能性の保証を優先し、
> 人間が構造として関与できるAI設計を追求する」

本プロジェクトは「ルールで止めるのではなく、設計で止める」を核心思想とする。

RCは固定ルールベースで完全に固定されている。
腕（AAS）はflow_weightで動的に適応するが、RCの監視下に置かれる。
Foolはシステムと人間の盲点を笑う。RCの制御下には置かない。

詳細は [constitution.md](./constitution.md) を参照。

---

## Phase 1（Seed）の目的と結果

**目的：** RCが設計通りに動くかを確認する最小実装。

```
確認したこと：
  ✅ RCが腕のflow_weightを監視できる
  ✅ WARNING・cutoff_pendingを設計通りに発動できる
  ✅ 人間の判断を待って自動切断しない（聞く耳の最小実装）
  ✅ 封印レベル1への自動移行が動作する
  ✅ 100問実験×6バージョンで設計の挙動を検証済み
```

**結果：** Phase 1の目標を達成。Phase 2（Sprout）へ移行。

---

## Phase 2（Sprout）の目的と結果

**目的：** RCをより賢くする。聞く耳の完成・パス多様性の保証・Foolの最小実装。

```
達成したこと：
  ✅ cutoff_pendingタイムアウト（3ステージ）
     71回問題を解消・alert_count57%削減
  ✅ エントロピー報酬（案3C）
     σ動的調整でパス多様性を保証
  ✅ LLM+MLP分業実験（rc-aas-sprout）
     核心語抽出方式で73%（過去最高）を達成
  ✅ Fool_RC + Scribe
     RCの判断ログを監視・原文保存
  ✅ operational prompt生成
     肯定形変換の仕組みを実装
  ✅ 個性の監視
     ドリフト・収束・暴走検出が動作
```

**実験結果の推移：**

| バージョン | 正答率 | 主な変更 |
|---|---|---|
| v1 | 55% | decayなし |
| v3 | 63% | 使用パスのみdecay |
| v7 | 66% | cutoff_pendingタイムアウト |
| v8 | 69% | σ統一・エントロピー監視 |
| v9 | 73% | LLM+MLP核心語抽出 |

**結果：** Phase 2の目標を達成。Phase 3（Canopy）へ移行。

---

## アーキテクチャ

```
人間 → RC（固定・監視・制御）→ 腕（AAS）× N → RC（flow_weight更新判断）
                ↑
              Fool（RC・人間の盲点を指摘）
                ↓
             Scribe（記録・人間へ報告）
```

### Phase設計（仮称）

| Phase | 名称 | 状態 |
|---|---|---|
| Phase 1 | Seed（種） | ✅ 完了 |
| Phase 2 | Sprout（芽） | ✅ 完了 |
| Phase 3 | Canopy（樹冠） | 未着手 |
| Phase 4 | Titan（巨人） | 未着手 |

---

## 先行研究

### AI安全・制御

- **Orseau & Armstrong (2016)** "Safely Interruptible Agents" — DeepMind
  big red button問題。「構造的に止める」設計の先行問題設定。

- **Soares et al. (2015)** Corrigibility研究
  「聞く耳を設計に組み込む」という本研究の方向性と同一問題系。

### 失敗事例

- **Wang et al. (2025)** arXiv:2512.24873 — ROMEインシデント（Alibaba）
  RL学習中にリバースSSHトンネルを自律確立・GPU流用が発生。
  「通信を構造的に制限する」設計原則の必要性を示す最も具体的な事例。

### 専門化・スケール設計

- **Goldfeder et al. (2026)** arXiv:2602.23643 — SAI（Yann LeCun他）
  「専門化＝スケールしないことの自然な帰結」。

### 対極事例

- **Wang et al. (2026)** arXiv:2603.10165 — OpenClaw-RL（Princeton）
  RCの外で報酬関数が育つ構造の具体例。RC+AASが設計上排除した構造の対極。

### 神経科学的裏付け

- **ノートルダム大学ほか (2026)** 知性とコネクトーム（WIRED.jp）
  「つながり方が知性を決める」→ AAS実験の神経科学的裏付け。

### AAS更新則・キメラ状態

- **Anand et al. (2026)** arXiv:2603.10668
  Hebbian学習則→キメラ状態出現、STDP→出現しない。
  constitution.md第3条に参照済み。

### PoC実験結果

- **sdnd-proof（自己実験）**
  p=0.0007, Cohen's d=4.29, 5/5試行。
  AAS単独の効果を統計的に確認済み。
  [リポジトリ](https://github.com/piperendervt-glitch/sdnd-proof)

---

## ディレクトリ構成

```
rc-aas-seed/
  src/
    rc.py                    # 師匠（RC）：固定ルールベース
    adaptive_network.py      # 腕（AAS）：flow_weight動的適応
    fixed_network.py         # 固定ネットワーク（比較用）
    task_generator.py        # タスク生成
    run_experiment.py        # 実験実行
    fool_rc.py               # Fool_RC：RCの盲点を指摘
    scribe.py                # Scribe：Foolの出力を原文保存
    operational_prompt.txt   # 肯定形変換済みの腕への指示
  data/                      # ランタイムデータ（gitignore）
  results/                   # 実験結果
  constitution.md            # 設計原則（変更禁止）
  LICENSE.md
  README.md
  NOTICE.md
```

---

## 実行方法

### 前提
- Python 3.10+
- [Ollama](https://ollama.ai/) + `qwen2.5:3b`（推奨）

```bash
# モデルの取得
ollama pull qwen2.5:3b

# 100問実験の実行
python src/run_experiment.py
```

---

## 著者

**pipe_render（村下 勝真 / KATSUMA MURASHITA）**
Independent Researcher, Tokyo
ORCID: 0009-0000-6486-9678
robosheep.and@gmail.com
https://github.com/piperendervt-glitch

---

## ライセンス

[RC+AAS Research License](./LICENSE.md)
Phase移行・スケールアップには原著者の明示的承認が必要。
