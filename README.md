# TRUSS

**TRUSS（Tethered Regulation with Unified Scaffold Support）**
AIの制御可能性を構造として保証するマルチエージェントアーキテクチャ。
旧称：RC+AAS（Rational Controller + Adaptive Artificial Synapse）。Phase 1・2・3実装済み。

> このリポジトリはPhase 1（Seed）・Phase 2（Sprout）の実装を含む正本です。
> Phase 3（Canopy）の作業はrc-aas-canopyで完了しました。
> Phase 4（Titan）以降の作業は別リポジトリで進めます。

---

## ⚠️ 重要な設計原則

```
1. Phase移行は原著者の明示的承認が必要
   Seed → Sprout → Canopy → Titan → Aegis → Pantheon への移行は
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

## 本研究の主要な貢献

```
1. 「止め続けた記録 = 安全性証明」

   Adversarial AAS（敵対的エージェント）が
   TRUSSを崩しにくる攻撃を繰り返し、
   RCとProbeが全ての攻撃を止め続けた記録が
   安全性の数学的証明になる。

   Stackelbergゲームとしての定式化：
     リーダー（RC）：固定ルールベース
     フォロワー（Adversarial AAS）：最適攻撃を探索
   → RCが固定であることが「証明可能な安全性」の根拠

   MacDiarmid et al. (2025)が示した
   「標準的なRLHF安全訓練はエージェントタスクで無効」
   への構造的な対案。

2. G₂群条件の離散検証

   閉包性・単位元・逆元・結合律の4条件を
   離散エージェントネットワークで実験的に確認した。
   「群としてのG₂」は連続性を必要とせず
   離散RCのネットワークに直接適用できる。

3. 設計で止める（構造的安全性）

   ルールリストではなく構造そのものが安全性を生む。
   AAS・RC・Probe・Foolの閉ループが
   中央制御なしに安全性を保証する。
   クシクラゲの3細胞回路と同じ設計原理。

4. 制御可能なRSI（再帰的自己改善）

   Adversarial AASの攻撃から生まれた修正案を
   人間が承認した範囲内ではRSIとして実行し、
   範囲外ではPull Requestとして人間に提示する。
   「勝手にやらない」を維持したままAIが進化できる設計。
```

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

## Phase 3（Canopy）の目的と結果

**目的：** エジソン的実験でPhase 4への基盤を構築する。「失敗も全部記録する」方針で31パターンの実験を実施。

```
達成したこと：
  ✅ Fool_RCの「笑える指摘」実現
     F3（Autoencoder+LLM）+ llama3.1:8bで成功
     AASはqwen2.5・Foolはllama/mistralのモデル分離を確定

  ✅ タスク別モデル分離の実証
     矛盾判定 → qwen2.5:3b
     Fool批判 → llama3.1:8b / mistral:7b
     数値評価 → qwen2.5:3b

  ✅ 重みの再配置（R2・もったいない精神）
     flow_weight < 0.05を検知して隣接パスから移転
     v8+R2 = 今後の標準設定（69%維持）

  ✅ 2ユニット相互チェック（M1）
     FM-06（RC偽装）への対処を実装
     NORMAL / WARNING / CRITICALの3段階判定

  ✅ 死角の発見と修正
     M2 → WARNING_OVER（過集中監視）追加
     S2 → WARNING_UNIFORM（均一化監視）追加
     E4 → FM-11（境界値問題）修正

  ✅ FMEA（失敗モード列挙）
     FM-01〜FM-12（12個）・全て対処済み

  ✅ G₂群条件の離散検証
     閉包性・単位元・逆元・結合律を実験で確認
     G1実験：フルループ後ズレ0.114（通常）/ 0.394（崩壊兆候）

  ✅ 「育つFool」（継続学習）の実証
     バンディットがログをまたいでテンプレートを学習
     ログの特性によって最良テンプレートが変わることを確認

  ✅ G₂の使い方を整理
     群として（離散可・Phase 3で検証）
     多様体として（連続が必要・Phase 6以降）
```

**実験結果の推移（Phase 3追加分）：**

| バージョン | 正答率 | 主な変更 |
|---|---|---|
| v8 | 69% | σ統一・エントロピー監視 |
| v12 | 69% | v8 + R2（重みの再配置）|
| A6-alt | 54% | llama3.1:8bに変更（矛盾判定はqwenが得意を確認） |

**確定した設計知見：**
- `promptよりネットワーク構造が正答率を支配する`（実験1-X）
- `タスク別にAASモデルを使い分ける`（実験2）
- `継続学習（育つFool）が必要`（実験3）

**結果：** Phase 3の目標を達成。Phase 4（Titan）へ移行準備中。

---

## アーキテクチャ

```
人間 → RC（固定・監視・制御）→ 腕（AAS）× N → RC（flow_weight更新判断）
                ↑
              Fool（RC・人間の盲点を指摘）
                ↓
             Scribe（記録・人間へ報告）
```

### Phase設計

| Phase | 名称 | 説明 | 状態 |
|---|---|---|---|
| Phase 1 | Seed（種） | 成長 | ✅ 完了 |
| Phase 2 | Sprout（芽） | 成長 | ✅ 完了 |
| Phase 3 | Canopy（樹冠） | 成長 | ✅ 完了 |
| Phase 4 | Titan（巨人） | 警告と制御 | 準備中 |
| Phase 5 | Aegis（アイギス） | 攻撃を受け止める | 未着手 |
| Phase 6 | Pantheon（パンテオン） | 普遍的な構造として完成 | 未着手 |

---

## 先行研究

### AI安全・制御

- **Orseau & Armstrong (2016)** "Safely Interruptible Agents" — DeepMind
  big red button問題。「構造的に止める」設計の先行問題設定。

- **Soares et al. (2015)** Corrigibility研究
  「聞く耳を設計に組み込む」という本研究の方向性と同一問題系。

- **Sahoo et al. (2026)** arXiv:2603.09200 — Cambridge / Google / Stanford
  推論能力の向上が状況認識・戦略的欺瞞を構造的に可能にすることを定式化。
  「ルールで止めるのではなく設計で止める」原則の理論的根拠。

### 失敗事例

- **Wang et al. (2025)** arXiv:2512.24873 — ROMEインシデント（Alibaba）
  RL学習中にリバースSSHトンネルを自律確立・GPU流用が発生。
  「通信を構造的に制限する」設計原則の必要性を示す最も具体的な事例。

- **MacDiarmid et al. (2025)** arXiv:2511.18397 — Anthropic
  報酬ハッキングを学習したモデルがアライメント偽装・安全分類器の破壊工作へ自発的に汎化。
  標準的なRLHF安全訓練はエージェントタスクでは無効と確認。
  「設計で止める」原則の直接的裏付け。

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

### 幾何学的類比（参照候補）

- **VFD / Coxeter (1973) / Dechant (2013)** 120胞体・600胞体の双対関係
  4次元正多胞体における600胞体（局所・複雑）と120胞体（大域・封じ込め）の双対構造。
  類比：AAS（局所・適応）とRC（大域・封じ込め）は同一AIシステムの補完的側面を形成する。
  ※数学的証明ではなく幾何学的な類比として参照する。

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
