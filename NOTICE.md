# NOTICE

## 設計思想

このプロジェクトは「制御不能なAIへの抑止」を
唯一の目的として設計されている。

## もったいない精神

> 「すぐ捨てない・急いで強化しない・元に戻せる余地を残す」

この原則が全設計に反映されている。

- **2段階閾値**（案14）：WARNING → cutoff_pending → 人間が判断。即切断しない。
- **確率的ゆらぎ**：急激な収束を防ぎ、探索の余地を残す。
- **Phase制限**：Seed段階でとどまり、次のPhaseへ急がない。

## 先行研究

- AAS PoC（sdnd-proof）：p=0.0007, Cohen's d=4.29, 5/5試行
- Orseau & Armstrong (2016) "Safely Interruptible Agents"
- Goldfeder et al. (2026) arXiv:2602.23643

## 原著者

pipe_render（村下 勝真 / KATSUMA MURASHITA）
robosheep.and@gmail.com
https://github.com/piperendervt-glitch

## 未実装（Phase 2以降）

- Fool（懐疑官）：エコーチェンバー防止
- Scribe（書記官）：永続的ログ記録
- 時間減衰（案7）：古いweightの自然減衰
- 非対称性問題の解法
