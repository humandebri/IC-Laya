# Laya独立開発の引き継ぎ

> Historical note: this document describes an earlier revision. Statements about missing implementations, tests, or measurements are not the current project status. Start with the [English local inference guide](GETTING_STARTED.md) and [current measurement index](README.md).

2026-09-22。元フォルダから必要な資料をコピー。元ファイルは保持。

## 開発の出発点

- 実装はIC-VerdictのopenJev導入前 `e701ad2ffcaee3070239996bf5178b74b009c1d8`。
- crates/laya-candle、decision-engine、core、tokenizer、pack変換・計測ツール、合成fixtureは配置済み。
- 独立したGitリポジトリ。共有workspace・外部path依存・submodule・remoteなし。
- 後続のIC-Verdict修正・最適化は未移植。古い実装を完成済みとは扱わない。

## 追加で引き継いだローカル資料

- checkpoints/laya-port/: checkpoint構成、tensor対応表の草案、調査レポート。weight本体ではない。Git対象外。
- .cache/upstream/: 取得済みLaya上流ソースとアーカイブ。参照用であり実行時依存ではない。Git対象外。上流LICENSEを保持。
- [移植手順](reference/MODEL_PORT.md)、[構造調査](reference/MODEL_PORT_FINDINGS.md)、[性能計測記録](reference/PERFORMANCE_MEASUREMENTS.md): 切り出し元に残っていた後続資料のコピー。相対リンクやパス、測定結果は元プロジェクト当時のもの。

## 次に確認すること

1. requirements-dev.txtの依存を専用の環境に用意する。
2. 合成fixtureによるPython/Rust検証を再実行する。
3. 構造調査に記載した固定revisionの実checkpointを取得し、native推論と参照実装のlogitsを比較する。
4. nativeで一致を確認した後、canisterの命令数・heapを実測する。

今回のPython検証は39件成功、2件がtorch未導入でエラー。Rust・実checkpoint推論・canister実行は未検証。
過去のartifactsは元プロジェクトの証跡であり、この独立環境での成功を示すものではない。
認証情報、既存canister状態、openJevモデル、ビルドキャッシュはコピーしていない。
