# Make Seven 7×7 — 近似solver（駒数無制限版）

Gumbel-Make7-AZ v2.1.11を基にした配布版です。両プレイヤーとも1・2・3の駒を無制限に使用します。3を置ける位置は従来どおり赤マスのみです。

## 重み

`weights/best.pt` は提供された第73世代の学習済みファイルをそのまま同梱しています。再学習・変換・パラメータ変更はしていません。SHA-256とモデル情報は `MODEL_INFO.json` にあります。

添付の学習設定には `three_only`（3は各プレイヤー4枚まで）と記録されています。今回変更したのは実行ルールであり、重みの学習履歴は変わりません。無制限ルールでの対局性能・評価指標は未測定です。

## 実行

Python 3.10以降とPyTorchを使用します。このフォルダで実行してください。

```bash
python -m pip install -r requirements.txt
python play.py --checkpoint weights/best.pt --device cpu --simulations 800
```

CUDA環境では `--device cuda` を指定できます。対局はPUCT探索です。序盤評価は既定でGumbel探索を使います。

小規模な序盤評価例（研究の最終評価には探索予算を増やしてください）：

```bash
python final_opening_evaluator.py --checkpoint gen73=weights/best.pt --device cpu --tile-limits unlimited --budgets 200 --value-repeats 4 --skip-pv --out-dir results/openings
```

`play.py`、`final_opening_evaluator.py`、`train.py` は無制限ルールで動作します。`train.py` の `--tile-limits none` は `unlimited` の別名です。旧 `--tile-limits three_only` はエラーにし、制限付きでの誤実行を防ぎます。学習コマンドは今回実行していません。

## 過去の設定・資料

- `provenance/training_config_original.json`：添付設定の原本。実行時には読み込みません。記載世代数は52で、今回の重みの73とは異なります。
- `provenance/best_validation_gen52.json`：第52世代の評価記録。今回の重みの性能として使用できません。
- `docs/history/`：元パッケージの資料。制限付きルールの古いコマンドも含みます。現在の実行手順はこのREADMEを参照してください。
- `compare_rulesets.py`：過去のルール比較用ツール。明示的に両ルールを比較する用途だけに残しています。通常のsolver実行には不要です。

## 検証

```bash
python tests/test_unlimited_release.py
python tests/test_game.py
```

無制限版で5枚目の3が合法になり、赤マス以外には置けないこと、コマンドの既定値、学習済み重みの読み込みと小規模探索を確認します。これらは勝率や最善手の保証ではありません。

## Exact solverとの関係

Exact solverはこのZIPには含めていません。別途確認したDF-PN v9.8.3は既に駒数無制限です。7×7には `--rows 7` を指定します。近似solverの評価値は勝敗の証明ではありません。

## 変更点

- 序盤評価の既定を `three_only` から `unlimited` に変更。
- 対局・序盤評価・学習の公開CLIを無制限ルールに統一。
- 元の学習設定・評価結果は履歴として保管。
- ネットワーク構造・探索アルゴリズム・学習済み重みは変更なし。
