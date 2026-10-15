> **v2.1.4 note:** exact-guided fine-tuning now selects checkpoints with
> **group-held-out AUC** by default, so exact positions used by the auxiliary
> loss cannot inflate `best_validation.pt`. `validation.csv` logs both all-DB
> and held-out raw/symmetrised metrics. Opening values now use the same
> mirror-symmetrised evaluator as the production solver. See
> [`GUMBEL_V2_1_4.md`](GUMBEL_V2_1_4.md).


> **v2.1 note:** this package includes the Gumbel root search from v2 plus an
> auxiliary 21-action `Q(s,a)` head. The Q head is trained only from actually
> searched root actions and is not used directly by MCTS yet. See
> [`GUMBEL_V2_1.md`](GUMBEL_V2_1.md). Training also writes
> `opening_q_diagnostics.csv` with policy, Q-head, MCTS-Q, visits and improved
> policy for every opening action.

# Make 7 AlphaZero (7×7 / タイル無制限)

7×7 Make 7 を自己対局で学習する AlphaZero 実装です。GPU 11GB を前提に構成しています。

---

## 1. 実装したルール

| 項目 | 内容 |
|------|------|
| 盤面 | 7×7、重力落下（Connect 4 方式） |
| 手番 | GREEN 先手 → YELLOW 後手 |
| 手 | (タイル値 1/2/3) × (列 A–G) の 21 通り |
| **タイル供給** | **無制限**（1・2・3 いずれも何枚でも使える） |
| 3 の配置制限 | 赤マスに着地する場合のみ可 |
| 1・2 の配置制限 | なし（赤マスにも置ける） |
| 勝利 | 自分のタイルだけの連続列の合計が **ちょうど 7** |
| 引き分け | 49 手で盤面が埋まり勝者なし |

### 赤マス配置

DF-PN キャンペーンの exact DB（3,733 局面・延べ 88,172 配置）から実測で確定させたものです。
**3 が赤マス以外に置かれた例はゼロ**、**1・2 が赤マスに置かれた例は 2,410 件**あったため、
「3 は赤マス限定、1・2 は無制限」と確定しました。

```
row 6   .  .  .  .  .  .  .
row 5   .  .  ■  .  ■  .  .      C, E
row 4   .  ■  .  .  .  ■  .      B, F
row 3   .  .  .  ■  .  .  .      D
row 2   ■  .  .  .  .  .  ■      A, G
row 1   .  .  .  .  .  .  .
row 0   .  .  .  .  .  .  .
        A  B  C  D  E  F  G
```

左右対称なので、**ミラー反転は厳密な対称性**です。データ拡張と推論の対称化の両方で使っています。

### タイル無制限にした影響

本来の Make 7 は各プレイヤー 1×11, 2×11, 3×4 の計 26 枚です。無制限にすると

- 状態が**盤面だけで完結**する（手牌残数を状態に持たなくてよい）
- ネットワーク入力が 8 プレーンで済み、学習が安定する
- ただし**元のゲームとは別のゲーム**になる（後述の注意を参照）

---

## 2. インストール

```bash
pip install -r requirements.txt   # torch は環境に合う CUDA ビルドを選ぶこと
```

動作確認（GPU 不要、数秒）:

```bash
python tests/test_game.py       # ルール検証 + exact DB 3,733 局面の再現
python tests/test_mcts.py       # 詰み発見・受け・バッチ処理
python tests/test_selfplay.py   # 自己対局・リプレイバッファ・アリーナ
```

---

## 3. 学習

```bash
python train.py --run-dir runs/make7 --preset base --generations 100
```

主なオプション:

| オプション | 既定値 | 説明 |
|-----------|--------|------|
| `--preset` | `base` | `small` / `base` / `large` |
| `--games-per-generation` | 256 | 1世代あたりの自己対局数 |
| `--parallel-games` | 128 | 同時進行させる対局数（= GPU バッチサイズ） |
| `--simulations` | 200 | 1手あたりの MCTS シミュレーション |
| `--batch-size` | 1024 | 学習バッチ |
| `--train-steps` | 400 | 1世代あたりの勾配ステップ |
| `--arena-games` | 80 | 昇格判定の対局数 |
| `--promote-threshold` | 0.55 | この勝率を超えたら新ネットを採用 |
| `--symmetrise` | off | 推論をミラーと平均（精度↑、速度↓） |
| `--resume` | off | `best.pt` から再開 |

中断は Ctrl+C、再開は `--resume` です。

### チェックポイントの種類

| ファイル | 更新条件 | 用途 |
|---|---|---|
| `latest.pt` | 毎世代 | 学習中の重み + オプティマイザ |
| `best.pt` | アリーナ通過時 | 自己対局を生成する側 |
| **`best_validation.pt`** | **検証スコア更新時** | **最終成果物** |
| `buffer.npz` | 毎世代 | リプレイバッファ |

**アリーナと検証は別物です。** アリーナは「前世代より強いか」を測るので、
局面評価が最良のモデルを選ぶとは限りません。実際、ある run では検証スコアが
第40世代でピークに達した後、昇格は続いたまま40世代にわたって低下しました。

`best_validation.pt` は `--validation-select` で指定した指標（既定 `auc`）が
更新されたときだけ保存されます。ログに `*BEST*` が出た世代が該当します。
最終的な分析やEEの結論には、`best.pt` ではなく**こちらを使ってください**。

`best_validation.json` に、その世代の全指標が記録されます。

```bash
cat runs/make7f/best_validation.json
```

### 並列自己対局（`--workers`）

```bash
python train.py --run-dir runs/make7 --preset base --generations 100 \
    --no-amp --train-steps 160 --batch-size 512 \
    --parallel-games 256 --games-per-generation 256 --workers 4
```

**推論は親プロセスに集約されます。** ワーカーはネットワークを一切持たず、
木探索だけを行い、葉の局面を共有メモリ経由で親に渡します。親が全ワーカー分を
1つのバッチにまとめて1回だけ forward します。

この設計にした理由は、1080 Ti では **batch 64 も 256 も同じ 11ms**（レイテンシ律速）
だからです。ワーカーごとにモデルを持たせると batch 256 が 4つの batch 64 に分割され、
forward 回数が4倍になって GPU がシリアル化し、**かえって遅くなります**。

結果として:

- VRAM はワーカー数に依存しない（モデルも CUDA コンテキストも1つだけ）
- forward 回数は単一プロセスと同じ、バッチサイズも同じ
- Python の木探索だけが並列化される
- 1世代の時間が `GPU時間 + Python時間` から `max(GPU時間, Python時間/N)` に近づく

ワーカー数は **空きコア数から2〜3引いた値**にしてください（親プロセスと
GPU への転送にコアが要ります）。12コア空きなら `--workers 4` から試して、
ログ末尾の `batch` 表示が `--games-per-generation` に近ければ集約が効いています。

### 11GB GPU のための設定

| preset | ブロック | チャンネル | パラメータ | 学習バッチ | 概算 VRAM |
|--------|---------|-----------|-----------|-----------|-----------|
| small | 6 | 96 | 1.1M | 1024 | 約 1.5GB |
| **base** | **10** | **128** | **3.4M** | **1024** | **約 2.5GB** |
| large | 15 | 192 | 11.5M | 768 | 約 5.5GB |

7×7 は盤面が小さいのでメモリは余ります。**11GB では `base` を推奨**します。
容量よりも自己対局のスループットの方が律速になるためで、深くするより
`--parallel-games` を増やす方が総合的に効きます。

VRAM に余裕がある場合の増やし方（効果の大きい順）:

1. `--parallel-games 192` … GPU 占有率が上がる
2. `--simulations 400` … データの質が上がる（時間は約2倍）
3. `--preset large` … 最後の手段

### 実行時間の目安

11GB クラス（RTX 2080 Ti / 3060 12GB 相当）で `base` / 256局 / 200 sims:

```
自己対局   約 3 分
学習        約 30 秒
アリーナ    約 30 秒
─────────────────
1 世代      約 4 分
100 世代    約 6〜8 時間
```

ボトルネックは GPU ではなく **MCTS の Python ツリー処理**です（実測 12,000 sims/秒）。
複数 GPU より、CPU コア数の多いマシンの方が効きます。

---

## 4. 出力ファイル

`--run-dir` に以下が書かれます。

### `metrics.csv` — 収束判定用

```
generation, loss, policy_loss, value_loss, arena_score, arena_elo,
promoted, baseline_score, baseline_elo, green_win_rate, yellow_win_rate,
draw_rate, mean_plies, buffer, selfplay_seconds, train_seconds
```

**EE で「訓練が収束した」と主張するには、このファイルの
`baseline_elo` を世代に対してプロットし、水平になっていることを示すのが最も直接的です。**

### `openings.csv` — 初手の変遷

各世代について、自己対局で選ばれた初手の頻度（21列）と、
ネットワークの初手評価値（21列）が記録されます。
2D の評価が世代とともにどう動いたかを追えます。

### `best.pt` / `latest.pt`

チェックポイント。`best.pt` はアリーナを通過した最新のネットワークです。

---

## 5. 初手の分析

```bash
python analyse_openings.py --checkpoint runs/make7/best.pt \
    --games 400 --simulations 400
```

合法な初手 14 通りそれぞれについて、3 つの独立した指標を出します。

| 指標 | 内容 |
|------|------|
| raw value | 探索なしの value head の評価 |
| root visit share | ルートで MCTS が割いた訪問の割合 |
| **empirical score** | **その初手を強制した 400 局での GREEN のスコア（95% Wilson 信頼区間つき）** |

論文に書くべきは **empirical score** です。最後に上位2手が
95% 信頼区間で分離しているかどうかも判定されます。分離していなければ
「対局数が足りない」と明示されるので、そのまま結論にしてはいけません。

出力は `opening_analysis.csv` にも保存されます。

---

## 6. DF-PN exact DB との突き合わせ

```bash
python validate_vs_dfpn.py --checkpoint runs/make7/best.pt \
    --database exact_propositions_campaign.txt
```

DF-PN が厳密に証明した局面に対して、ネットワークの評価の符号が一致するかを
ply ごとに集計します。

### 重要な注意（EE に必ず書くべき点）

**DF-PN の DB は「タイル制限あり」の本来のゲームで証明されたものです。
本実装は「タイル無制限」変種です。** 両者で局面自体はどちらも到達可能ですが、
制限ありで GREEN 勝ちと証明された局面が、無制限でも GREEN 勝ちとは限りません。
制限を外すと**両者に**選択肢が増えるためです。

したがってこのスクリプトが測るのは **「2つのルール間の整合性」であって
「正解に対する精度」ではありません**。

- 一致率が高い → ネットワークが自己対局の癖ではなく Make 7 の実構造を学んでいる証拠
- 一致率が低い → 「ネットが弱い」のか「2変種が本当に乖離している」のか判別できない

ply 別の内訳を見ると判別の助けになります。供給制限が効き始めるのは終盤なので、
**乖離が ply とともに増えるなら「変種の違い」、ply によらず一様に低いなら「学習不足」**
と読めます。

---

## 7. 対局

```bash
python play.py --checkpoint runs/make7/best.pt --simulations 800 --show-thinking
```

`2D` のようにタイル値＋列で入力します。`#` が赤マス、括弧つきが YELLOW です。

---

## 8. 検証済みの内容

| 項目 | 結果 |
|------|------|
| exact DB 3,733 局面の再現 | 違法手 **0**、勝利判定の早発 **0** |
| 赤マス配置 | 88,172 配置から実測、違反 0 |
| ミラー対称性 | 合法手・方策・状態・価値すべてで一致を確認 |
| MCTS の詰み発見 | 情報ゼロの評価器でも訪問の **94%** を詰み手に集中 |
| MCTS の受け | 同条件で相手の詰みを阻止 |
| アリーナの公平性 | 同一プレイヤー同士でスコア 0.475（≒ 0.5） |
| 価値ターゲット | 決着局で手番ごとに符号が反転することを確認 |

### 参考：探索深度と引き分け率（情報ゼロの評価器）

| sims | GREEN | YELLOW | 引分 | 平均手数 |
|------|-------|--------|------|---------|
| 8 | 0.36 | 0.19 | 0.45 | 41.2 |
| 32 | 0.33 | 0.41 | 0.27 | 29.6 |
| 128 | 0.36 | 0.28 | 0.36 | 32.1 |
| 400 | 0.38 | 0.47 | 0.16 | 35.5 |

探索を深めると引き分けは減ります。学習済みネットワークではこの表と異なる値になるはずで、
**この表自体がベースラインとして比較対象になります**。

---

## 9. ファイル構成

```
make7/
  game.py       ルール・盤面・勝利判定・エンコード
  mcts.py       バッチ化 PUCT 探索
  model.py      ResNet (policy + value) と推論ラッパ
  selfplay.py   自己対局生成・ミラー拡張・リプレイバッファ
  arena.py      対戦・Elo・Wilson 信頼区間
train.py                学習ループ
analyse_openings.py     初手の統計分析
validate_vs_dfpn.py     DF-PN DB との突き合わせ
play.py                 人間との対局
tests/                  検証スクリプト
```

---

## 10. 結論の書き方について

AlphaZero が出すのは**証明ではなく統計**です。EE では次の区別を明示してください。

- 「2D は最適初手**である**」 … AlphaZero だけでは言えない
- 「第 N 世代のネットワークによる 400 局の自己対局において、2D の GREEN スコアは
  X（95% CI [a, b]）であり、他の初手より有意に高かった」 … 言える

`analyse_openings.py` が信頼区間と分離判定を出すのは、この区別を
そのまま書けるようにするためです。

## v2.1.2: exact-proposition diagnostics

See `GUMBEL_V2_1_2.md`. When `--validation-database` is supplied, training now
writes `exact_proposition_predictions.csv` in addition to `validation.csv`, so
individual DF-PN propositions can be tracked across generations. v2.1.2 also
uses rescaled completed Q in Gumbel policy improvement and lowers the default
Q-head supervision threshold to three root visits.

## v2.1.3: calibration + exact-guided fine-tuning

See `GUMBEL_V2_1_3.md`. v2.1.3 keeps AUC as the main validation criterion,
adds leakage-safe group-stratified threshold calibration on held-out proof
groups, logs raw and mirror-symmetrised validation side by side, and optionally
adds a small class/group-balanced exact value loss using calibration groups only.


## v2.1.5

See `GUMBEL_V2_1_5.md`. v2.1.5 adds replay-buffer inheritance, TD(lambda)-safe parity diagnostics, policy/opening entropy, and an arena + held-out-AUC dual promotion gate.


## v2.1.6
Fixes replay-buffer bucket reconstruction after load so balanced sampling remains valid when inheriting a buffer.

## v2.1.7: forced-opening branch coverage

v2.1.7 can reserve part of self-play for 2A/2B/2C/2D so low-prior opening
branches receive enough downstream training. The default CLI fraction is 0.25.
The forced empty-root sample still teaches value/Q but is masked out of policy
loss. Organic and forced opening counts are logged separately in
`opening_sources.csv`; `Hopen` uses only organic games so the intervention does
not hide policy collapse. See `GUMBEL_V2_1_7.md`.

## v2.1.8

See `GUMBEL_V2_1_8.md`. Adds organic/forced opening outcome diagnostics, raw-vs-training organic root policy logs, and 5% organic-root policy-target smoothing (`--policy-target-smoothing 0.05`).

## v2.1.9 — global policy-target smoothing

Policy-target smoothing now applies to every non-masked policy teacher rather
than only the organic empty root. Use `--policy-target-smoothing 0.05` (the
default). Search and action selection remain unsmoothed, and forced opening
roots remain masked from policy loss. Console diagnostics show `Hpi raw/train`;
`metrics.csv` adds `policy_target_entropy_raw` and
`policy_target_entropy_train`.

## v2.1.10 — final opening evaluator

Training is unchanged from v2.1.9.  `final_opening_evaluator.py` provides the
frozen-checkpoint 2A/2B/2C/2D evaluation used for the final EE analysis.  See
`FINAL_OPENING_EVALUATION.md`.

## v2.1.11 — simplified final opening evaluator

Training remains unchanged. The final evaluator now excludes stochastic
full-game robustness matches and focuses on high-budget post-opening value,
root-value repeat CIs/pairwise differences, deterministic PVs, and
cross-checkpoint reproducibility. See `FINAL_OPENING_EVALUATION.md` and
`GUMBEL_V2_1_11.md`.
