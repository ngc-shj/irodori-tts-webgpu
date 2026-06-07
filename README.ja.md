# irodori-tts-webgpu

[English](README.md) | 日本語

[Irodori-TTS](https://github.com/Aratako/Irodori-TTS)（ゼロショット音声クローン対応の
日本語フローマッチング TTS）を、**ブラウザの WebGPU だけで完全実行**します（サーバ側推論なし）。
PyTorch モデルを ONNX に書き出し、rectified-flow のサンプリングループ・CFG・トークナイズ・
duration ロジックは JavaScript で再実装し、[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)
（WebGPU EP）で動かします。

同一のランタイムコア（`runtime/pipeline.mjs`）が、ヘッドレス検証では Node（onnxruntime-node）、
ブラウザでは onnxruntime-web WebGPU として、コード変更なしで動作します。

## 動作の仕組み — オフライン書き出し vs ブラウザ内推論

**推論サーバは存在しません**。処理は3フェーズに分かれます。

| フェーズ | 場所 / タイミング | コスト |
| --- | --- | --- |
| **1. Export**（`export/`、Python） | 開発機または CI で**一度だけ**（モデル変更時のみ） | PyTorch → ONNX。静的な `artifacts/onnx/*.onnx` を生成。重い依存はすべてここ。 |
| **2. Serve**（`web/serve.py` や任意の静的ホスト / CDN / HF Hub） | デプロイ時 | **なし** — `.onnx` を配るだけ。計算はしない。 |
| **3. Inference**（ブラウザ、WebGPU） | 生成のたび、**クライアント側** | TTS 全体がユーザーの GPU で動く。テキストと参照音声は端末から出ない。 |

つまり「サーバ側」は一度きりの**ビルド**（フェーズ1）と**静的ファイル配信**（フェーズ2）だけ。
実際の音声合成は 100% ブラウザ内です。

```text
[一度]   export (Python) ──▶ artifacts/onnx/*.onnx ──▶ 任意の静的ホストに配置
[実行時] ブラウザが *.onnx を取得 ──▶ WebGPU で音声生成   (サーバ計算ゼロ)
```

## ステータス — エンドツーエンド検証済み

JS パイプライン全体が、公式 PyTorch ランタイムを**ビット忠実に**再現します
（ヘッドレス、同一入力での onnxruntime-node CPU vs PyTorch）。

| 段階 | 指標 |
| --- | --- |
| tokenizer（transformers.js 経由の llm-jp-3-150m） | id 完全一致 |
| text encoder | max\|Δ\| 4.5e-6 |
| speaker encoder | max\|Δ\| 2.0e-5 |
| duration → seqLen | 完全一致（139 == 139） |
| RF Euler ループ + CFG（DiT step） | latent max\|Δ\| 1.2e-4 |
| DAC decode | **波形 corr = 1.000000** |

## アーキテクチャ

```text
text ──tokenize(llm-jp)──▶ text_encoder.onnx ─┐
ref.wav ─decode/normalize─▶ dacvae_encoder.onnx ─▶ speaker_encoder.onnx ─┤
                                                  duration.onnx ─▶ seqLen │
                                                                          ▼
        seeded noise x0 ─▶  RF Euler ループ ×N  (各 step: dit.onnx, batch=3 CFG)
                                    │  v = v_cond + s_text·(v_cond−v_text⁻) + s_spk·(v_cond−v_spk⁻)
                                    ▼
                              latent ─▶ dacvae_decoder.onnx ─▶ 48kHz 波形
```

ONNX 化されるのは**順伝播グラフのみ**（`artifacts/onnx/*.onnx`）。制御フロー
（サンプリングループ、CFG 合成、スケジュール、トークナイズ、duration→length）はすべて
`runtime/pipeline.mjs` にあり、`irodori_tts/rf.py` + `inference_runtime.py` を写したものです。

### Export 時に解決した2つの障壁

- **複素 RoPE**（`view_as_complex`）は ONNX に書き出せない → 数学的に等価な実数値・
  完全シンボリックな実装（`export/rope_patch.py`）に置換。モンキーパッチで適用（upstream は無改変）。
- duration 予測器の **data-dependent guards** → export 時にバイパス。

Export は `torch.onnx.export(dynamo=True)` + `dynamic_shapes` を使い、バッチと全シーケンス長を
シンボリックに保ちます（可変テキスト + バッチ CFG に必須）。

## ディレクトリ構成

```text
export/      Python: ONNX export + parity + capture（Irodori-TTS に依存）
runtime/     pipeline.mjs — 環境非依存の推論コア
web/         index.html + app.mjs（ORT-web WebGPU）+ serve.py
tokenizer/   llmjp_tok/ — llm-jp/llm-jp-3-150m 用の高速 tokenizer.json
tests/       Node ヘッドレス parity テスト（onnxruntime-node）
artifacts/   ONNX モデル + capture データ（gitignore。export/ で生成）
```

> **トークナイザ注意:** Irodori-TTS-500M-v3 のチェックポイント設定は
> `text_tokenizer_repo = llm-jp/llm-jp-3-150m`（vocab 99574）であり、`config.py` の
> デフォルト `sbintuitions/sarashina2.2-0.5b` **ではありません**。llm-jp を使うこと。

## ブラウザアプリを動かす（ローカル macOS）

```bash
python3 web/serve.py    # 標準ライブラリのみ・venv 不要（macOS は python でなく python3）
# WebGPU 対応ブラウザ（Chrome/Edge。Safari はフラグ要）で http://127.0.0.1:8137/web/ を開く
# 参照音声の .wav を選び、テキストを入力、生成 を押す
```

UI には **fp32 / fp16** セレクタと、実機 WebGPU 上で DiT step + DAC decode を計測して
ms/step と外挿リアルタイム係数を表示する **計測** ボタンがあります。精度を切り替えて
再計測すれば比較できます（fp16 には fp16 アーティファクトが必要、下記参照）。

モデルは localhost から配信（fp32、計 ~2.3 GB）なのでダウンロードコストはなく、
すべて Metal バックエンドの GPU で動きます。

## アーティファクト再生成（自己完結 — Irodori-TTS の別 clone 不要）

`setup_env.sh` はローカル `.venv` を作り、`irodori_tts`・`dacvae`・`descript-audiotools`
を `--no-deps` で導入します（CUDA torch / torchcodec / silentcipher、および
`tensorboard → protobuf<4` のピンを巻き込まないため。後者は onnx/onnxscript 全体の
インストールを解決不能にする）。[`uv`](https://docs.astral.sh/uv/) が必要。成功すると
以下のコマンドを表示します。

```bash
bash export/setup_env.sh   # .venv + 依存 + tokenizer/llmjp_tok
```

### fp32 アーティファクト（必須） → `artifacts/onnx/`

```bash
.venv/bin/python export/export_dacvae_decoder.py
.venv/bin/python export/export_text_encoder.py
.venv/bin/python export/export_dit.py
.venv/bin/python export/export_rest.py        # speaker, duration, dacvae encoder
# 任意（tests/ が使う parity capture 用）:
.venv/bin/python export/golden.py --no-ref --out outputs/golden_noref   # 参照 wav
.venv/bin/python export/capture_sampler.py                              # -> artifacts/ref/
```

### fp16 アーティファクト（任意・ブラウザで高速） → `artifacts/onnx_fp16/`

fp16 は各コンポーネントごとに専用ステップで生成します（理由は下記 *fp16 — コンポーネント別*）。
DiT とデコーダは意図的に `convert_fp16.py` を**通しません**。

```bash
.venv/bin/python export/export_dit_fp16.py        # DiT — half() モデルから直接書き出し
.venv/bin/python export/convert_fp16.py           # dacvae encoder（事後 fp16 変換）
.venv/bin/python export/rewrite_convtranspose.py  # decoder: ConvTranspose -> Conv (-> dacvae_decoder_subpix.onnx)
.venv/bin/python export/convert_fp16_decoder_mixed.py \
    --in artifacts/onnx/dacvae_decoder_subpix.onnx \
    --out artifacts/onnx_fp16/dacvae_decoder.onnx # decoder fp16（Snake は fp32 維持）
```

Export スクリプトはモデル定義のために `irodori_tts`（インストール済みパッケージ）を import し、
ONNX 向けにモンキーパッチ（RoPE、guards）します（upstream は無改変）。`PYTHONPATH` も
外部 clone も不要です。

## ヘッドレス検証（Node）

```bash
npm install
node tests/ort_test.mjs       # onnxruntime-node が Python ORT とビット一致
node tests/loop_verify.mjs    # RF ループ + decode が PyTorch capture と corr 1.0
node tests/full_verify.mjs    # 全チェーンが PyTorch capture と corr 1.0
```

`loop_verify`/`full_verify` は `export/capture_sampler.py`（seed 固定の PyTorch 参照実行）
が生成する `artifacts/ref/` を必要とします。

## 自分のアプリに組み込む（git submodule）

推論コア [`runtime/pipeline.mjs`](runtime/pipeline.mjs) は**依存ゼロの単一 ES モジュール**で、
何も import せず、すべて（`ort`、生成済みセッション、トークナイザ）を注入する設計です。
そのため submodule として vendor 化し、自前の UI/サーバから駆動するのが容易です
（`web/app.mjs` はその呼び出し例の一つで、コピー元に使えます）。

**1. submodule を追加**（コミットにピン留め。API は素の JS でビルド不要）:

```bash
git submodule add https://github.com/ngc-shj/irodori-tts-webgpu vendor/irodori-tts-webgpu
git -C vendor/irodori-tts-webgpu checkout <commit>     # ピン留め
# 後で: git submodule update --remote   # ピンを進める
```

**2. アーティファクトとトークナイザを用意** — これらは**リポジトリに含まれません**
（`artifacts/` は gitignore、fp16 で ~0.65 GB / fp32 で ~2.3 GB）。`export/` で生成する
（上記 *アーティファクト再生成*）か自前でホストし、6つの `*.onnx`（+ `*.onnx.data`）と
`tokenizer/llmjp_tok/` を静的ファイルとして配信します。fp16 では、デコーダは **必ず**
Conv 書き換え版にしてください — naive な `convert_fp16.py` ではなく、*Decoder fp16* の
2ステップで生成します。

**3. 結線**（ブラウザ、WebGPU）。コンストラクタは `{ ort, sessions, tokenizer }` を取り、
`sessions` は6キーすべてが必要です:

```js
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.webgpu.mjs";
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6";
import { IrodoriTTS } from "./vendor/irodori-tts-webgpu/runtime/pipeline.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/";
env.allowRemoteModels = false; env.allowLocalModels = true;
env.localModelPath = "/tokenizer/";          // /tokenizer/llmjp_tok/* を配信

const base = "/models";                       // .onnx（+ .onnx.data）をホストする場所
const names = { text:"text_encoder", speaker:"speaker_encoder", duration:"duration",
                dit:"dit", dac:"dacvae_decoder", enc:"dacvae_encoder" };
const opt = (n) => ({ executionProviders:["webgpu"], graphOptimizationLevel:"all",
  externalData:[{ path:`${n}.onnx.data`, data:`${base}/${n}.onnx.data` }] });
const sessions = {};
for (const [k, n] of Object.entries(names))
  sessions[k] = await ort.InferenceSession.create(`${base}/${n}.onnx`, opt(n));

const tokenizer = await AutoTokenizer.from_pretrained("llmjp_tok");
const tts = new IrodoriTTS({ ort, sessions, tokenizer });

// refWav: Float32Array, モノラル, 48 kHz（リサンプルは自前で。web/app.mjs の fileToMono48k 参照）
const { audio, sampleRate, seqLen } = await tts.synthesize(text, refWav, 48000,
  { numSteps: 16, seed: 0 });                 // audio: Float32Array @48 kHz
```

fp16/fp32 の混在は、`base` をモデルごとに fp16 / fp32 フォルダへ向けて行います
（`web/app.mjs` の `baseFor` 参照）。**Node / ヘッドレス**では、同じコードが
`onnxruntime-node` + `executionProviders:["cpu"]` で動きます。モジュールは
`normalizeText`・`lufsNormalize`・`integratedLoudness` も再エクスポートしています。

## fp16 — コンポーネント別（WebGPU・M3 Pro 実測）

fp16 はコンポーネント単位で選択します（UI チェックボックス。`export/export_dit_fp16.py` +
`export/convert_fp16.py`）。各々を分離してブラウザで実測:

| コンポーネント | WebGPU での fp16 | fp32 比の速度 |
| --- | --- | --- |
| **DiT** | ✅ 安定 | ~168 vs ~234 ms/step（**1.4×**） |
| **encoder** | ✅ 安定 | 無視できる（1回のみ実行） |
| **decoder** | ✅ 安定（Conv 書き換え済み） | decode ~530 vs ~997 ms（**1.9×**） |

**推奨: 3つすべて fp16**（UI 既定） — **fp32 と聴感上同一**かつ高速。3.6 秒のクリップを
~2.3 秒で生成（RTF 0.65×）でリアルタイムを十分下回ります。（計測の見積もりは保守的で、
全 step を batch=3 CFG で計時しますが、スケジュール後半は batch=1 で走ります。）

### Decoder fp16 には ConvTranspose 書き換えが必要だった

デコーダを素朴に fp16 化すると **WebGPU でノイズ**になります（CPU/fp32 は正常）。これは
**数値の問題ではありません** — オフラインであらゆる方法でエミュレートしてもモデルは fp16 で
健全でした（fp16 重み、fp16 活性化、conv 内 fp16 累積まですべて corr ≈ 1.0。中間値の最大は
~13 で fp16 上限 65504 に遠く及ばない）。原因は **onnxruntime-web の WebGPU fp16
`ConvTranspose` カーネル**で、これが壊れています（fp16 `Conv` は正常）。op 単位の
mixed precision で切り分け、公開 issue
[microsoft/onnxruntime#26367](https://github.com/microsoft/onnxruntime/issues/26367)
/ [#26732](https://github.com/microsoft/onnxruntime/issues/26732) とも一致。ort-web
1.26.0 でも未修正です。decode コストはアップサンプリングの ConvTranspose が支配的なので、
単に fp32 に留めても高速化しません。

対策: `export/rewrite_convtranspose.py` が4つの ConvTranspose 層を、数学的に等価な
**`Conv` のみで構成した sub-pixel（polyphase）形**に置換します（`Conv` →
reshape/transpose/reshape の pixel-shuffle）。4層すべてが `kernel = 2·stride, pad = stride/2`
の規則形なので、各々が `Cout·s` チャンネルを出す kernel-3 Conv になり、長さ `Lin·s` に
シャッフルされます — 元の ConvTranspose と厳密一致を確認（corr 1.0）。これでデコーダ全体が
`Conv` のみになり、fp16 でクリーンに動きます。

再生成は上記 *アーティファクト再生成* の *fp16 アーティファクト* にあるデコーダ2ステップで
行います（`convert_fp16.py` はもうデコーダを触りません）。

> **検証:** 書き換えはオフラインで検証済み（2つの fp32 グラフ比較、corr 1.0 — ここでは
> CPU 比較が有効）。ただし **onnxruntime CPU は fp16→fp32 に upcast** するため WebGPU の
> fp16 経路は判定できません。fp16 の音質はブラウザ実機（計測/生成）で確認してください。

## 既知の制限 / TODO

- **VoiceDesign（caption/emoji スタイル）** と no-ref 経路はブラウザアプリ未対応
  （ベース 500M-v3 の音声クローンのみ）。
- **fp16 encoders**: text/speaker/duration グラフは依然 fp32（小さい。合計 ~650 MB）。

## ライセンス

本リポジトリの推論/ランタイムコードは [MIT License](./LICENSE) で公開しています。

モデル重みと Irodori-TTS のアーキテクチャは、それぞれの upstream ライセンスに従います:

| コンポーネント | 出典 | ライセンス |
| --- | --- | --- |
| Irodori-TTS（コード + 500M-v3 重み） | [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) · [model card](https://huggingface.co/Aratako/Irodori-TTS-500M-v3) | MIT |
| llm-jp-3-150m tokenizer | [llm-jp/llm-jp-3-150m](https://huggingface.co/llm-jp/llm-jp-3-150m) | Apache-2.0 |
| DACVAE | [facebookresearch/dacvae](https://github.com/facebookresearch/dacvae) | Apache-2.0 |
