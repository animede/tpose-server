# tpose-server

1枚のキャラクター画像から、**3D化・自動リグ用のTポーズ立ち絵**
(正面 / 背面 / 左真横 / 右真横 / 左前45度 / 右前45度)を生成する専用サーバ。

- 用途: [image-3d] のマルチビュー入力(Hunyuan3D-2mv)と、rig-service の
  自動リグ・VRM化(Tポーズ前提)の前処理。
- モデル: **Qwen-Image-Edit-2511**(fp8-lightning、4steps)1系統のみ。
- 由来: 統合サーバ `diffusers-server` の `apps/tpose/` を、必要なモデル経路だけを
  持つ独立リポジトリとして切り出したもの(移植の範囲は下記「diffusers-server との関係」)。

生成物の例(入力: ぬいぐるみ写真1枚 → 正面・背面):
正面/背面ともTポーズ・同一デザイン・白背景で出力され、透過PNG(`_nobg.png`)も
併せて作れる。

---

## セットアップ

### venv(既定は diffusers-server と共有)

`diffusers` は **git 版(0.40.0.dev0)を検証済みの状態で使う**必要があるため、
既定では diffusers-server の venv をそのまま使う(`run.sh` の `DS_VENV`)。
新しく `pip install "git+https://github.com/huggingface/diffusers"` すると
別コミットが入りうる点に注意。

```bash
./run.sh                                   # /home/animede/diffusers-server/venv を使う
DS_VENV=/path/to/venv ./run.sh             # 別の環境を使う
```

独立した環境を作る場合の依存は `requirements.txt` を参照(torch と diffusers は
別途インストールが必要)。

### モデルの重み

`$DS_COMFYUI_DIR`(既定 `~/ComfyUI`)配下を優先的に参照し、無ければ HF Hub から
自動ダウンロードする(通常の HF キャッシュ `~/.cache/huggingface` に保存)。

| 用途 | ローカル(優先) | HF Hub |
|---|---|---|
| Edit transformer | `models/diffusion_models/qwen_image_edit_2511_bf16.safetensors` | `Comfy-Org/Qwen-Image-Edit_ComfyUI` |
| Lightning 4steps LoRA | `models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` | `lightx2v/Qwen-Image-Edit-2511-Lightning` |
| vae / text_encoder / tokenizer | — | `Qwen/Qwen-Image` |
| processor | — | `Qwen/Qwen-Image-Edit-2509` |
| 背景除去(anime) | — | `skytnt/anime-seg`(`isnetis.onnx`、176MB) |
| アップスケーラ | `DS_UPSCALE_MODEL` で任意のパス | `ai-forever/Real-ESRGAN`(`RealESRGAN_x2.pth`、64MB) |

## 起動

```bash
./run.sh                       # http://0.0.0.0:8610
DS_TPOSE_PORT=9000 ./run.sh    # ポート変更
```

ブラウザで `http://<host>:8610/` を開くと専用UIが出る。
diffusers-server(8601)と同時に起動できるが、**同じGPUを使うので同時生成すると
VRAM を食い合う**(このサーバのピークは約35GB)。

## API

| メソッド | パス | 説明 |
|---|---|---|
| POST | `/api/tpose/generate` | 生成ジョブ投入(multipart)。`{"job_id": ...}` を返す |
| GET | `/api/tpose/jobs/{id}` | ジョブ状態(ポーリング。1.5秒間隔を想定) |
| GET | `/api/tpose/jobs/{id}/images/{key}.png` | ビュー画像(表示用 inline)。`{key}_nobg` で透過版 |
| GET | `/api/tpose/jobs/{id}/download/{key}.png` | ビュー画像(ダウンロード) |
| GET | `/api/tpose/jobs/{id}/download.zip` | 全ビュー + 入力画像のZIP |
| GET | `/api/tpose/jobs/{id}/input.png` | 前処理後の入力画像 |
| POST | `/api/tpose/jobs/{id}/edit` | 生成後の追加編集(何度でも)。`use_reference=true` で元画像を2枚目の参照に渡せる(既定 false、ポーズを引き戻す事故があるため) |
| POST | `/api/tpose/jobs/{id}/upscale` | 生成済みビューを2048へアップスケール(Real-ESRGAN x2)。`views` / `target`(既定2048) |
| POST | `/api/tpose/jobs/{id}/undo` | 直前の編集を取り消す(1世代) |
| GET | `/api/tpose/views` | ビューID・プリセット一覧(UI用) |
| GET | `/api/status` | ロード状態・VRAM |
| GET | `/api/progress` | 生成の進捗(step粒度) |
| POST | `/api/unload` | VRAM 解放(`{"target": "edit"\|"all"}`) |
| POST | `/api/remove_bg` | 任意画像の背景除去(GPU不使用) |

### `POST /api/tpose/generate` の主なパラメータ

| 名前 | 既定 | 説明 |
|---|---|---|
| `image` | 必須 | 入力キャラクター画像(全身でも胸像でも可) |
| `views` | 全6ビュー | カンマ区切り(`front,back,left,right,front_left_45,front_right_45`) |
| `seed` | 0 | 0 = ランダム |
| `subject` | `auto` | `auto`(中立語彙)/ `animal`(毛皮・肉球)/ `human`(髪・手) |
| `palms` | `forward` | 手のひらを正面へ向ける(リグ用の標準)/ `natural` |
| `paw_pads` | `auto` | `auto` / `none` / 色名(例 `pink`) |
| `claws` | `none` | `none`(爪なし)/ `auto` / 自由記述 |
| `fur_color` | 空 | 毛色の色名。空のとき **`animal` に解決される場合だけ**入力画像から自動推定する(人物・中立では推定しない。CLAUDE.md 参照)。明示すれば中立でも使われる |
| `tail` | 空 | しっぽ形状の自由記述(未指定だとビューごとに形が揺れる) |
| `body` | 空 | 体型の自由記述。**脚が伸びる劣化への主要な対処** |
| `costume` | 空 | **背面から見た衣装**の自由記述(背面ビューのみ)。前開きのベスト等で「背中にも前開きが描かれる」場合の対処。丈・範囲まで書くこと |
| `extra_prompt` | 空 | 追記(爪抑制文より前に差し込まれる) |
| `recolor` | 空 | 生成直後に走る色調整Editの指示 |
| `remove_bg` | false | 透過版 `<key>_nobg.png` も作る |
| `bg_method` | `anime` | `anime`(キャラ向け)/ `rembg`(汎用) |

使用例:

```bash
# 生成(正面と背面だけ、透過版つき)
curl -X POST http://127.0.0.1:8610/api/tpose/generate \
  -F "image=@character.png" -F "views=front,back" -F "seed=42" \
  -F "subject=animal" -F "remove_bg=true"
# -> {"job_id":"3b4cde03a7f6"}

# 状態確認
curl http://127.0.0.1:8610/api/tpose/jobs/3b4cde03a7f6

# 個別ダウンロード
curl -O http://127.0.0.1:8610/api/tpose/jobs/3b4cde03a7f6/download/front.png
```

### アップスケール(2048)

生成後に「選択したビューを2048へアップスケール」を押すと、`<key>_2048.png`
(透過版があれば `<key>_2048_nobg.png` も)が追加されます。1024版は残ります。

**Real-ESRGAN x2(RRDBNet、16.7M params)による決定論的な拡大**で、拡散モデルでの
再生成ではないため髪型・衣装がドリフトしません。元画像を編集すると古い2048版は
自動で破棄されます。

同一スケール(2048同士)での実測:

| | ラプラシアン分散 | 最大エッジ勾配 |
|---|---|---|
| Lanczos 2048 | 11.0 | 666 |
| **Real-ESRGAN 2048** | **51.9** | **1017** |
| (参考)元 1024 | 108.0 | 960 |

Lanczos に対してエッジは明確にシャープ(ぼけない拡大)。構造も保存されている
(2048版を1024へ戻して元と比較して PSNR 38.0dB / 相関 0.99924)。
ただし **1024で描かれた情報量が2048分に増えるわけではない**(画素あたりの情報密度は
元の1024より低い)。より精細な絵が必要なら `DS_TPOSE_SIZE` を上げたネイティブ生成を
検討すること(ただしモデルは約1MP前提の学習で、構図が乱れやすい・再生成なので
ドリフトする、というリスクがある。未検証)。

### 3D化に渡すビュー

**image-3d(Hunyuan3D-2mv)へ渡すのは `front` / `back` / `left` / `right` の4枚**
(APIレスポンスの `for_3d: true` がその印)。`MVImageProcessorV2` のビュータグは
front/left/back/right に限られているため、**45度ビュー(`for_3d: false`)を
left/right スロットへ入れないこと**(カメラ事前分布を誤らせる。45度は参考出力)。

真横(`left` / `right`)は 2026-07-29 に追加した。Tポーズの真横投影は
「手前の腕がカメラを向いて完全に短縮する」難しい構図で、**角度で指示すると3/4止まり**に
なる(実測: 被写体bboxの幅/高さ比が 0.56〜0.84。真横なら 0.4前後)。
**「鼻が画面端を向く / 見える目と耳は片方だけ / 胴体はエッジオン / 両腕は強く短縮されて
肩と胴体の陰にほぼ隠れる」と“見えるもの”で指示する**ことで安定する(実測 0.38〜0.52)。
なお**腕の短縮の度合いは seed で変わる**(短い切り株状〜やや伸びた状態)。
腕が長く見えるときは seed を変えて生成し直すこと(1ビュー約16秒)。
詳しい実測値・失敗例は `tpose/prompts.py` の `_SIDE_ARMS_CLAUSE` のコメントにある。

## 主な環境変数

| 名前 | 既定 | 説明 |
|---|---|---|
| `DS_TPOSE_PORT` / `DS_TPOSE_HOST` | 8610 / 0.0.0.0 | 起動ポート(`run.sh`) |
| `DS_VENV` | diffusers-server の venv | 使用する Python 環境(`run.sh`) |
| `DS_TPOSE_SIZE` | 1024 | 生成解像度(正方形) |
| `DS_COMFYUI_DIR` | `~/ComfyUI` | モデル重みのローカル優先ディレクトリ |
| `DS_QUANT` | `fp8-lightning` | `gguf-q4_k_m` 等 / `none`(bf16のまま) |
| `DS_OFFLOAD` | `auto` | `none` / `group` / `group_lowvram` / `model_cpu` |
| `DS_EDIT_TE_OFFLOAD` | `auto` | text_encoder の CPU 退避(`on` / `off`) |
| `DS_QWEN_TILED_VAE` | `1` | 共有VAEの encode/decode を常時 tiled 化 |
| `DS_TERMINAL_PROGRESS` | `0` | 起動ターミナルへ進捗バーを出す |
| `DS_ANIME_SEG_PROVIDER` | `cpu` | 背景除去(anime)の ONNX 実行プロバイダ |
| `DS_UPSCALE_MODEL` | (HFから取得) | アップスケーラの重み(spandrel が読める ESRGAN系 .pth/.safetensors) |

## 実測(RTX PRO 6000 Blackwell、1024²、seed 固定)

| 項目 | 値 |
|---|---|
| 初回モデルロード | 約30〜39秒 |
| 1ビューの生成 | 5〜11秒 |
| 4ビュー1セット | 約70秒(ロード込み) |
| ピークVRAM | 35.0〜35.5GB(48GB専有でも収まる) |
| 背景除去(CPU) | 0.6〜1.4秒/枚 |
| 2048アップスケール | 2秒/枚(ピークVRAM 4.1GB) |

## リポジトリ構成

```
tpose-server/
├── app.py            # FastAPI(薄い。ルーティングと静的配信のみ)
├── engine/           # Qwen-Image-Edit 1系統(diffusers-server の families/qwen_image から Edit だけ抽出)
│   ├── paths.py      #   モデルパス・リポジトリ定数
│   ├── runtime.py    #   RuntimeConfig(DS_QUANT 等)
│   ├── state.py      #   シングルトン状態(shared / edit_group・ロック)
│   ├── shared.py     #   vae / text_encoder / tokenizer
│   ├── edit.py       #   QwenImageEditPlusPipeline のロードと Lightning 制御
│   ├── generate.py   #   run_edit()
│   └── lifecycle.py  #   unload() / get_status()
├── core/             # 汎用ユーティリティ(config / gpu / resolve / loaders / optimize / progress / bg)
├── tpose/            # アプリ層(プロンプト・ジョブ管理・API ルーター)
├── static/           # 専用UI(単一ページ)
└── outputs/          # 生成物(outputs/tpose/<job_id>/ 配下)
```

## diffusers-server との関係

移植したもの: `apps/tpose/` 一式、`families/qwen_image` の **Edit 経路のみ**、
`core/` の汎用部分、統合UIの Tポーズタブ。

移植していないもの: T2I / I2I / ControlNet / Inpaint / Layered / charsheet /
scene_angles / outpaint、FLUX.2 / Z-Image / LTX-2.3 / JoyAI / Mage-Flow の各ファミリー、
`core/registry.py`(ファミリー間の排他ロード。モデルが1つしかないため不要)、
`core/llm.py`(LLMプロンプト支援)。

そのため diffusers-server 側にあった「ファミリー切替時の自動 unload」「Edit 変種
(edit_angles 系)との相互排他」のロジックは、このリポジトリには存在しない。
実装上の注意点は [CLAUDE.md](CLAUDE.md) にまとめてある。

## ライセンス

Apache License 2.0([LICENSE](LICENSE))。モデルの重み・LoRA は各配布元の
ライセンスに従うこと。
