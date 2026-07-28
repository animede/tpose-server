# CLAUDE.md — tpose-server

このリポジトリで作業する際に知っておくべき注意点。統合サーバ `diffusers-server` の
CLAUDE.md から、**Tポーズ生成に関係する知見だけ**を抜き出して再構成したもの
(元番号は参照のため括弧で残してある。より詳しい実測ログは
`/home/animede/diffusers-server/CLAUDE.md` を参照)。

## まず読むもの

1. `README.md` — セットアップ・API一覧・実測値
2. `tpose/prompts.py` の冒頭コメント — **プロンプト設計の根拠が実測値つきで全部ここにある**
   (爪・後足の指・後頭部の黒髪化・体型ドリフト・手のひらの向き。試して効かなかった
   対処も記録してある)
3. `engine/edit.py` — モデルのロード手順(順序に意味がある)

## アーキテクチャの要点

- モデルは **Qwen-Image-Edit-2511(fp8-lightning、4steps)1系統だけ**。
  diffusers-server の `core/registry.py`(FamilyRegistry によるファミリー間の排他
  ロード)は移植していない。ロードは `engine.load()`、生成は `engine.generate_edit()`。
- GPU は `core.gpu.generation_lock` 1本で排他(同時1件)。tpose ジョブ同士は
  `tpose/jobs.py` の `current_job_id` で同時1件に制限する。
- 生成 → `outputs/<mode>_<ts>_<uuid>.png` に保存 → ジョブ側が
  `outputs/tpose/<job_id>/<view>.png` へコピーする、という2段構え(元実装のまま)。

## 必ず守ること

1. **Edit は `QwenImageEditPlusPipeline` を使い、画像は必ずリストで渡す**(旧知見2番)。
   `pipe(image=[img1, img2, ...], ...)`。単体画像で渡すと Plus 条件付けが効かず
   キャラクターの同一性が崩れる(1枚でも `[img]`)。参照画像は最大3枚
   (`MAX_EDIT_IMAGES`)。tpose の2段目は `[生成した正面, 元画像, (しっぽ参照)]`。

2. **text_encoder の GPU 復帰は `pipe(...)` を呼ぶ直前に行う**(旧知見23番、最重要)。
   `QwenImageEditPlusPipeline.__call__()` は `encode_prompt()` を呼ぶ**前**に
   `device = self._execution_device` を確定させ、その device をエンコードへ渡す。
   `_execution_device` は「最初に見つかった nn.Module の現在地」なので、text_encoder が
   CPU に残ったまま呼ぶと device が `"cpu"` に解決され、あとから GPU へ戻しても
   `input_ids`(cpu)と `embed_tokens.weight`(cuda)の食い違いで落ちる。
   そのため `core/optimize.py` は
   - `enable_text_encoder_cpu_offload(pipe)`: encode_prompt を**ラップ**して
     「冒頭でGPUへ戻す + 呼び出し後にCPUへ退避」する(冒頭の復帰は旧知見31番。
     negative 指定時にエンコードが2回連続で呼ばれるため必須)
   - `ensure_text_encoder_on_gpu(pipe)`: **呼び出し側が `pipe(...)` の直前に明示的に呼ぶ**

   という2段構成になっている。新しい生成経路を足すときも必ずこの順序を守ること。

3. **`_load_*_locked` 系はロック非再入**(旧知見14番)。`engine/state.py` の `lock` は
   `threading.Lock()`(非再入)。呼び出し側が既に保持している前提なので、内部で
   `with state.lock:` を書くとデッドロックする。

4. **fp8-lightning の fuse 手順を変えない**(旧知見12番)。`core/loaders.py` の
   `fuse_lightning_lora_and_cast_to_fp8()`:
   - `transformer.load_lora_adapter()` には `prefix="transformer"`(既定値)を使う。
     `prefix=None` にすると `Target modules {...} not found` になる。
   - `enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)`
     は呼んだ瞬間に圧縮される(遅延評価ではない)。直後の `empty_cache()` で
     bf16 分の VRAM が実際に返る(実測: fuse後 ~38GB → casting後 ~19GB)。
   - fuse 後は LoRA を無効化できない(`lightning_merged=True` は常に True を返す仕様)。

5. **transformer を丸ごと CPU へ退避する実装は書かないこと**(旧知見33番)。
   ホストRAM が枯渇してスワップ暴走 → システムフリーズ → OOMキラーで開発環境ごと
   強制終了、という事故を過去に2回起こしている。VRAM が足りない場合は素直に
   CUDA OOM で落とすか、block-level group offloading を使う。
   (text_encoder だけの CPU 退避(2番)は別物で、これは安全。)

6. **`pipe.set_adapters([])` は空リストで KeyError になる**(旧知見13番、diffusers のバグ)。
   無効化には `pipe.disable_lora()` を使う。

7. **processor は `AutoProcessor` ではなく `Qwen2VLProcessor` を明示的に使う**(旧知見10番)。
   transformers 5.x では `AutoProcessor` が `preprocessor_config.json` の
   `processor_class` を解決できず、画像処理を持たない素の `Qwen2Tokenizer` へ
   フォールバックすることがある(生成時に `KeyError: 'pixel_values'`)。

8. **GGUF 量子化 transformer には LoRA を適用できない**(旧知見11番)。PEFT が
   `GGUFLinear` を認識しない。`engine/edit.py` は try/except で捕捉して
   `lora_available=False` にし、steps/cfg を通常品質側(30steps/cfg4.0)へ
   フォールバックする。`DS_QUANT=gguf-q4_k_m` を使うときはこの挙動になる。

9. **共有VAE は常時 tiled**(`DS_QWEN_TILED_VAE=1` 既定、旧知見56番)。
   `AutoencoderKLQwenImage._encode()/_decode()` が `use_tiling` を尊重してタイル経路へ
   切り替わる。大きなキャンバスの encode で `Tried to allocate 4.45 GiB` 系の OOM を
   起こした実績があるため既定で有効。`0` で旧動作に戻せる。

10. **sm_120(Blackwell)では sage 系 attention backend が使えない**(旧知見21番)。
    "no kernel image is available" になる。`core/config.py` の
    `KNOWN_GOOD_ATTN_BACKENDS` / `KNOWN_BAD_ATTN_BACKEND_PREFIXES` に反映済み。

11. **venv は diffusers-server と共有する**(旧知見5番・6番)。`diffusers` は git 版
    (0.40.0.dev0)を検証済みの状態で使っている。新しく
    `pip install "git+https://github.com/huggingface/diffusers"` すると別コミットが
    入りうる。また `/home/animede/comfy-env` は稼働中の ComfyUI と共有しているので
    **直接 `pip install` しないこと**。

## プロンプト設計の教訓(Tポーズ固有、旧知見57番)

詳細な実測値・試して効かなかった対処は `tpose/prompts.py` の冒頭コメントにある。
要点だけ:

- **末尾の1文が最も強く効く。** 効かせたい指示は末尾に置き、その後ろに別の句を足すと
  既存の指示が壊れる(爪抑制文の後ろに何か足すと爪が戻る)。`extra_prompt` を
  爪抑制文より前へ差し込んでいるのはこのため。
- **否定文は逆効果。**「〜しない」と書くと文中の語が誘引になる(`without claws` で
  爪が残り、`do not turn the head into human hair` で黒髪化が悪化した)。
  肯定形で書くか、**語彙そのものを変える**(視点句の "head" → "fur")。
- **相対表現より具体的な色名。**「同じ毛色で」では効かず、`cream white` のような
  色名を書くと効く。だから入力画像から毛色を自動推定している
  (`tpose/generate.py` の `sample_fur_color()`。rembg はCPU処理なので**GPUロックを
  取る前に**呼ぶ)。
- **部位は名指しする。**「all four paws」のような総称ではモデルは手にしか適用しない
  (後足の爪が残った)。"hind feet" と具体的に書く。
- **「〜を向ける」だけの指示は弱い。**「その結果として何が見えるか」まで書くと安定する
  (手のひらの向きが seed で揺れた問題の対処)。
- **プロンプトを盛るほど体型がドリフトする**(脚が伸びる)。既定文言は短く保ち、
  体型は `body` パラメータで具体的に言語化する。汎用的な「体型を維持せよ」文は無効だった。
- **背面では衣装も「正面の見え方」が複製される**(2026-07-28、ユーザー報告
  「衣装のベストが後ろのときに前側のようになりました」)。前開きのボレロを着た人物の
  背面に**前開きのV字**が描かれ、隙間からインナーが見えた。原因は末尾の `_KEEP_CLAUSE`
  (「デザイン・衣装・色を全く同じに保つ」)が最強の位置で「正面と同じ見た目」を
  要求していたこと。ここでも**汎用文は効かず**(「衣装は背面から見え背パネルと縫い目が
  見える」を末尾へ追加 → 効果なし。片方の seed では**ボレロ自体が消えた**)、
  `costume` パラメータでの**具体記述だけが効いた**。**丈・範囲まで書くこと**:
  「背中を一枚で覆う」とだけ書くとボレロが膝丈のワンピースへ伸びた。
  実測(背中中央のインナー露出。低いほど良いが、裾から見える分があるので0が正解ではない):
  修正なし 51.2% / 80.1%(ユーザー報告 70.2%)→ 汎用文 79.8% / 46.3%(無効)→
  丈なしの具体記述 1.9% / 0.8%(ワンピース化)→ **丈ありの具体記述 8.3% / 13.0%(正解)**。
- **「seed 1本で直った」は直ったと言えない。** 不安定な症状は必ず複数seedで再現率を
  測ること(後頭部の黒髪化は「直った」と誤って記録し、後で 2/3 で再発していると判明した)。
- **色名の自動推定は「毛で覆われた被写体」限定**(2026-07-28、ユーザー報告
  「何も指定しないと髪型は維持されるが後ろだけ白くなる」)。`sample_fur_color()` は
  被写体全体の**低彩度画素**の中央値を毛色とみなすので、**服や肌が低彩度な人物キャラ
  では誤推定**する(実測: 茶髪+白Tシャツのアニメキャラ → `"cream white"` → 背面
  プロンプトが「後頭部は cream white」になり、**サーバが自分で後頭部を白くしていた**)。
  頭部だけを測る案も検証したが帽子・アクセサリを拾って不安定だったため
  (momo は上10%が帽子の赤、同キャラは上15%以上で肌が混ざり pink)、
  **推定は `animal` のときだけ**行う(`tpose/jobs.py`)。中立/人物で後頭部の色を
  固定したい場合は `fur_color` を明示指定する。
- **2パス目(生成後の編集)のプロンプトに1パス目の構図指示を混ぜない**(2026-07-28、
  ユーザー指摘「Qwen-Image-Edit は一部だけの変更ができるはず。編集用プロンプトに
  Tポーズ用の指示が入っていないか」→ そのとおりだった)。旧
  `build_edit_prompt()` は `"{指示}, keep the pose, composition, design and background
  exactly the same, plain white background, full body visible from head to toe"` で、
  部分編集を壊す要素が3つあった:
    1. `design ... exactly the same` が指示自体と矛盾する(「髪を黒く」はデザイン変更)。
       矛盾すると、モデルは折り合いをつけるため対象領域を作り直す = 髪型ごと変わる。
    2. `plain white background, full body visible from head to toe` は**1パス目用の
       構図指示**で、2パス目に持ち込むと「全身を描き直す」方向へ働く。
    3. **語順**: 「末尾が最強」なのに、末尾を keep 文が占め指示が最弱の先頭にあった。
  現行は keep 文を短い**前置き**にし、指示を**末尾**へ置く。実測(茶髪のアニメキャラ、
  `make the hair black`): 髪色だけが変わり、ボブの髪型・前髪・ヘッドホン・ポーズ・
  服装・画角はすべて維持された。

## 背景除去(旧知見58番)

- 方式は `core/bg.py` の `remove_background(img, method=...)` で選ぶ:
  `"anime"`(SkyTNT/anime-segmentation の ISNet、`skytnt/anime-seg` の `isnetis.onnx`、
  Apache-2.0)/ `"rembg"`(isnet-general-use、汎用)。
  **Tポーズの被写体は必ずキャラクターなので既定は `anime`**(淡い色の毛の取りこぼしが
  少ない: 実測 27,232px → 13,733px)。汎用の `/api/remove_bg` だけ後方互換で `rembg` 既定。
- 生成画像の切り抜き(`tpose/jobs.py` の `_cutout_rgba()`)は、**背景が「生成された
  白一色」であることを利用した自前のキーイング**が本体で、ニューラルのソフトアルファは
  補助にしか使わない(淡い毛の内部が半透明のまま残るのが症状の本質だった)。
  境界から連結した明るい領域だけを背景とし、落ちた画素のうち明るいものだけを回収する
  (影は輝度が低いので巻き込まない)。
- 背景除去は CPU(ONNX)処理なので、**GPUロックを解放してから**実行する。

## UI の罠(旧知見57番)

- `static/style.css` の `.view-tile-image-wrap img` は **`display: none` が既定**で、
  JS が読み込み完了時に `img.style.display = "block"` へ切り替える前提。この1行を
  忘れると「APIは200で画像を返しているのにタイルが空」になる。
- ポーリングでUIを更新するときは **差分だけ当てる**。毎回グリッドを作り直して
  `?t=Date.now()` を付けると、そのたびに再取得されて表示がフラッシュする。
  この実装はサーバ側の `rev`(画像が書き換わった回数)が変わったときだけ
  `?v=<rev>` で再読み込みする(`_bump_view_rev()` を生成・色調整・編集・undo の
  4箇所で呼んでいる)。
- 検証は headless Chrome + CDP が有効(`--remote-allow-origins=*` が必須。無いと
  WSハンドシェイクが403)。スクリーンショットの目視だけでは「小さくて見えない」と
  「非表示」を区別できないので、`getComputedStyle(img).display` や
  `getBoundingClientRect()` を直接検査すること。

## 既知の制約・未検証事項

- 45度ビューの「左右」の区別は不安定(front-left と front-right で同じ向きが出る
  事例あり)。3D用途では使わない参考出力のため放置している。
- 45度ビューでは腕がやや前下がりにドリフトする(front/back のポーズ忠実度が最良)。
- 出力の足元基準の正規化(bbox中心・接地・等倍スケール)は未実装。
- 真横(90度)ビューは提供しない(6通りの方法すべてで破綻した。`tpose/prompts.py` 参照)。
- `DS_QUANT=gguf-q4_k_m` 経路はこのリポジトリでは未検証(移植はしてある)。
