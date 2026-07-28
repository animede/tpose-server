# -*- coding: utf-8 -*-
"""
生成本体(Qwen-Image-Edit)。

移植元: diffusers-server/families/qwen_image/generate.py の run_edit() と共通ヘルパー
(さらにその元は Ｑwenimage-edit-diffusers/app.py の各エンドポイント関数の中身)。
T2I / I2I / ControlNet / Inpaint / Canny / Layered / edit_angles 系は本リポジトリでは
使わないため移植していない。

この関数群は「画像(PIL.Image 等)は既に読み込み・前処理済み」という前提で呼ばれる。
outputs/ への保存・メタデータ dict の組み立てまでをこのモジュールが担当し、
app.py / tpose 側はアップロードファイルの読み込みと HTTPException 変換のみを行う。

生成の排他制御(GPU 同時1件)は呼び出し側(core.gpu.generation_lock)が担うため、
ここでは torch.cuda.empty_cache() / reset_peak_memory_stats() の「生成直前」呼び出しのみ
行う(元実装のロジックをそのまま踏襲)。
"""
import os
import time
import uuid
from datetime import datetime
from typing import Optional

import torch
from PIL import Image

from core import progress as progress_mod
from core.optimize import (
    disable_text_encoder_cpu_offload,
    enable_text_encoder_cpu_offload,
    ensure_text_encoder_on_gpu,
    should_offload_edit_text_encoder,
)

from engine import edit as edit_mod
from engine import state
from engine.paths import (
    LIGHTNING_SHIFT,
    LIGHTNING_TRUE_CFG_SCALE,
    NEGATIVE_PROMPT_DEFAULT,
)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

MAX_EDIT_IMAGES = 3


def _round16(x: int) -> int:
    return max(16, round(x / 16) * 16)


def _new_output_path(mode: str, ext: str = "png"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{mode}_{ts}_{uuid.uuid4().hex[:8]}.{ext}"
    path = os.path.join(OUTPUTS_DIR, name)
    return path, name


def preprocess_image(image: Image.Image, target_pixels: int = 1024 * 1024) -> Image.Image:
    """ComfyUI の ImageScaleToTotalPixels 相当。アスペクト比維持で総画素数を揃え、16の倍数に丸める。

    抽出元: app.py _preprocess_image()。
    """
    image = image.convert("RGB")
    w, h = image.size
    if w <= 0 or h <= 0:
        raise ValueError("invalid image size")
    scale = (target_pixels / (w * h)) ** 0.5
    new_w = _round16(w * scale)
    new_h = _round16(h * scale)
    if (new_w, new_h) != (w, h):
        image = image.resize((new_w, new_h), Image.LANCZOS)
    return image


def make_generator(seed: Optional[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is None or seed < 0:
        return None, None
    return torch.Generator(device=device).manual_seed(seed), seed


def _reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_gb():
    return torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else None


def _progress_callback(mode: str, total_steps: int, extra: "dict | None" = None):
    """diffusers の callback_on_step_end に渡すクロージャを作る。

    全 Qwen-Image 系パイプライン(pipeline_qwenimage*.py)は
    callback_on_step_end(self, step_index, timestep, callback_kwargs) の
    シグネチャで step_index を 0-origin で渡してくる(実装確認済み)。
    dict を返す必要がある(callback_kwargs をそのまま素通しする)。
    """
    progress_mod.start_generating(mode, total_steps, extra)

    def _cb(pipe, step_index, timestep, callback_kwargs):
        progress_mod.update_step(step_index + 1, total_steps)
        return callback_kwargs

    return _cb


def run_edit(req: dict, src_images: list) -> dict:
    """抽出元: app.py generate_edit()。src_images は preprocess_image() 済みリスト。

    Phase 1 申し送り事項(REBUILD_PLAN §4 Phase 2 タスク4)への対処: 48GB専有では
    Edit の標準解像度(参照画像から自動推定される1024²相当)で VAE decode の一時ピークが
    ~5GBほど乗り、OOMすることを実機確認済み(height=640,width=640 明示で成功: 39GB)。
    そのため任意パラメータ height / width を追加した。未指定(None)時は旧挙動のまま
    (QwenImageEditPlusPipeline が参照画像から自動推定する解像度を使う)。
    """
    if not src_images:
        raise ValueError("参照画像を1〜3枚アップロードしてください。")
    if len(src_images) > MAX_EDIT_IMAGES:
        raise ValueError(f"参照画像は最大{MAX_EDIT_IMAGES}枚までです。")

    pipe = edit_mod.get_edit_pipeline()

    steps_eff = req["steps"]
    cfg_eff = req["cfg"]
    shift_eff = req.get("shift")
    lightning = req.get("lightning", True)
    lightning_active = edit_mod.set_edit_lightning(pipe, lightning)
    lightning_unavailable = lightning and not lightning_active
    if lightning and lightning_active:
        if steps_eff <= 0 or steps_eff > 12:
            steps_eff = 4
        cfg_eff = LIGHTNING_TRUE_CFG_SCALE
        shift_eff = shift_eff if shift_eff is not None else LIGHTNING_SHIFT
    elif lightning_unavailable:
        if steps_eff <= 8:
            steps_eff = 30
        if cfg_eff <= 1.5:
            cfg_eff = 4.0

    edit_mod.set_scheduler_shift(pipe.scheduler, shift_eff)

    generator, used_seed = make_generator(req.get("seed", -1))

    # 任意の解像度指定(OOM回避用)。未指定なら None のままパイプラインに渡し、
    # 参照画像からの自動推定(旧挙動)に任せる。
    width = _round16(req["width"]) if req.get("width") else None
    height = _round16(req["height"]) if req.get("height") else None

    # TE(text_encoder) CPU退避(DS_EDIT_TE_OFFLOAD、既定 auto)。プロンプトエンコード後に
    # text_encoder をCPUへ移し、denoise/VAE decodeの間VRAMを空けておくことで、48GB専有でも
    # 1024²相当(height/width未指定時の自動推定を含む)がOOMなく収まるようにする
    # (CLAUDE.md 3番/22番、core.optimize.enable_text_encoder_cpu_offload 参照)。
    # 重要: text_encoder のGPU復帰は pipe(...) を呼ぶ直前に明示的に行うこと
    # (ensure_text_encoder_on_gpu())。QwenImageEditPlusPipeline.__call__() は
    # encode_prompt を呼ぶより前に self._execution_device(= text_encoder 等の現在地から
    # 解決される)を確定させるため、encode_prompt 側で GPU へ戻すのでは手遅れ
    # (input_ids は cpu ・ embed_tokens.weight は cuda という食い違いでdevice mismatch
    # エラーになることを実機デバッグで特定した。core.optimize.ensure_text_encoder_on_gpu
    # のdocstring参照)。CPUへ戻すタイミングは encode_prompt 完了直後のまま(ラップ内)。
    te_offload = should_offload_edit_text_encoder(width, height)
    if te_offload:
        enable_text_encoder_cpu_offload(pipe)
        ensure_text_encoder_on_gpu(pipe)
    else:
        disable_text_encoder_cpu_offload(pipe)
        # text_encoder は t2i_pipe / i2i_pipe / edit系パイプライン間で同一インスタンスを
        # 共有している(CLAUDE.md 8番)。disable_text_encoder_cpu_offload() は「この pipe
        # インスタンス自身が前回有効化していた場合のみ」GPUへ戻す実装のため、直前に
        # 別のパイプライン(例: run_i2i())経由でCPU退避されたケースを取りこぼす
        # 可能性がある(2026-07-18、I2IのOOM修正に伴う整合性対応)。保険として明示的に
        # 確認する。
        ensure_text_encoder_on_gpu(pipe)

    _reset_vram_stats()
    t0 = time.time()
    # 重要: QwenImageEditPlusPipeline には必ずリスト(image=[img, ...])で渡すこと。
    # 単体画像で渡すと Plus 条件付けが効かずキャラクター同一性が崩れる。
    result = pipe(
        image=src_images,
        prompt=req["prompt"],
        negative_prompt=req.get("negative_prompt") or None,
        num_inference_steps=steps_eff,
        true_cfg_scale=cfg_eff,
        width=width,
        height=height,
        generator=generator,
        callback_on_step_end=_progress_callback(
            "tpose" if req.get("_progress_extra") else "edit",
            steps_eff,
            req.get("_progress_extra"),
        ),
    )
    elapsed = time.time() - t0
    peak_vram_gb = _peak_vram_gb()
    progress_mod.set_phase("decoding")

    out_image = result.images[0]
    out_path, out_name = _new_output_path("edit")
    out_image.save(out_path)

    meta = {
        "mode": "edit",
        "prompt": req["prompt"],
        "negative_prompt": req.get("negative_prompt", NEGATIVE_PROMPT_DEFAULT),
        "steps": steps_eff,
        "cfg": cfg_eff,
        "shift": shift_eff,
        "width": width,
        "height": height,
        "seed": used_seed,
        "lightning": lightning_active,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
        "image_url": f"/outputs/{out_name}",
        "num_reference_images": len(src_images),
        "edit_transformer_fallback": state.edit_group["fallback"],
        "quant": state.edit_group["quant"],
        "lightning_requested": lightning,
        "lightning_unavailable_reason": (
            "GGUF量子化中はLoRA非対応のためフォールバックしました(steps/cfgを通常品質側に自動調整)"
            if lightning_unavailable else None
        ),
        "te_offload": te_offload,
    }
    return meta
