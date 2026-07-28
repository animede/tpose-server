# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit のモデルパス・リポジトリ定数群。

移植元: diffusers-server/families/qwen_image/paths.py(さらにその元は
Ｑwenimage-edit-diffusers/pipeline_manager.py 行46-236)。
本リポジトリは Edit のみを使うため、T2I / 2512 / ControlNet / Layered /
Multiple-angles LoRA 関連の定数は移植していない。値・実測コメントは無変更。
"""
import os

from core.resolve import COMFYUI_MODELS_DIR

# --- Edit: Qwen-Image-Edit-2511 transformer ---
EDIT_TRANSFORMER_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2511_bf16.safetensors"
)
EDIT_TRANSFORMER_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
EDIT_TRANSFORMER_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2511_bf16.safetensors"

# フォールバック(2511 bf16 が読めない場合、2509 fp8 を使う)
EDIT_TRANSFORMER_FALLBACK_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "diffusion_models", "qwen_image_edit_2509_fp8_e4m3fn.safetensors"
)
EDIT_TRANSFORMER_FALLBACK_HF_REPO = "Comfy-Org/Qwen-Image-Edit_ComfyUI"
EDIT_TRANSFORMER_FALLBACK_HF_FILE = "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"
EDIT_TRANSFORMER_FALLBACK_PREFIX = "model.diffusion_model."

# --- Edit: Lightning 4steps LoRA ---
EDIT_LORA_LIGHTNING_PATH = os.path.join(
    COMFYUI_MODELS_DIR, "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
)
EDIT_LORA_LIGHTNING_HF_REPO = "lightx2v/Qwen-Image-Edit-2511-Lightning"
EDIT_LORA_LIGHTNING_HF_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

# --- GGUF 量子化 transformer(unsloth 配布。DS_QUANT で opt-in)---
# Q4_K_M は bf16(約40GB)の約1/3(約12〜13GB)。GGUF のテンソルキー名は diffusers
# ネイティブの命名と一致している(prefix変換不要、実機確認済み)。
# 既知の制約: GGUF量子化 transformer には LoRA を適用できない(PEFT が GGUFLinear を
# 認識しない)。そのため Lightning は無効になり、steps/cfg が通常品質側へ自動調整される。
EDIT_GGUF_HF_REPO = "unsloth/Qwen-Image-Edit-2511-GGUF"
EDIT_GGUF_FILENAME_TEMPLATE = "qwen-image-edit-2511-{suffix}.gguf"
# "Qwen/Qwen-Image-Edit-2511" は2511専用のHFリポジトリで transformer/config.json を持つ。
EDIT_GGUF_CONFIG_REPO = "Qwen/Qwen-Image-Edit-2511"

# --- fp8 Lightning マージ版(DS_QUANT=fp8-lightning、既定)---
# GGUF量子化transformerにはLoRAを適用できない(上記)ため、代わりに
# 「bf16 transformerをロード -> Lightning LoRAをその場でfuse(重みに焼き込み)
# -> enable_layerwise_casting(storage_dtype=fp8_e4m3fn) でストレージのみfp8に圧縮」
# という手順で高速化する(実機検証済み: fuse後 38.06GB -> layerwise casting後 19.05GB)。
FP8_LIGHTNING_QUANT_VALUES = {"fp8-lightning", "fp8_lightning", "fp8lightning"}

# --- 共有コンポーネント(vae / text_encoder / tokenizer)の由来リポジトリ ---
BASE_REPO = "Qwen/Qwen-Image"
EDIT_PROCESSOR_REPO = "Qwen/Qwen-Image-Edit-2509"  # 2511 transformer でも processor は 2509 のものを使う

# 画像生成で定番の低品質除外セット(手・指の破綻、低解像度、透かし等)。
# 注意: Lightning LoRA 使用時(true_cfg_scale=1.0)は CFG が無効化されるため
# negative_prompt 自体が効かない(LIGHTNING_TRUE_CFG_SCALE 参照)。
NEGATIVE_PROMPT_DEFAULT = (
    "lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer fingers, "
    "cropped, worst quality, low quality, jpeg artifacts, signature, watermark, "
    "username, blurry"
)

# Lightning LoRA 推奨パラメータ(lightx2v 推奨値)
LIGHTNING_SHIFT = 3.0
LIGHTNING_TRUE_CFG_SCALE = 1.0
DEFAULT_SHIFT = None  # None ならスケジューラの事前学習済み既定値を使う
