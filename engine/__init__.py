# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit エンジン(このリポジトリで使う唯一のモデル経路)。

移植元: diffusers-server/families/qwen_image/(T2I / I2I / Edit / ControlNet /
Inpaint / Layered / edit_angles 系を持つパッケージ)から **Edit だけ**を抜き出したもの。
モデルファミリーが1つしかないため、diffusers-server の `core/registry.py`
(FamilyRegistry による排他ロード)は移植していない。ロードは `load()`、生成は
`generate_edit()` を直接呼ぶ。

サブモジュール構成:
  paths.py     - モデルパス・リポジトリ定数
  runtime.py   - RuntimeConfig・quant判定(DS_QUANT 等)
  state.py     - シングルトン状態(shared / edit_group・ロック・オフロードモード)
  shared.py    - 共有コンポーネント(vae/text_encoder/tokenizer)のロード
  edit.py      - Edit(QwenImageEditPlusPipeline)のロードと Lightning 制御
  generate.py  - 生成本体(run_edit)
  lifecycle.py - unload() / get_status()
"""
from engine.edit import get_edit_pipeline
from engine.generate import (
    MAX_EDIT_IMAGES,
    OUTPUTS_DIR,
    make_generator,
    preprocess_image,
    run_edit,
)
from engine.lifecycle import get_status, is_any_loaded, unload
from engine.paths import NEGATIVE_PROMPT_DEFAULT
from engine.state import get_runtime_config, lock

__all__ = [
    "load",
    "generate_edit",
    "get_edit_pipeline",
    "get_status",
    "is_any_loaded",
    "unload",
    "run_edit",
    "preprocess_image",
    "make_generator",
    "MAX_EDIT_IMAGES",
    "OUTPUTS_DIR",
    "NEGATIVE_PROMPT_DEFAULT",
    "get_runtime_config",
    "lock",
]


def load() -> None:
    """Edit グループ(共有コンポーネント + transformer + パイプライン)をロードする。

    冪等(ロード済みなら何もしない)。生成の前に1回呼ぶ。
    """
    get_edit_pipeline()


def generate_edit(request: dict) -> dict:
    """request = {"prompt": ..., "steps": ..., "_images": [PIL.Image, ...], ...}。

    戻り値は run_edit() のメタデータ dict(image_url / elapsed_s / peak_vram_gb 等)。
    """
    return run_edit(request, request["_images"])
