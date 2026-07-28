# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit のシングルトン状態(共有コンポーネント + Editグループ + プロセス内ロック)。

移植元: diffusers-server/families/qwen_image/state.py。本リポジトリは Edit のみを
使うため、T2I / ControlNet / Layered / edit_angles 系のグループ辞書は移植していない
(それらとの相互排他ロジック(families/qwen_image/family.py)も不要になった)。

注意: この lock は非再入。"_load_*_locked" 系の関数は、呼び出し側が既にこの lock を
保持している前提の規約(内部で `with lock:` を再度書くとデッドロックする)。
"""
import threading

from core.gpu import free_vram_gb
from core.optimize import resolve_offload_mode

from engine.runtime import RuntimeConfig

# ============================================================================
# シングルトン状態
# ============================================================================
lock = threading.Lock()

_runtime_config: "RuntimeConfig | None" = None
_offload_mode: "str | None" = None  # プロセス内で一度決定したら使い回す

shared = {
    "vae": None,
    "text_encoder": None,
    "tokenizer": None,
    "loaded": False,
    "load_time_s": None,
}

edit_group = {
    "transformer": None,
    "edit_pipe": None,
    "scheduler": None,
    "processor": None,
    "loaded": False,
    "load_time_s": None,
    "fallback": False,
    "quant": None,
    "lora_available": None,
    "lora_unavailable_reason": None,
    "lightning_merged": False,  # fp8-lightning: LoRAが重みにfuse済みで無効化不可
}


def get_runtime_config() -> RuntimeConfig:
    global _runtime_config
    if _runtime_config is None:
        _runtime_config = RuntimeConfig()
    return _runtime_config


def get_offload_mode(small_transformer_active: bool = False) -> str:
    """オフロードモードはプロセス内で一度だけ決定し、以後のロードで使い回す
    (共有コンポーネントへの二重フック登録を避けるため)。

    small_transformer_active=True で最初に呼ばれた場合、"none"判定の閾値を
    core.config.GGUF_VRAM_FREE_THRESHOLD_GB まで緩和する(GGUF量子化 transformer は
    小さいため、より少ない空きVRAMでも全常駐できる)。
    """
    global _offload_mode
    if _offload_mode is None:
        config = get_runtime_config()
        free_gb = free_vram_gb()
        _offload_mode = resolve_offload_mode(
            config.offload, free_gb, small_transformer_active=small_transformer_active
        )
        print(
            f"[engine] free VRAM: {free_gb:.1f} GB -> offload_mode={_offload_mode}"
            f"{' (小型transformer向け閾値緩和適用)' if small_transformer_active else ''}"
        )
    return _offload_mode


def load_device(offload_mode: str):
    """transformer のロード先デバイス("none" なら直接 GPU、それ以外は CPU 経由)。"""
    import torch

    if offload_mode == "none":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return "cpu"


def reset_offload_mode_cache() -> None:
    """unload 後に空きVRAMが変わっている可能性を考慮して再決定を許可したい場合に使う
    (既定の unload() 実装からは呼ばない。挙動互換を優先)。
    """
    global _offload_mode
    _offload_mode = None
