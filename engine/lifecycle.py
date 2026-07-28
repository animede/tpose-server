# -*- coding: utf-8 -*-
"""
unload() / get_status() の実装。

移植元: diffusers-server/families/qwen_image/lifecycle.py。本リポジトリは Edit のみを
使うため、T2I / ControlNet / Layered / edit_angles 系グループの解放分岐は移植していない
(共有コンポーネントは「Editグループが未ロードなら一緒に解放する」という同じ判定)。
"""
import gc

from core import gpu

from engine import state


def unload(target: str = "all") -> dict:
    """VRAM 解放。target: "edit"(Editグループのみ)/ "all"(共有含め全て)。

    共有コンポーネント(vae/text_encoder/tokenizer、bf16で~15.7GB)は、この呼び出しの
    結果として Edit グループがロードされていなければ target に関わらず一緒に解放する
    (再ロードは各 get_*_pipeline() が冪等に行う。実測 ~25秒)。
    """
    freed = []
    with state.lock:
        edit_group = state.edit_group
        shared = state.shared

        if target in ("edit", "all") and edit_group["loaded"]:
            edit_group["transformer"] = None
            edit_group["edit_pipe"] = None
            edit_group["scheduler"] = None
            edit_group["processor"] = None
            edit_group["loaded"] = False
            edit_group["quant"] = None
            edit_group["lora_available"] = None
            edit_group["lora_unavailable_reason"] = None
            edit_group["lightning_merged"] = False
            freed.append("edit")

        if shared["loaded"] and not edit_group["loaded"]:
            shared["vae"] = None
            shared["text_encoder"] = None
            shared["tokenizer"] = None
            shared["loaded"] = False
            freed.append("shared")

        gc.collect()
        gpu.empty_cache()
        gpu.reset_peak_stats()
    print(f"[engine] unloaded: {freed}")
    return {"freed": freed}


def get_status() -> dict:
    runtime_config = state.get_runtime_config()
    edit_group = state.edit_group
    shared = state.shared

    return {
        "offload_mode": state._offload_mode,
        "runtime_config": repr(runtime_config),
        "shared_loaded": shared["loaded"],
        "shared_load_time_s": shared["load_time_s"],
        "edit_loaded": edit_group["loaded"],
        "edit_load_time_s": edit_group["load_time_s"],
        "edit_transformer_fallback": edit_group["fallback"],
        "edit_quant": edit_group["quant"],
        "edit_lora_available": edit_group["lora_available"],
        "edit_lora_unavailable_reason": edit_group["lora_unavailable_reason"],
        "edit_lightning_merged": edit_group["lightning_merged"],
        "gpu_busy": gpu.generation_lock.locked(),
        "vram": gpu.vram_snapshot(),
    }


def is_any_loaded() -> bool:
    return state.shared["loaded"] or state.edit_group["loaded"]
