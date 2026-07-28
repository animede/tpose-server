# -*- coding: utf-8 -*-
"""
Qwen-Image-Edit のランタイム構成(環境変数)。

移植元: diffusers-server/families/qwen_image/runtime.py。
本リポジトリは Edit のみを使うため、T2I モデル選択(DS_T2I_MODEL)と
Layered の量子化設定(DS_LAYERED_QUANT)は移植していない。既定値は無変更。
"""
from core.config import env_bool, env_str

from engine.paths import FP8_LIGHTNING_QUANT_VALUES

DEFAULT_QUANT = "fp8-lightning"


class RuntimeConfig:
    """環境変数から読み取るランタイム構成。

    DS_OFFLOAD(旧 QWENIMG_OFFLOAD) : "auto"(既定)/ "none" / "group" / "group_lowvram" / "model_cpu"
    DS_ATTN(旧 QWENIMG_ATTN)       : "default"(既定)/ "native" / "_native_flash" / "_native_cudnn" /
                       "_native_efficient" / "flex" / "xformers"
    DS_COMPILE(旧 QWENIMG_COMPILE) : "0"(既定)/ "1"
    DS_DEVICE(旧 QWENIMG_DEVICE)   : "cuda"(既定)
    DS_QUANT(旧 QWENIMG_QUANT)     : "fp8-lightning"(既定)/ "gguf-q4_k_m" 等 / "none"(bf16のまま)
    DS_QWEN_TILED_VAE              : "1"(既定、共有VAEのencode/decodeを常時tiled化。
                       大きなキャンバスのVAE encode/decode でのOOM対策)/ "0"(旧動作)
    """

    def __init__(self):
        self.offload = env_str("DS_OFFLOAD", "auto").strip().lower()
        self.attention_backend = env_str("DS_ATTN", "default").strip().lower()
        self.compile = env_bool("DS_COMPILE", False)
        self.device = env_str("DS_DEVICE", "cuda")
        self.quant = env_str("DS_QUANT", DEFAULT_QUANT).strip().lower()
        self.tiled_vae = env_bool("DS_QWEN_TILED_VAE", True)

    def __repr__(self):
        return (
            f"RuntimeConfig(offload={self.offload!r}, attention_backend={self.attention_backend!r}, "
            f"compile={self.compile}, device={self.device!r}, quant={self.quant!r}, "
            f"tiled_vae={self.tiled_vae})"
        )


def quant_suffix(quant: str) -> "str | None":
    """DS_QUANT の値からGGUFファイル名サフィックス(例: 'gguf-q4_k_m' -> 'Q4_K_M')を得る。
    'none'/空/'bf16'/'fp8-lightning' なら None(GGUF経路ではない)を返す。
    """
    quant = (quant or "none").strip().lower()
    if quant in ("", "none", "off", "bf16", "fp8") or quant in FP8_LIGHTNING_QUANT_VALUES:
        return None
    if quant.startswith("gguf-"):
        quant = quant[len("gguf-"):]
    elif quant.startswith("gguf_"):
        quant = quant[len("gguf_"):]
    return quant.upper()


def is_fp8_lightning_quant(quant: str) -> bool:
    """DS_QUANT が fp8-lightning(マージ済み高速化オプション)かどうか。"""
    return (quant or "none").strip().lower() in FP8_LIGHTNING_QUANT_VALUES
