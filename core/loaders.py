# -*- coding: utf-8 -*-
"""
streaming single-file / multi-shard ロード、GGUF ロード、
bf16 -> Lightning LoRA fuse -> fp8 layerwise casting、の共通関数群。

移植元: diffusers-server/core/loaders.py(さらにその元は
Ｑwenimage-edit-diffusers/pipeline_manager.py の各 _load_* / _fuse_* 関数)。
本リポジトリは Edit 1系統しか使わないため、複数シャードのストリーミングロード
(T2I/Layered 用)と Multiple-angles LoRA 併用の fuse ヘルパーは移植していない。

方針:
  - 元実装は各関数が QwenImageTransformer2DModel 固有だった(from_config /
    QwenImageTransformer2DModel.load_config を直接呼ぶ)。Phase 1 で他の transformer クラス
    (Edit/Layered は同じ QwenImageTransformer2DModel だが、将来ファミリーが増える場合に備え)
    にも使えるよう、model_cls 引数を追加して汎用化した。呼び出しパターン自体
    (config_repo/config_subfolder/path/load_device の組み合わせ)は元実装のまま。
  - _fuse_lightning_lora_and_cast_to_fp8 は「Edit と T2I で共通の関数として使えるよう
    引数化されていた」という元実装の設計(CLAUDE.md 8番)をそのまま踏襲する。
    ロジック・コメント中の実測値は変更しない。
  - torch の import はモジュールトップレベル、CUDA 初期化を伴う処理(実際のロード・
    empty_cache 等)は各関数内で行う(呼び出されない限り CUDA は初期化されない)。
"""
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import safe_open

from core.resolve import resolve_model_path

__all__ = [
    "load_safetensors_streaming",
    "load_transformer_from_config",
    "load_transformer_gguf",
    "fuse_lightning_lora_and_cast_to_fp8",
]


def load_safetensors_streaming(
    transformer,
    path: str,
    load_device,
    strip_prefix: str = "",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """safetensors を1テンソルずつ meta model へ流し込み、CPU/GPU の一時ピークを抑える。

    注意: init_empty_weights() で作った meta モデルのパラメータ dtype は既定で float32。
    set_module_tensor_to_device() は dtype= を渡さないと value を「meta 側の既存 dtype」
    (= float32)へ再キャストしてしまうため、bf16(や fp8)のつもりが実質 fp32 でロードされ、
    ホストRAM/VRAMを本来の2〜4倍消費するバグになる。必ず dtype= を明示すること。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_safetensors_streaming()
    (行410-447)。ロジック・docstring の注意事項とも無変更。
    """
    expected_keys = set(transformer.state_dict().keys())
    loaded_keys = set()
    unexpected_keys = []
    load_device = str(load_device)

    with safe_open(path, framework="pt", device=load_device) as f:
        for raw_key in f.keys():
            if raw_key == "__index_timestep_zero__":
                continue
            key = raw_key[len(strip_prefix):] if strip_prefix and raw_key.startswith(strip_prefix) else raw_key
            if key not in expected_keys:
                unexpected_keys.append(key)
                continue
            tensor = f.get_tensor(raw_key)
            set_module_tensor_to_device(transformer, key, load_device, value=tensor, dtype=dtype, clear_cache=False)
            loaded_keys.add(key)
            del tensor

    missing_keys = sorted(expected_keys - loaded_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"transformer state_dict mismatch: missing={len(missing_keys)}, "
            f"unexpected={len(unexpected_keys)}"
        )


def load_transformer_from_config(
    model_cls,
    config_repo: str,
    config_subfolder: str,
    path: str,
    load_device,
    strip_prefix: str = "",
):
    """config だけ HF Hub から取得して meta model を作り、streaming ロードで重みを流し込む。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_transformer_from_config()
    (行450-462)。元実装は QwenImageTransformer2DModel 固定だったが、Phase 1 での
    engine/ 移植時にそのまま渡せるよう model_cls を引数化した
    (呼び出しパターンは同一: model_cls.load_config(...) -> init_empty_weights() 内で
    model_cls.from_config(...) -> streaming ロード)。
    """
    config = model_cls.load_config(config_repo, subfolder=config_subfolder)
    with init_empty_weights():
        transformer = model_cls.from_config(config)
    load_safetensors_streaming(transformer, path, load_device, strip_prefix=strip_prefix)
    transformer.eval()
    return transformer


def load_transformer_gguf(model_cls, path: str, config_repo: str):
    """GGUF量子化 transformer をロードする(実機検証済みの引数構成)。

    from_single_file() は既定でCPUにロードする(GGUF量子化テンソルはuint8格納なので
    Q4_K_Mで約12GBとホストRAM圧迫が小さい)。呼び出し側で .to("cuda") するか
    configure_transformer_offload() に渡すこと。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の _load_transformer_gguf()
    (行626-643)。model_cls を引数化した点のみ変更。
    """
    from diffusers import GGUFQuantizationConfig

    transformer = model_cls.from_single_file(
        path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        config=config_repo,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer.eval()
    return transformer


def fuse_lightning_lora_and_cast_to_fp8(
    transformer,
    lora_local_path: str,
    lora_hf_repo: str,
    lora_hf_file: str,
) -> None:
    """Lightning 4steps LoRA を transformer の重みに fuse(焼き込み)し、
    その後 enable_layerwise_casting でストレージのみ fp8_e4m3fn に圧縮する。

    Edit(2511)・T2I(無印 Qwen-Image)・Layered の全てで共通に使う(呼び出し側が対象
    transformer に合った lora_local_path/lora_hf_repo/lora_hf_file を渡す)。

    実機検証済みの手順・実測値(RTX PRO 6000 Blackwell, bf16 2511 transformer基準):
      1. bf16 transformer ロード直後: 38.06GB(Edit)/ 約38GB(T2I、同等サイズ)
      2. Lightning LoRA を transformer.load_lora_adapter() でロード
         (LoRAファイルは ComfyUI/Musubi 形式のキー名(lora_down/lora_up/alpha)なので、
         diffusers の _convert_non_diffusers_qwen_lora_to_diffusers() で
         PEFT形式(lora_A/lora_B, "transformer."プレフィックス付き)に変換してから渡す。
         変換後のキーには "transformer." プレフィックスが付くため、
         load_lora_adapter(..., prefix="transformer")(デフォルト値)を使うこと。
         prefix=None を指定するとプレフィックス不一致で
         "Target modules {...} not found in the base model" エラーになる(実機で確認済み)。
      3. fuse_lora() で重みに焼き込み: 38.86GB(LoRAの差分だけ一時的に増加)
      4. unload_lora() + empty_cache() でLoRAモジュールを除去: 38.06GB(fuse後のbf16のまま)
      5. enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
         + empty_cache(): 19.05GB(ほぼ瞬時にストレージがfp8化される。
         LayerwiseCastingHook.initialize_hook() が module.to(dtype=storage_dtype) を
         即座に呼ぶため、bf16確保分はこの時点で解放される)

    fuse後は通常のLoRAアダプタとして無効化できない(重みに永続的に焼き込まれるため)。
    そのため quant="fp8-lightning" の transformer は常にLightning適用状態になる。

    抽出元: Ｑwenimage-edit-diffusers/pipeline_manager.py の
    _fuse_lightning_lora_and_cast_to_fp8()(行1148-1197)。ロジック・docstring とも無変更。
    """
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_qwen_lora_to_diffusers

    lora_path = resolve_model_path(lora_local_path, lora_hf_repo, lora_hf_file)
    raw_sd = {}
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            raw_sd[key] = f.get_tensor(key)
    converted_sd = _convert_non_diffusers_qwen_lora_to_diffusers(raw_sd)

    transformer.load_lora_adapter(converted_sd, adapter_name="lightning_fuse", prefix="transformer")
    transformer.fuse_lora(adapter_names=["lightning_fuse"])
    transformer.unload_lora()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
