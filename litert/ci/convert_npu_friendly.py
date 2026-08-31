#!/usr/bin/env python3
"""NPU コンパイラが刻まない形で Qwen3 を書き出す。

OpenVINO の compiler plugin が受け付けない op が層ごとに現れるため、
既定の書き出しではグラフが 57 個に分割され、実行時に落ちる。

  DYNAMIC_UPDATE_SLICE (op 151, 302回) : KV キャッシュの書き込み
  STABLEHLO_COMPOSITE  (op 206, 356回) : HLFB で注釈された RoPE / 正規化など
  FILL                 (op  94, 110回)

どちらも litert-torch 側に無効化する手段があるので、変換前に潰す。
  - kv_cache: use_dus=False 相当（index_copy に落とす）
  - model_config: enable_hlfb=False（複合opの注釈をやめる）

使い方: convert_npu_friendly.py --model_size=0.6b --checkpoint_path=... --output_path=...
"""
from absl import app, flags

from litert_torch.generative.examples.qwen import qwen3
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.utilities import converter

flags_ = converter.define_conversion_flags("qwen")
_MODEL_SIZE = flags.DEFINE_enum("model_size", "0.6b", ["0.6b", "1.7b", "4b"],
                                "The size of the model to convert.")
_NO_DUS = flags.DEFINE_bool("no_dus", True, "DYNAMIC_UPDATE_SLICE を使わない")
_NO_HLFB = flags.DEFINE_bool("no_hlfb", True, "STABLEHLO_COMPOSITE を出さない")

_BUILDER = {"0.6b": qwen3.build_0_6b_model,
            "1.7b": qwen3.build_1_7b_model,
            "4b": qwen3.build_4b_model}
_CONFIG = {"0.6b": qwen3.get_0_6b_model_config,
           "1.7b": qwen3.get_1_7b_model_config,
           "4b": qwen3.get_4b_model_config}


def main(_):
    size = _MODEL_SIZE.value

    if _NO_DUS.value:
        # update() は use_dus で分岐するが、呼び出し側が既定 True を渡してくるので
        # 実装そのものを差し替える。
        kv_utils._update_kv_impl = kv_utils._update_kv_base_impl
        print("patched: KV 更新を index_copy に（DYNAMIC_UPDATE_SLICE を回避）")

    if _NO_HLFB.value:
        orig = _CONFIG[size]

        def patched():
            c = orig()
            c.enable_hlfb = False
            return c

        setattr(qwen3, _CONFIG[size].__name__, patched)
        print("patched: enable_hlfb=False（STABLEHLO_COMPOSITE を回避）")

    converter.build_and_convert_to_tflite_from_flags(_BUILDER[size])


if __name__ == "__main__":
    app.run(main)
