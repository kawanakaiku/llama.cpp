#!/usr/bin/env python3
"""NPU コンパイラが刻まない形で Qwen3 を書き出す。

既定の書き出しだとグラフが 57 個に分割され、実行時に落ちる。
Google 純正の gemma-4 E2B は 2 分割で、同じ手順で問題なく動く。
つまり分割数そのものが問題。

分割の原因は plugin が受け付けない op が層ごとに現れること:
  DYNAMIC_UPDATE_SLICE (151) : KV キャッシュ書き込み（既定）
  STABLEHLO_SCATTER    (190) : index_copy に替えても結局これになる
  STABLEHLO_COMPOSITE  (206) : HLFB で注釈された RoPE / 正規化
  FILL                 (94)

対策:
  - enable_hlfb=False          複合opの注釈をやめる
  - KV 更新を one-hot マスクと行列積で書き直す（scatter 系を完全に消す）
"""
import torch
from absl import app, flags

from litert_torch.generative.examples.qwen import qwen3
from litert_torch.generative.layers import kv_cache as kv_utils
from litert_torch.generative.layers import scaled_dot_product_attention as sdpa_lib
from litert_torch.generative.quantize import quant_attrs, quant_recipes
from litert_torch.generative.utilities import converter

flags_ = converter.define_conversion_flags("qwen")
_MODEL_SIZE = flags.DEFINE_enum("model_size", "0.6b", ["0.6b", "1.7b", "4b"],
                                "The size of the model to convert.")
_KV_UPDATE = flags.DEFINE_enum("kv_update", "mask", ["dus", "index_copy", "mask"],
                               "KV キャッシュの更新方法")
_NO_HLFB = flags.DEFINE_bool("no_hlfb", True, "STABLEHLO_COMPOSITE を出さない")
_KV_DTYPE = flags.DEFINE_enum("kv_dtype", "float32", ["float32", "float16"],
                              "KV キャッシュの型。fp16 にすると転送量が半分になる")
_NO_PREFILL = flags.DEFINE_bool("no_prefill", False,
                                "prefill 署名を作らない。prefill と decode は"
                                "それぞれ重みを丸ごと1本ずつ持つので、"
                                "落とすとモデルがほぼ半分になる。"
                                "プロンプトは decode を1トークンずつ回せばよい")
_W_INT4 = flags.DEFINE_bool("weights_int4_channelwise", False,
                            "重みを int4 チャンネル単位にする。"
                            "ブロック単位の int4 は plugin が拒否するがこちらは未検証")

_BUILDER = {"0.6b": qwen3.build_0_6b_model,
            "1.7b": qwen3.build_1_7b_model,
            "4b": qwen3.build_4b_model}
_CONFIG = {"0.6b": "get_0_6b_model_config",
           "1.7b": "get_1_7b_model_config",
           "4b": "get_4b_model_config"}


def _scatter_free_update(cache, input_pos, k_slice, v_slice):
    """scatter を使わずに KV キャッシュを更新する。

    レイアウトは BTNH。位置 input_pos の行だけ差し替えたいので、
    one-hot 行列 [S, T] を作って
        新しいキャッシュ = 元 * (1 - 使う行) + one_hot^T @ 新しい行
    と書く。使うのは iota / 比較 / cast / matmul / mul / add だけで、
    どれも NPU コンパイラが受け付ける。
    """
    def one(buf, sl):
        sl = sl.to(buf.dtype)   # KV を fp16 にしたとき slice は fp32 で来る
        t = buf.shape[1]
        pos = input_pos.reshape(-1).to(torch.long)             # [S]
        idx = torch.arange(t, device=buf.device)               # [T]
        onehot = (idx.reshape(1, t) == pos.reshape(-1, 1)).to(buf.dtype)  # [S, T]
        b, s, n, h = sl.shape
        flat = sl.permute(1, 0, 2, 3).reshape(s, b * n * h)    # [S, B*N*H]
        add = torch.matmul(onehot.transpose(0, 1), flat)       # [T, B*N*H]
        add = add.reshape(t, b, n, h).permute(1, 0, 2, 3)      # [B, T, N, H]
        keep = (1.0 - onehot.sum(0)).reshape(1, t, 1, 1).to(buf.dtype)
        return buf * keep + add

    return kv_utils.KVCacheEntry(one(cache.k_cache, k_slice),
                                 one(cache.v_cache, v_slice))


def main(_):
    size = _MODEL_SIZE.value

    if _KV_UPDATE.value == "index_copy":
        kv_utils._update_kv_impl = kv_utils._update_kv_base_impl
        print("patched: KV 更新を index_copy に")
    elif _KV_UPDATE.value == "mask":
        kv_utils._update_kv_impl = _scatter_free_update
        kv_utils._update_kv_base_impl = _scatter_free_update
        print("patched: KV 更新を one-hot マスク + 行列積に（scatter 系を消す）")

    if _KV_DTYPE.value != "float32":
        # 呼び出し側が dtype を渡していないだけなので、既定値ごと差し替える。
        # 毎トークン 294MB の往復が律速なので、半分になるだけで効く。
        import functools
        dt = getattr(torch, _KV_DTYPE.value)
        orig_from = kv_utils.KVCache.from_model_config.__func__

        @classmethod
        @functools.wraps(orig_from)
        def _typed(cls, *a, **kw):
            kw.setdefault("dtype", dt)
            return orig_from(cls, *a, **kw)

        kv_utils.KVCache.from_model_config = _typed

        # SDPA は q/k/v の型が揃っていないと弾く。
        # キャッシュを fp16 のまま保って転送量を減らしたいので、q を合わせる。
        for _n in ("scaled_dot_product_attention",
                   "scaled_dot_product_attention_with_hlfb",
                   "scaled_dot_product_attention_transposed",
                   "scaled_dot_product_attention_transposed_with_hlfb"):
            _f = getattr(sdpa_lib, _n, None)
            if _f is None:
                continue

            def _mk(f):
                @functools.wraps(f)
                def w(q, k, v, *a, **kw):
                    # キャッシュは fp16 のまま持って転送量を半分にするが、
                    # 演算は q の型（fp32）に合わせて上げる。
                    # 逆向き（q を fp16 に落とす）だと GQA の tfl.broadcast_to が
                    # fp16 で legalize できずに変換が落ちる。
                    if k.dtype != q.dtype:
                        k = k.to(q.dtype)
                    if v.dtype != q.dtype:
                        v = v.to(q.dtype)
                    return f(q, k, v, *a, **kw)
                return w

            setattr(sdpa_lib, _n, _mk(_f))
        print(f"patched: KV キャッシュを {_KV_DTYPE.value} に / SDPA の型を合わせる")

    if _NO_HLFB.value:
        name = _CONFIG[size]
        orig = getattr(qwen3, name)

        def patched():
            c = orig()
            c.enable_hlfb = False
            return c

        setattr(qwen3, name, patched)
        print("patched: enable_hlfb=False")

    if _NO_PREFILL.value:
        # _add_signatures は prefill_seq_lens をループするだけなので、
        # 空にすれば prefill 署名が作られない。
        # mask_as_input=True ならマスク長の計算にも使われない。
        flags.FLAGS.prefill_seq_lens = []
        print("patched: prefill 署名を作らない（decode のみ）")

    if _W_INT4.value:
        # 既定の書き出しは重みが二重に入る（元の int8 + NPU ブロブの fp16）。
        # 元を int4 にすれば両方が縮む。ブロック単位は
        # "Unsupported Quantization type 3" で弾かれたのでチャンネル単位で試す。
        orig_recipe = converter.get_quant_recipe_from_flag

        def _recipe(name, model_config):
            return quant_recipes.full_dynamic_recipe(
                mcfg=model_config,
                weight_dtype=quant_attrs.Dtype.INT4,
                granularity=quant_attrs.Granularity.CHANNELWISE)

        converter.get_quant_recipe_from_flag = _recipe
        print("patched: 重みを int4 チャンネル単位に")

    converter.build_and_convert_to_tflite_from_flags(_BUILDER[size])


if __name__ == "__main__":
    app.run(main)
