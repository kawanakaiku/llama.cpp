#!/bin/bash
# NPU が刻まない書き出しを試す。分割数が落ちるかを見るのが目的なので、
# 実験中は /continue を打たない（セッションを残して次の試行に使う）。
set -e
ROOT=/home/runner/build
ART=/home/runner/my-artifact
V=$ROOT/.venv/bin
SIZE="${1:-0.6b}"
TAG="${2:--nodus-nohlfb}"
KVLEN="${3:-1280}"
PREFILL="${4:-128}"
KVDTYPE="${5:-float32}"
mkdir -p "$ART"; cd "$ROOT"

echo "==================== 書き出し (パッチ版) ===================="
rm -rf "$ROOT/out"
$V/python "$ROOT/convert_npu_friendly.py" \
  --model_size="$SIZE" \
  --checkpoint_path="$ROOT/ckpt-$SIZE" \
  --output_path="$ROOT/out" \
  --output_name_prefix="qwen3-${SIZE}${TAG}" \
  --prefill_seq_lens="$PREFILL" \
  --kv_cache_max_len="$KVLEN" \
  --quantize=dynamic_int8 \
  --mask_as_input=True \
  --transpose_kv_cache=False \
  --kv_update=mask --no_hlfb=True --kv_dtype="$KVDTYPE"
TFL=$(ls "$ROOT/out"/*.tflite | head -1)
ls -la "$TFL"

echo "==================== AOT ===================="
rm -rf "$ROOT/aot"
$V/python "$ROOT/aot_lnl_linux.py" "$TFL" --out "$ROOT/aot" \
  --config optimize_fq_after_matmul=true
NPUTFL=$(ls "$ROOT/aot"/*.tflite | head -1)
ls -la "$NPUTFL"

echo "==================== 梱包 ===================="
OUT="$ART/qwen3-${SIZE}${TAG}_intel_LNL.litertlm"
$V/litert-lm-builder \
  system_metadata --str author "npu-friendly export" --str npu_target "IntelOpenVINO_LNL" \
  llm_metadata --path "$ROOT/ref_dump/LlmMetadataProto.pbtext" \
  hf_tokenizer --path "$ROOT/ckpt-$SIZE/tokenizer.json" \
  tflite_model --path "$NPUTFL" --model_type prefill_decode \
  output --path "$OUT"
rm -rf "$ROOT/out" "$ROOT/aot"
cd "$ART" && split -b 1900M -d -a 2 "$OUT" "$(basename "$OUT").part" && rm -f "$OUT"
sha256sum ./*.part* > SHA256SUMS.txt
ls -la "$ART"
echo NPU_FRIENDLY_DONE
