#!/bin/bash
# 公式の export_hf 経路で焼く。
# こちらは AOT まで統合されていて、gemma-3n/gemma-4 系のアーキテクチャも通る。
# 自前の scatter-free 化とは別経路なので、まず素で通るかを見る。
set -e
ROOT=/home/runner/build
ART=/home/runner/my-artifact
V=$ROOT/.venv/bin
HF_MODEL="${1:-google/gemma-4-E2B-it}"
NAME="${2:-gemma-4-E2B-it}"
CACHE_LEN="${3:-4096}"
PREFILL="${4:-128}"
RECIPE="${5:-dynamic_wi8_afp32}"
mkdir -p "$ART"; cd "$ROOT"

echo "==================== 環境 ===================="
# export_hf モードは build_npu_litertlm.sh を通らないので、ここで venv を作る。
if ! swapon --show | grep -q swapfile2; then
  sudo fallocate -l 32G /swapfile2 && sudo chmod 600 /swapfile2
  sudo mkswap /swapfile2 >/dev/null && sudo swapon /swapfile2
fi
free -g | sed -n '2p;3p'
if [ ! -x "$V/python" ]; then
  python3 -m venv "$ROOT/.venv"
  $V/pip -q install -U pip
  $V/pip -q install litert-lm-builder huggingface_hub tomli-w
  $V/pip -q install tensorflow-cpu
  $V/pip -q install --pre "openvino==2026.3.0.dev20260622" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
  $V/pip -q install ai-edge-litert ai-edge-litert-sdk-intel
  # export_hf は新しめの機能なので git から入れる
  $V/pip -q install "git+https://github.com/google-ai-edge/litert-torch.git"
fi
$V/python -c "import litert_torch, ai_edge_litert, ai_edge_litert_sdk_intel; print('deps ok')"

echo "==================== export_hf があるか ===================="
if ! $V/python -c "import litert_torch.generative.export_hf.export as e; print('ok')" 2>/dev/null; then
  echo "PyPI 版に export_hf が無いので git から入れ直す"
  $V/pip install -q --upgrade "git+https://github.com/google-ai-edge/litert-torch.git"
fi
$V/python -c "
import inspect, litert_torch.generative.export_hf.export as e
ps = inspect.signature(e.export).parameters
need = ['aot_backend','aot_soc_model','litert_lm_model_type_override','bundle_litert_lm']
print('export() の引数数:', len(ps))
print('必要な引数:', {n: (n in ps) for n in need})
"

echo "==================== gemma4 の case を足す ===================="
$V/python "$ROOT/patch_gemma4.py"

echo "==================== 書き出し + AOT ===================="
rm -rf "$ROOT/hf_out"
$V/python "$ROOT/export_hf_run.py" \
  --model="$HF_MODEL" \
  --output_dir="$ROOT/hf_out" \
  --task=text_generation \
  --prefill_lengths="[$PREFILL]" \
  --cache_length="$CACHE_LEN" \
  --quantization_recipe="$RECIPE" \
  --split_cache=True \
  --externalize_embedder=True \
  --bundle_litert_lm=True \
  --aot_backend=intel_openvino \
  --aot_soc_model=LNL \
  --keep_temporary_files=True
echo "EXPORT_HF_OK"
ls -la "$ROOT/hf_out"

echo "==================== 成果物 ===================="
LM=$(ls "$ROOT/hf_out"/*.litertlm 2>/dev/null | head -1)
[ -z "$LM" ] && { echo "litertlm が出ていない"; exit 1; }
OUT="$ART/${NAME}_selfbuilt_intel_LNL.litertlm"
cp "$LM" "$OUT"
ls -la "$OUT"
cd "$ART" && split -b 1900M -d -a 2 "$OUT" "$(basename "$OUT").part" && rm -f "$OUT"
sha256sum ./*.part* > SHA256SUMS.txt
ls -la "$ART"
echo EXPORT_HF_DONE
