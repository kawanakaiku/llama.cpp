#!/bin/bash
# Intel NPU (Lunar Lake) 向けの .litertlm を最初から最後まで作る。
# GitHub Actions ランナー（Ubuntu / 4core / 16GB）で無人実行する前提。
#
#   litert-torch で静的shape書き出し → aot_compile(LNL) → litert-lm-builder で梱包
#
# 対話セッション(upterm)は途中で切れるので、全部これ1本にまとめてある。
#
# 使い方: build_npu_litertlm.sh <0.6b|1.7b|4b> [quantize] [prefill_len] [kv_len]
set -euo pipefail

SIZE="${1:-1.7b}"
QUANT="${2:-dynamic_int8}"     # int4 系は plugin が拒否する（後述）
PREFILL="${3:-128}"
KVLEN="${4:-1280}"
MASKIN="${5:-True}"        # mask_as_input
TRKV="${6:-True}"          # transpose_kv_cache
TAG="${7:-}"               # 出力名の接尾辞（実験の識別用）

ROOT=/home/runner/build
ART=/home/runner/my-artifact
V=$ROOT/.venv/bin
HF_REPO="Qwen/Qwen3-${SIZE^^}"   # 0.6b->0.6B, 1.7b->1.7B, 4b->4B

mkdir -p "$ROOT" "$ART"
cd "$ROOT"

step() { echo; echo "==================== $* ===================="; date -Is; }

# ---------------------------------------------------------------- 0. スワップ
# 4B の書き出しは 16GB では足りない。ディスクは 70GB 以上空いている。
step "スワップ確保"
if ! swapon --show | grep -q swapfile2; then
  sudo fallocate -l 32G /swapfile2
  sudo chmod 600 /swapfile2
  sudo mkswap /swapfile2 >/dev/null
  sudo swapon /swapfile2
fi
free -g | sed -n '2p;3p'

# ---------------------------------------------------------------- 1. 環境
step "依存パッケージ"
if [ ! -x "$V/python" ]; then
  python3 -m venv "$ROOT/.venv"
  $V/pip -q install -U pip
  $V/pip -q install litert-torch ai-edge-litert litert-lm-builder huggingface_hub
  # litert-torch は tflite のスキーマを tensorflow から取るので必須
  $V/pip -q install tensorflow-cpu
  # AOT コンパイラ本体。openvino は SDK が要求する nightly に固定される
  $V/pip -q install --pre "openvino==2026.3.0.dev20260622" \
      --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly
  $V/pip -q install ai-edge-litert-sdk-intel
fi
$V/python -c "import litert_torch, ai_edge_litert, ai_edge_litert_sdk_intel; print('deps ok')"

# ---------------------------------------------------------------- 2. 重み取得
step "チェックポイント取得 $HF_REPO"
CKPT="$ROOT/ckpt-$SIZE"
$V/python - "$HF_REPO" "$CKPT" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2],
                  allow_patterns=["*.json", "*.safetensors", "*.txt"], max_workers=8)
print("CKPT_OK")
PY
du -sh "$CKPT"

# 埋め込みを共有しているモデルは lm_head.weight を保存していないが、
# litert-torch のローダは無条件に要求するので補っておく（Qwen3-4B など）
$V/python "$ROOT/fix_tied_lm_head.py" "$CKPT"

# ---------------------------------------------------------------- 3. 参照メタデータ
# 梱包に必要な LlmMetadata は公式の Qwen3 litertlm から借りる。
# 中身は tokenizer とテンプレートの定義だけなので重みとは独立。
step "参照 litertlm からメタデータを取り出す"
if [ ! -f "$ROOT/ref_dump/LlmMetadataProto.pbtext" ]; then
  $V/python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repo = "litert-community/Qwen3-0.6B"
# ファイル名は変わりうるので一番小さい .litertlm を拾う
files = [f for f in api.list_repo_files(repo) if f.endswith(".litertlm")]
info = api.model_info(repo, files_metadata=True)
sizes = {s.rfilename: (s.size or 0) for s in info.siblings}
target = min(files, key=lambda f: sizes.get(f, 1 << 62))
print("参照:", target)
hf_hub_download(repo, target, local_dir="/home/runner/build/ref")
print("REF_OK")
PY
  $V/litert-lm-peek --litertlm_file "$(ls $ROOT/ref/*.litertlm | head -1)" \
      --dump_files_dir "$ROOT/ref_dump" > "$ROOT/ref_peek.txt" 2>&1
fi
ls -la "$ROOT/ref_dump"

# ---------------------------------------------------------------- 4. 書き出し
# ここが肝。gpu_dynamic_shapes を立てない＝静的shapeで出る。
# 公開されている汎用 .litertlm は動的shapeなので NPU コンパイラが受け付けない。
step "tflite 書き出し (size=$SIZE quant=$QUANT prefill=$PREFILL kv=$KVLEN)"
rm -rf "$ROOT/out"
$V/python -m litert_torch.generative.examples.qwen.convert_v3_to_tflite \
  --model_size="$SIZE" \
  --checkpoint_path="$CKPT" \
  --output_path="$ROOT/out" \
  --output_name_prefix="qwen3-$SIZE" \
  --prefill_seq_lens="$PREFILL" \
  --kv_cache_max_len="$KVLEN" \
  --quantize="$QUANT" \
  --mask_as_input=$MASKIN \
  --transpose_kv_cache=$TRKV
TFL=$(ls "$ROOT/out"/*.tflite | head -1)
ls -la "$TFL"

# ---------------------------------------------------------------- 5. AOT
step "Intel NPU (LNL) 向け AOT コンパイル"
rm -rf "$ROOT/aot"
$V/python "$ROOT/aot_lnl_linux.py" "$TFL" --out "$ROOT/aot" \
  --config optimize_fq_after_matmul=true
NPUTFL=$(ls "$ROOT/aot"/*.tflite | head -1)
ls -la "$NPUTFL"

# ---------------------------------------------------------------- 6. 梱包
step "litertlm に梱包"
OUT="$ART/qwen3-${SIZE}${TAG}_intel_LNL.litertlm"
$V/litert-lm-builder \
  system_metadata --str author "built on GitHub Actions runner" \
                  --str base_model "$HF_REPO" \
                  --str npu_target "IntelOpenVINO_LNL" \
                  --str quantize "$QUANT" \
  llm_metadata --path "$ROOT/ref_dump/LlmMetadataProto.pbtext" \
  hf_tokenizer --path "$CKPT/tokenizer.json" \
  tflite_model --path "$NPUTFL" --model_type prefill_decode \
  output --path "$OUT"
ls -la "$OUT"

# 分割でもう一部コピーが要るので、先に中間物を捨ててディスクを空ける
rm -rf "$ROOT/out" "$ROOT/aot"
df -h / | tail -1

# ---------------------------------------------------------------- 7. 分割
# GitHub Release は 1 ファイル 2GB まで。
step "リリース用に 1.9GB ずつ分割"
cd "$ART"
split -b 1900M -d -a 2 "$OUT" "$(basename "$OUT").part"
rm -f "$OUT"
sha256sum ./*.part* > SHA256SUMS.txt
cat > RESTORE.md <<'MD'
# 復元方法

```bash
cat qwen3-*.litertlm.part* > qwen3.litertlm
sha256sum -c SHA256SUMS.txt   # 分割片の検証
```

Windows (PowerShell):
```powershell
cmd /c copy /b (Get-ChildItem *.part* | Sort-Object Name | ForEach-Object {$_.Name}) -join '+' qwen3.litertlm
```

実行:
```bash
pip install litert-lm==0.14.0 openvino
python litert/chat_litert.py --model qwen3.litertlm
```
MD
ls -la "$ART"
echo
echo "BUILD_ALL_DONE"
