#!/usr/bin/env python3
"""tie_word_embeddings なモデルに lm_head.weight を補う。

litert-torch のローダは lm_head.weight を無条件に pop するので、
埋め込みを共有していて lm_head を保存していないチェックポイント
（Qwen/Qwen3-4B など）は KeyError で落ちる。
埋め込みの複製を 1 シャード足して index を更新する。
"""
import json, os, sys

from safetensors.torch import load_file, save_file

ckpt = sys.argv[1]
idx_path = os.path.join(ckpt, "model.safetensors.index.json")
single = os.path.join(ckpt, "model.safetensors")

EMB = "model.embed_tokens.weight"
LM = "lm_head.weight"

if os.path.exists(idx_path):
    with open(idx_path, encoding="utf-8") as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    if LM in wm:
        print("SKIP: lm_head.weight はすでにある")
        sys.exit(0)
    if EMB not in wm:
        sys.exit(f"想定外: {EMB} が index に無い")
    shard = load_file(os.path.join(ckpt, wm[EMB]))
    out = "model-lmhead.safetensors"
    save_file({LM: shard[EMB].clone()}, os.path.join(ckpt, out))
    wm[LM] = out
    idx["metadata"]["total_size"] = idx["metadata"].get("total_size", 0) + shard[EMB].numel() * shard[EMB].element_size()
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)
    print(f"ADDED: {LM} <- {EMB} ({out})")
elif os.path.exists(single):
    st = load_file(single)
    if LM in st:
        print("SKIP: lm_head.weight はすでにある")
        sys.exit(0)
    st[LM] = st[EMB].clone()
    save_file(st, single)
    print(f"ADDED: {LM} <- {EMB} (single shard)")
else:
    sys.exit("safetensors が見つからない")
