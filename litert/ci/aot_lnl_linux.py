#!/usr/bin/env python3
"""tflite を Intel NPU (Lunar Lake) 向けに AOT コンパイルする。runner(Linux)用。"""
import argparse, os, pathlib, re, sys, time


# 2GB 超のバッファを 1 回で write すると環境によっては落ちるので分割する
def _wb(self, data):
    v = memoryview(data).cast("B")
    with open(self, "wb") as f:
        for i in range(0, len(v), 1 << 29):
            f.write(v[i:i + (1 << 29)])
    return len(v)


pathlib.Path.write_bytes = _wb

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.intel_openvino import target as ov

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--out", required=True)
p.add_argument("--soc", default="LNL", choices=["LNL", "PTL"])
p.add_argument("--subgraphs", default="")
p.add_argument("--config", action="append", default=[])
a = p.parse_args()

kw = {}
if a.config:
    kw["intel_openvino_configs_map"] = ",".join(a.config)
sg = [int(x) for x in a.subgraphs.split(",") if x.strip()] or None


def dump_error(exc):
    """apply_plugin は詳細を一時ファイルに書くだけなので、ログに写しておく。
    ランナーが消えたあとでも原因を追えるようにするため。"""
    m = re.search(r"(\S+\.error)", str(exc))
    if not (m and os.path.exists(m.group(1))):
        return
    path = m.group(1)
    print("--- " + path + " ---")
    with open(path, errors="replace") as f:
        lines = [l.rstrip() for l in f]
    keep, seen = [], set()
    for l in lines:
        if ("ERROR" in l or "Partitioned subgraph" in l
                or "not supported" in l or "Exception" in l):
            if l not in seen:
                seen.add(l)
                keep.append(l)
    print("\n".join(keep[:60]))
    print("--- 末尾 ---")
    print("\n".join(lines[-15:]))


print("入力 : %s (%.3f GB)" % (a.model, os.path.getsize(a.model) / 1e9), flush=True)
t = time.perf_counter()
try:
    aot_compile.aot_compile(a.model, output_dir=a.out,
                            target=ov.Target(soc_model=ov.SocModel(a.soc)),
                            keep_going=False, subgraphs_to_compile=sg, **kw)
    print("AOT_OK %.1fs" % (time.perf_counter() - t))
except Exception as e:
    print("AOT_FAIL %s: %s" % (type(e).__name__, e))
    dump_error(e)
    sys.exit(1)
for f in sorted(os.listdir(a.out)):
    print("  %.1f MB  %s" % (os.path.getsize(os.path.join(a.out, f)) / 1e6, f))
