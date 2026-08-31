#!/usr/bin/env python3
"""tflite を Intel NPU (Lunar Lake) 向けに AOT コンパイルする。runner(Linux)用。"""
import argparse, os, pathlib, sys, time

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

print(f"入力 : {a.model} ({os.path.getsize(a.model)/1e9:.3f} GB)", flush=True)
t = time.perf_counter()
try:
    aot_compile.aot_compile(a.model, output_dir=a.out,
                            target=ov.Target(soc_model=ov.SocModel(a.soc)),
                            keep_going=False, subgraphs_to_compile=sg, **kw)
    print(f"AOT_OK {time.perf_counter()-t:.1f}s")
except Exception as e:
    print(f"AOT_FAIL {type(e).__name__}: {e}")
    sys.exit(1)
for f in sorted(os.listdir(a.out)):
    print(f"  {os.path.getsize(os.path.join(a.out,f))/1e6:.1f} MB  {f}")
