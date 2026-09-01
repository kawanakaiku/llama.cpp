#!/usr/bin/env python3
"""export_hf を呼ぶ。

上流の export_main.py は `def main(_)` を引数なしで呼んでいて
(`TypeError: main() missing 1 required positional argument`)、
`python -m` では起動できない。fire を直接叩く。
"""
import fire
from litert_torch.generative.export_hf import export as lib

if __name__ == "__main__":
    fire.Fire(lib.export)
