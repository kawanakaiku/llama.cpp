#!/usr/bin/env python3
"""litert_lm_builder に case 'gemma4' を足す。

export_hf の match (model_type) は qwen3 / qwen2 / gemma3 / function_gemma /
gemma3n しか見ておらず、gemma4 は generic_model に落ちる。
llm_model_type_pb2.Gemma4 は存在するのに使われていない（Issue #1005）。
既に修正済みなら何もしない。
"""
import io
import sys

import litert_torch.generative.export_hf.core.litert_lm_builder as m

p = m.__file__
s = io.open(p, encoding="utf-8").read()

if "case 'gemma4'" in s:
    print("PATCH_SKIP: すでに gemma4 がある")
    sys.exit(0)

anchor = """    case 'gemma3n':"""
if anchor not in s:
    sys.exit("想定の match 文が見つからない。上流が変わった可能性がある")

add = """    case 'gemma4':
      llm_metadata.llm_model_type.CopyFrom(
          llm_model_type_pb2.LlmModelType(gemma4=llm_model_type_pb2.Gemma4())
      )
"""
io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(anchor, add + anchor, 1))
print("PATCH_OK:", p)
