# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# RUN: %PYTHON %s | FileCheck %s

"""Exercise the `node.meta["mlir.attrs"]` / `node.meta["mlir.arg_attrs"]`
contract honored by FxImporter. Generic attribute names are used; no
downstream dialect is involved."""

import torch
import torch.nn as nn

from torch_mlir import fx
from torch_mlir.extras.annotate import annotate_arg, annotate_module


def run(f):
    print(f"{f.__name__}")
    print("-" * len(f.__name__))
    f()
    print()


class TwoLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(self.fc1(x))


# CHECK-LABEL: test_arg_attrs_from_meta
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32> {my.secret, my.tag = "ciphertext"})
@run
def test_arg_attrs_from_meta():
    def annotate(prog):
        for n in prog.graph.nodes:
            if n.op == "placeholder" and n.name == "x":
                n.meta["mlir.arg_attrs"] = {
                    "my.secret": True,
                    "my.tag": "ciphertext",
                }
                break

    m = fx.export_and_import(
        TwoLinear(), torch.randn(1, 4), annotate=annotate
    )
    print(m)


# CHECK-LABEL: test_op_attrs_from_meta
# CHECK:       torch.aten.linear {{.*}} {my.layer_index = 0 : i64, my.range_lo = -1.000000e+00 : f64}
# CHECK:       torch.aten.linear {{.*}} {my.layer_index = 1 : i64, my.range_lo = 2.500000e+00 : f64}
@run
def test_op_attrs_from_meta():
    def annotate(prog):
        seen = []
        for n in prog.graph.nodes:
            if n.op == "call_function" and "linear" in n.name:
                seen.append(n)
        for i, n in enumerate(seen):
            n.meta["mlir.attrs"] = {
                "my.layer_index": i,
                "my.range_lo": -1.0 if i == 0 else 2.5,
            }

    m = fx.export_and_import(
        TwoLinear(), torch.randn(1, 4), annotate=annotate
    )
    print(m)


# CHECK-LABEL: test_arg_attr_skipped_when_false
# Boolean False means "unit attr absent"; other args get no attribute either.
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32>)
@run
def test_arg_attr_skipped_when_false():
    def annotate(prog):
        for n in prog.graph.nodes:
            if n.op == "placeholder" and n.name == "x":
                n.meta["mlir.arg_attrs"] = {"my.secret": False}
                break

    m = fx.export_and_import(
        TwoLinear(), torch.randn(1, 4), annotate=annotate
    )
    print(m)


# CHECK-LABEL: test_annotate_arg_helper_by_name
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32> {my.flag})
@run
def test_annotate_arg_helper_by_name():
    m = fx.export_and_import(
        TwoLinear(),
        torch.randn(1, 4),
        annotate=lambda p: annotate_arg(p, "x", {"my.flag": True}),
    )
    print(m)


# CHECK-LABEL: test_annotate_arg_helper_by_index
# CHECK:       func.func @main(%arg0: !torch.vtensor<[1,4],f32> {my.flag})
@run
def test_annotate_arg_helper_by_index():
    m = fx.export_and_import(
        TwoLinear(),
        torch.randn(1, 4),
        annotate=lambda p: annotate_arg(p, 0, {"my.flag": True}),
    )
    print(m)


# CHECK-LABEL: test_annotate_module_helper_last
# Only the last op produced by `fc1` should carry the attr.
# CHECK:       torch.aten.linear {{.*}} {my.tag = "first"}
# CHECK-NOT:   torch.aten.linear {{.*}} my.tag
@run
def test_annotate_module_helper_last():
    def annotate(prog):
        count = annotate_module(prog, "fc1", {"my.tag": "first"}, mode="last")
        assert count == 1, f"expected 1 match, got {count}"

    m = fx.export_and_import(TwoLinear(), torch.randn(1, 4), annotate=annotate)
    print(m)
