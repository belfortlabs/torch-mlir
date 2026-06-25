#!/usr/bin/env bash
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

# Dependencies installed inside the manylinux container by cibuildwheel's
# `before-all` step (see [tool.cibuildwheel.linux] in pyproject.toml).
#
# We build with the manylinux default toolchain (gcc-toolset), which is the
# proven path for building LLVM/MLIR in manylinux. We deliberately do NOT use
# the el8-module clang: clang 21.1.8 from that module crashes with SIGILL
# (illegal instruction) on GitHub-hosted runners while compiling MLIR.
#
# We only need ccache here (to reuse the LLVM/MLIR object cache across the
# per-Python-version builds and across CI runs). ninja comes from the pip
# build requirements (CMAKE_GENERATOR=Ninja).
set -euo pipefail

echo ":::: Installing build dependencies for the manylinux build"

# ccache lives in EPEL on the AlmaLinux 8 base.
dnf install -y --quiet epel-release || true
dnf install -y --quiet ccache || dnf install -y ccache

echo ":::: Tool versions"
gcc --version | head -1 || true
ccache --version | head -1 || true

echo ":::: Done installing build dependencies"
