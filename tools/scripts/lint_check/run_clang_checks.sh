#!/bin/bash
# Copyright (C) [2022-2025] by Cambricon.

set -ex

# Build all the target first.
# Note:
# - Use target '//tools:genesis-opt' to cover only local compile outputs.
./run.sh -f //tools:genesis-opt -d -j --use-clang

# Use a clean env to run refresh_compile_commands as the bazel aquery commands
# may use env variables outside which will influence the complie_commands.json.
env -i PATH="$PATH" NEUWARE_HOME="$NEUWARE_HOME" bazel run //build_tools/scripts:refresh_compile_commands

# Rerun build before IWYU, as refresh_compile_commands may influence bazel state.
# Since it has bazel cache, it doesn't actually affect the time too much.
./run.sh -f //tools:genesis-opt -d -j --use-clang

# Run IWYU and return.
env -i PATH="$PATH" NEUWARE_HOME="$NEUWARE_HOME" ./build_tools/scripts/iwyu_tool.py -j 64 -p . -- -Xiwyu --cxx17ns -Xiwyu '--verbose=1' -Xiwyu '--error=1' -Xiwyu --transitive_includes_only ||
  (cat ./build_tools/scripts/IWYU.md; exit -1);

# Run clang-tidy and check change.
run-clang-tidy-12 -fix -format -quiet -j 64
git diff --exit-code
