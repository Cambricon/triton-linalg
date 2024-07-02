#!/bin/bash
# Copyright (C) [2022-2025] by Cambricon.

# ==============================================================================
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ==============================================================================

# Runs all the lint checks that we run on Gitlab locally.

# WARNING: this script *makes changes* to the working directory and the index.

set -uo pipefail

FINAL_RET=0
LATEST_RET=0

SCRIPTS_DIR="$(dirname $0)"
source ${SCRIPTS_DIR}/common.sh
BASE_REF="${1:-master}"
pip install yapf -y

apt install clang-format -y

enable_update_ret

echo $BASE_REF

echo "***** yapf *****"
# Don't fail script if condition is false
files=`find . -name "*.py" |grep -v "./triton/"`
SKIP_FILE_LIST="./triton/"
SKIP_FILE_LIST=$(echo $SKIP_FILE_LIST | tr ' ' '\n')
for file in $files; do
  if echo "$SKIP_FILE_LIST" | grep -Fxq "$file"; then
    echo "***** skip $file for yapf *****"
    continue
  fi
  yapf $file -i --lines 1-2000
done

echo "***** clang-format *****"
git-clang-format --style=file $BASE_REF
git diff --exit-code -- ./lint_check

check_ret
