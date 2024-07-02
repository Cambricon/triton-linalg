#!/bin/bash
# Copyright (C) [2022-2025] by Cambricon.

FINAL_RET=0
LATEST_RET=0

function update_ret() {
  LATEST_RET="$?"
  if [[ "${LATEST_RET}" -gt "${FINAL_RET}" ]]; then
    FINAL_RET="${LATEST_RET}"
  fi
}

# Update the exit code after every command
function enable_update_ret() {
  trap update_ret DEBUG
}


function check_ret() {
  if (( "${FINAL_RET}" != 0 )); then
    echo "Encountered failures. Check error messages and changes to the working" \
         "directory and git index (which may contain fixes) and try again."
  fi

  exit "${FINAL_RET}"
}
