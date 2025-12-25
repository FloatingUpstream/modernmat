#!/usr/bin/env bash

set -euo pipefail

for arg in "$@"; do
  if [[ "${arg}" == *"/_deps/"* ]]; then
    exit 0
  fi
done

exec clang-tidy "$@"
