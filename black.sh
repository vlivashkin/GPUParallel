#!/usr/bin/env bash
set -ex

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_PATH

black --line-length 120 gpuparallel