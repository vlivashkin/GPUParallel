#!/usr/bin/env bash
set -ex

SCRIPT_PATH=$(dirname "$(readlink -f "$0")")
cd $SCRIPT_PATH/..

python3 -m pytest tests