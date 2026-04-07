#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 <DATA_PATH> <OUTPUT_PATH> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --max-count N    Max samples per split (default: all)"
    echo "  --skip-convert   Skip WDS conversion (reuse existing)"
    echo "  --skip-test      Skip validation test"
    echo "  --clean          Remove output dir before conversion"
    exit 1
}

if [ $# -lt 2 ]; then
    usage
fi

DATA_PATH="$1"
OUTPUT_PATH="$2"
shift 2

python make_flaudio_wds.py \
    --data-path "$DATA_PATH" \
    --output-path "$OUTPUT_PATH" \
    "$@"
