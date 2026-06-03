#!/bin/sh
set -eu

export HARM_DETECTION_ROOT="${HARM_DETECTION_ROOT:-/workspace}"
mkdir -p \
  "$HARM_DETECTION_ROOT/artifacts/lake" \
  "$HARM_DETECTION_ROOT/artifacts/models" \
  "$HARM_DETECTION_ROOT/artifacts/reports"

exec "$@"
