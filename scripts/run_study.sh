#!/usr/bin/env bash
# Use an activated environment. Doctor reports the calculated disk requirement.
set -euo pipefail
study_root="${BURGERS_WORK_DIR:-artifacts/study}"
study_device="${BURGERS_DEVICE:-auto}"
for stage in doctor matrix verify-solver benchmark generate-data validate-data train evaluate summarize build-report; do
    burgers "$stage" --config configs/study.yaml --work-dir "$study_root" --device "$study_device"
done
