#!/usr/bin/env bash
# Differential gate (audit Section 9): build the working tree as the candidate,
# run the corpus on both sides and compare against the frozen baseline.
#
# Usage: audit/harness/gate.sh <manifest.yml> <report.md>
# Env:   OB_HARNESS_DIR  work directory holding lib_baseline / base.rds
#        OB_SEEDS        seeds per scenario (default 3; use 20 for integration:
#                        with 3 seeds the medians of layer E are noisy)
set -euo pipefail
manifest="$1"; report="$2"
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../.." && pwd)"
W="${OB_HARNESS_DIR:?set OB_HARNESS_DIR}"; seeds="${OB_SEEDS:-3}"
base="$W/base_s${seeds}.rds"
[ -f "$base" ] || Rscript "$here/run_side.R" "$W/lib_baseline" "$base" "$seeds"
rm -rf "$W/lib_candidate"
"$here/build_side.sh" "$root" "$W/lib_candidate" >/dev/null
Rscript "$here/run_side.R" "$W/lib_candidate" "$W/cand.rds" "$seeds" >/dev/null
# Cumulative scope: the current manifest first, then every other manifest in
# audit/diffs (changes already accepted on this branch).
others=$(ls "$root"/audit/diffs/*-manifest.yml 2>/dev/null | grep -v -F "$(basename "$manifest")" | paste -sd, - || true)
Rscript "$here/compare.R" "$base" "$W/cand.rds" "$manifest${others:+,$others}" "$report"
