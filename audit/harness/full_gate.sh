#!/usr/bin/env bash
# Integration gate: layers A-C/E (gate.sh), D (perf), F (fuzz under ASan/UBSan)
# and the testthat suite, for the working tree against the frozen baseline.
# Usage: audit/harness/full_gate.sh <manifest.yml> <tag>
set -uo pipefail
here="$(cd "$(dirname "$0")" && pwd)"; root="$(cd "$here/../.." && pwd)"
W="${OB_HARNESS_DIR:?}"; m="$1"; tag="$2"; out="$root/audit/diffs"
rc=0
OB_SEEDS="${OB_SEEDS:-20}" "$here/gate.sh" "$m" "$out/$tag-report.md" || rc=1
[ -f "$W/perf_base.rds" ] || Rscript "$here/perf.R" "$W/lib_baseline" "$W/perf_base.rds" 3 >/dev/null
Rscript "$here/perf.R" "$W/lib_candidate" "$W/perf_cand.rds" 3 >/dev/null
Rscript "$here/perf_compare.R" "$W/perf_base.rds" "$W/perf_cand.rds" "$out/$tag-perf.md" || rc=1
ASAN="$(gcc -print-file-name=libasan.so)"
SAN='-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined'
printf 'CXXFLAGS=%s\nCXX17FLAGS=%s\nLDFLAGS=-fsanitize=address,undefined\n' "$SAN" "$SAN" > "$W/Makevars.san"
[ -d "$W/lib_baseline_san" ] || R_MAKEVARS_USER="$W/Makevars.san" "$here/build_side.sh" "$W/obwoe-baseline" "$W/lib_baseline_san" >/dev/null
rm -rf "$W/lib_candidate_san"; R_MAKEVARS_USER="$W/Makevars.san" "$here/build_side.sh" "$root" "$W/lib_candidate_san" >/dev/null
export ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
[ -f "$W/fuzz_base_san.rds" ] || LD_PRELOAD="$ASAN" Rscript "$here/fuzz.R" "$W/lib_baseline_san" "$W/fuzz_base_san.rds" 200 >/dev/null
LD_PRELOAD="$ASAN" Rscript "$here/fuzz.R" "$W/lib_candidate_san" "$W/fuzz_cand_san.rds" 200 > "$W/fuzz_cand.log" 2>&1 || { echo "sanitizer abort:"; tail -30 "$W/fuzz_cand.log"; rc=1; }
Rscript "$here/fuzz_compare.R" "$W/fuzz_base_san.rds" "$W/fuzz_cand_san.rds" "$out/$tag-fuzz.md" || rc=1
"$here/run_tests.sh" | tail -2 | tee "$out/$tag-tests.txt" || rc=1
echo "FULL GATE: $([ $rc -eq 0 ] && echo PASS || echo FAIL)"; exit $rc
