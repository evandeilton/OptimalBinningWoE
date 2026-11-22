#!/usr/bin/env bash
# Build one side (baseline or candidate) of the differential harness.
#
# Usage: build_side.sh <source_dir> <library_dir>
#
# The source tree is copied to a temporary directory and installed into
# <library_dir>. When the available Rcpp is older than 1.0.13 (no variadic
# DataFrame::create beyond 20 columns), src/OB_Utils.cpp -- which is NOT part
# of any binning engine -- is patched with a variadic shim. The patch is the
# same for both sides, so it cannot bias the comparison.
set -euo pipefail
src="$1"; lib="$2"
tmp="$(mktemp -d)"
cp -r "$src"/. "$tmp/pkg"
rm -rf "$tmp/pkg/.git" "$tmp/pkg/audit" "$tmp/pkg/src/"*.o "$tmp/pkg/src/"*.so
if Rscript -e 'quit(status = as.integer(packageVersion("Rcpp") >= "1.0.13"))'; then
  python3 - "$tmp/pkg/src/OB_Utils.cpp" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
shim = r'''
// ---- build-only shim (audit harness): variadic data.frame for old Rcpp ----
template <typename T>
static void ob_df_push(Rcpp::List& out, Rcpp::CharacterVector& nm,
                       const Rcpp::traits::named_object<T>& o) {
  out.push_back(Rcpp::wrap(o.object)); nm.push_back(o.name);
}
template <typename... Args>
static Rcpp::DataFrame ob_df(const Args&... args) {
  Rcpp::List out; Rcpp::CharacterVector nm;
  (ob_df_push(out, nm, args), ...);
  out.attr("names") = nm;
  return Rcpp::DataFrame(out);
}
'''
i = s.index('using namespace Rcpp;') + len('using namespace Rcpp;')
s = s[:i] + shim + s[i:]
s = s.replace('return DataFrame::create(', 'return ob_df(')
open(p, 'w').write(s)
PY
fi
mkdir -p "$lib"
MAKEFLAGS="-j$(nproc)" R CMD INSTALL --no-docs --no-multiarch --no-test-load \
  --library="$lib" "$tmp/pkg" > "$tmp/install.log" 2>&1 || { tail -40 "$tmp/install.log"; exit 1; }
echo "installed $(basename "$src") -> $lib"
rm -rf "$tmp"
