# CRAN Submission Comments — OptimalBinningWoE 1.13.4

## Summary

Single-defect bug-fix release. It fixes the check ERROR reported for 1.13.3 on
`r-release-macos-arm64` and `r-oldrel-macos-arm64`. No other flavour was
affected: both macOS x86_64 flavours, all Linux flavours and all Windows
flavours were OK.

There are no new features, no new dependencies and no user-visible API change.

This arrives only days after 1.13.3 was accepted, which the incoming checks
flag as `Days since last update: 2`. The submission is made this soon because
it repairs a failing CRAN check on two flavours, not to ship anything new.

## The failure and its cause

```
── Failure ('test-obwoe-sql.R:559:3'): multinomial models require an explicit class ──
Expected `sql_eval_case_num(cases[["x_woe"]], df$x, "x")` to equal `...[]`.
Differences:
  `actual[131:137]`: -0.010 -0.051 0.090 -0.010 -0.010 -0.010 -0.051
`expected[131:137]`: -0.010 -0.051 0.090 -0.051 -0.010 -0.010 -0.051
```

One observation out of 900 was scored by the wrong bin. It is the observation
whose value is *exactly* a fitted cut point, and the test asserts that the
generated SQL puts it in the same bin `cut(..., right = TRUE)` does.

`obwoe_sql()` writes each cut point as the shortest decimal string that parses
back to the identical double, so that an observation on a boundary cannot
drift into the neighbouring bin. Each candidate was checked with
`as.numeric()` before being emitted.

**That check is not sound on aarch64.** R accumulates the decimal digits of a
string in `LDOUBLE`, which on aarch64 macOS is no wider than a `double`, so
beyond about fifteen digits `as.numeric()` can land one bit from the nearest
double. It then rejects literals that do round trip and accepts literals that
do not. With every candidate rejected, the search fell through to a fallback
that wrote fewer digits than the value needs: the cut point
`-0.13964785691628961` came out as `-0.13964785691629`, which is the smaller
number, so the observation equal to the cut point failed
`x <= -0.13964785691629` and was scored one bin up.

The search now decides for itself rather than asking `as.numeric()`. A
candidate with `nd` decimals is checked as `m / 10^nd`, where `m` is its digits
read as an integer: while `m` is below 2^53 and `nd` is at most 22, both
operands are exact, so IEEE 754 gives the correctly rounded quotient -- the
nearest double to the candidate -- on every platform R runs on. Candidates
outside those bounds are not judged; the value falls back to seventeen
significant digits, the width that identifies a double uniquely.
`as.numeric()` must still agree before a candidate is accepted -- not to decide
the question, but so that a literal R itself reads back as a different double
is never written into an audit artifact.

Cut points that are exact in binary still read short. About five per cent of
values now carry one more digit than in 1.13.3, being those the new check
declines to judge.

The same commit fixes a second, smaller defect found while auditing that code
path: `digits` rounded to the requested number of decimal places and then
formatted the result with R's default seven significant digits, so
`digits = 8` on `1234.5678901234` emitted `1234.568`.

### Verification

The fix was verified on the failing architecture, not only by inspection.
`macos-latest` -- Apple silicon -- was restored to the GitHub Actions check
matrix, where it had been disabled while `infer` had no ARM64 binary. On that
runner the multinomial boundary test that fails on 1.13.3 now passes.

Enabling it also surfaced, and this release fixes, two test-side consequences
of the same `LDOUBLE` limitation. A test that reads a generated literal back
with `as.numeric()` and compares bit-for-bit is testing R's string-to-double
conversion, not the generated SQL, so those comparisons now probe the platform
and relax to a few ULP where that conversion is inexact. They remain exact on
Linux, on Windows and on macOS x86_64.

Three regression tests were added to `tests/testthat/test-obwoe-sql.R`:

* a bulk round trip over 3,500 values;
* a boundary check on cut points taken from continuous data; and
* a diagnostic that spells out a one-bit disagreement between the SQL and
  `obwoe_apply()`, which `waldo` otherwise reports only as "actual != expected
  but don't know how to show the difference".

The pre-existing boundary test used a feature whose cut points are small
integers, which any literal writer renders exactly, and so could not catch
this class of defect.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.6.x, GCC, `R CMD check --as-cran --run-donttest` on
the built tarball with vignettes:

```
0 errors | 0 warnings | 2 notes
```

Tests, examples (including `--run-donttest`) and all three vignettes build and
run cleanly. The test suite runs 2,043 assertions with no failures and no
warnings.

**NOTE 1 — `checking CRAN incoming feasibility`: `Days since last update: 2`.**
Expected, and explained above: this release exists to repair the check ERROR
on the two aarch64 macOS flavours.

**NOTE 2 — `checking HTML version of manual`.** The check machine has no
`tidy` binary and no `V8` package, so HTML validation and math rendering were
skipped. This is a property of the machine, not of the package.

### INFO — installed size

```
installed size is 72.3Mb
sub-directories of 1Mb or more:
  libs  70.6Mb
```

Unchanged from 1.13.3, and explained there: the package compiles 37 binning
algorithms as separate translation units, and essentially all of the reported
size is debug symbols from the local `-g` compiler default. The same object
after `strip --strip-debug` is 2.9 MB. We have not added `-Os` or an explicit
strip step to `src/Makevars`, since CRAN policy asks packages not to override
the platform's compiler flags.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, R 4.6.x, GCC — `0 errors | 0 warnings | 2 notes`
* **GitHub Actions**: ubuntu-latest (R devel, release, oldrel-1, oldrel-2,
  oldrel-3), windows-latest (R release) and macos-latest (R release).
  `macos-latest` is Apple silicon, so the matrix now covers the architecture
  this release repairs; it had been disabled while `infer` had no ARM64
  binary, which is no longer the case.
* **macOS builder (aarch64, R release)**: to be verified before submission, as
  a second reading of the architecture the GitHub Actions runner already
  covers.
* **win-builder (R-release and R-devel)**: to be verified before submission

---

## Dependencies

No change. No new `Imports`, `Depends` or `Suggests`.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
