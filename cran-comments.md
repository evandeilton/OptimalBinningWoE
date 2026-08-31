# CRAN Submission Comments — OptimalBinningWoE 1.13.5

## Summary

Single-defect release. It fixes the two test failures reported for 1.13.4 by
the additional `noLD` check — x86_64 Linux with R-devel configured using
`--disable-long-double`. All thirteen regular flavours were OK on 1.13.4.

The failing code is in the test suite, not in the package. No package code
changed, so no generated SQL, no fitted binning and no user-visible API changed
with this release.

## The failure and its cause

```
── Failure ('test-obwoe-sql.R:147:3'): boundaries hold for cut points that need full double precision ──
Expected `sql_eval_case_num(case, cp, "x")` to equal `as.numeric(seq_along(cp))`.
Differences:
  `actual[1:4]`: 2.0 2.0 3.0 4.0
`expected[1:4]`: 1.0 2.0 3.0 4.0

── Failure ('test-obwoe-sql.R:148:3'): boundaries hold for cut points that need full double precision ──
Expected `sql_eval_case_num(case, x, "x")` to equal `as.numeric(...)`.
Differences:
  `actual[417:423]`: 4.0 4.0 3.0 2.0 4.0 7.0 4.0
`expected[417:423]`: 4.0 4.0 3.0 1.0 4.0 7.0 4.0
```

Both failures are the same observation: the one whose value is *exactly* the
first fitted cut point, reported in bin 2 rather than bin 1.

**The generated SQL is correct.** `obwoe_sql()` wrote that cut point as
`-1.3003598122031599`, all seventeen significant digits the value needs, and
any conforming reader — every database this SQL targets included — parses that
string back to exactly the fitted double.

The fault is in the miniature CASE evaluator that the tests use to score the
generated SQL, in `tests/testthat/helper-germancredit.R`. It read each literal
back with `as.numeric()`, and **`as.numeric()` is not a conforming reader on a
build without long double.** R accumulates the decimal digits of a string in
`LDOUBLE`; where that is no wider than a `double`, a literal carrying more than
fifteen digits can come back a few ULP from the double it names. Here
`-1.3003598122031599` was read one ULP low, so the observation equal to the cut
point failed `x <= -1.3003598122031599` and fell through to the next branch.
The evaluator put it in the wrong bin; the SQL did not.

The evaluator now decodes a literal without trusting `as.numeric()` with
anything it cannot be shown to parse exactly:

* Digits below 2^53 with at most 22 decimals are read as `m / 10^nd`, one
  correctly rounded division of two exactly representable doubles — the same
  argument the literal *writer* has used since 1.13.4.
* Otherwise `sprintf()`, which hands the rendering to the C library and is
  correctly rounded on every platform, confirms `as.numeric()`'s answer by
  printing it back. Exactly one double prints back as a literal that carries
  more digits than are needed to separate two doubles, which is precisely the
  case R gets wrong.
* Where it does not print back, the neighbouring doubles are bracketed around
  the literal by exact decimal comparison and the nearer of the two wins, the
  midpoint decided as `2*lit` against `a + b` so that no inexact subtraction
  enters.

This also removes a smaller inaccuracy in the other direction: where `LDOUBLE`
is 80 bits, as on x86_64, `as.numeric()` can double-round a sixteen-digit
literal and so disagree with a database over a literal the package considers
valid.

### Verification

`noLD` cannot be reached from the submission machine, so the failing reader was
reproduced rather than assumed. R's `R_strtod` accumulation loop was
re-implemented in R with `LDOUBLE` standing in as a plain `double`, and the
package's own `as.numeric()` calls were rebound to it.

* That simulation reproduces the CRAN failure exactly: with the 1.13.4 helper
  it fails `test-obwoe-sql.R:147:3` and `test-obwoe-sql.R:148:3`, the same two
  assertions, with the same wrong bin for the same observation. With the new
  helper both pass.
* Every numeric literal the test suite actually reads — 166 distinct values —
  was decoded under that simulation. Eleven are misread by `as.numeric()`
  there; the new reader recovers all 166 exactly.
* A wider sweep of 40,611 doubles spanning 1e-30 to 1e30, including magnitudes
  that need more than 22 decimals and values whose seventeenth significant
  digit is zero, was written by `obwoe_sql()`'s literal writer and read back:
  exact in every case, both natively and under the simulated `noLD` reader.
* The full test suite passes under the simulated reader.

One regression test was added to `tests/testthat/test-obwoe-sql.R`, asserting
that the evaluator's reader recovers the identical double from the literal
`obwoe_sql()` writes, over 3,600 values. The boundary tests are only as
trustworthy as that reader, and on `noLD` this is what fails first if it
regresses.

---

## R CMD check results

### Local check

x86_64-pc-linux-gnu, R 4.6.x, GCC, `R CMD check --as-cran --run-donttest` on
the built tarball with vignettes:

```
0 errors | 0 warnings | 2 notes
```

Tests, examples (including `--run-donttest`) and all three vignettes build and
run cleanly. The test suite runs 2,048 assertions with no failures and no
warnings.

**NOTE 1 — `checking CRAN incoming feasibility`: `Days since last update`.**
Expected. This release exists to repair the failing `noLD` check reported
against 1.13.4.

**NOTE 2 — `checking HTML version of manual`.** The check machine has no
`tidy` binary and no `V8` package, so HTML validation and math rendering were
skipped. This is a property of the machine, not of the package.

### INFO — installed size

```
installed size is 72.3Mb
sub-directories of 1Mb or more:
  libs  70.6Mb
```

Unchanged from 1.13.4, and explained there: the package compiles 37 binning
algorithms as separate translation units, and essentially all of the reported
size is debug symbols from the local `-g` compiler default. The same object
after `strip --strip-debug` is 2.9 MB. We have not added `-Os` or an explicit
strip step to `src/Makevars`, since CRAN policy asks packages not to override
the platform's compiler flags.

---

## Test environments

* **Local**: x86_64-pc-linux-gnu, R 4.6.x, GCC — `0 errors | 0 warnings | 2 notes`
* **Local, simulated `noLD` reader**: the full test suite, as described above.
* **GitHub Actions**: ubuntu-latest (R devel, release, oldrel-1, oldrel-2,
  oldrel-3), windows-latest (R release) and macos-latest (R release, Apple
  silicon).
* **win-builder (R-release and R-devel)**: to be verified before submission.

---

## Dependencies

No change. No new `Imports`, `Depends` or `Suggests`.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
