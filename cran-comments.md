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

`obwoe_sql()` writes each cut point as the shortest fixed-notation decimal
string that parses back to the identical double, so that an observation on a
boundary cannot drift into the neighbouring bin. The old implementation found
that string by asking `format(v, digits = dg, scientific = FALSE)` for
`dg = 1, ..., 17` and returning the first one that satisfied
`as.numeric(s) == v`, falling back to `digits = 22`.

`format()`'s significant-digit search runs in `long double`. On aarch64 macOS
`long double` is no wider than `double`, so the search can conclude that fewer
digits suffice than actually do; every width from 1 to 17 then fails the
round-trip test and the fallback returns the same short string. The cut point
`-0.13964785691628961` was written as `-0.13964785691629`, which is the
smaller number, so the observation equal to the cut point failed
`x <= -0.13964785691629` and was scored one bin up. On x86_64 the 80-bit
extended long double leaves the search enough precision to terminate
correctly, which is why the failure was confined to one architecture.

Literals are now built with `sprintf("%.*f", ...)`, which delegates to the C
library's correctly rounded binary-to-decimal conversion and behaves the same
way on every platform R supports. The round-trip check is unchanged, and the
search is now over decimal places rather than over `format()`'s notion of
significant digits.

The same commit fixes a second, smaller defect found while auditing that code
path: `digits` rounded to the requested number of decimal places and then
formatted the result with R's default seven significant digits, so
`digits = 8` on `1234.5678901234` emitted `1234.568`.

### Verification

The emitted SQL is byte-identical to 1.13.3 on x86_64 for 33,000 random and
adversarial doubles — the fix changes *which* code produces the digits, not
the digits themselves, on a platform where the old code was already correct.
The only difference found is for denormals, where the old code emitted
scientific notation that the documentation promises never to use.

Two regression tests were added to `tests/testthat/test-obwoe-sql.R`:

* a bulk round trip over 3,500 values, which asserts
  `as.numeric(.ob_sql_num(v)) == v` exactly; and
* a boundary check on cut points taken from continuous data.

The pre-existing boundary test used a feature whose cut points are small
integers, which any literal writer renders exactly, and so could not catch
this class of defect.

We have no aarch64 macOS machine, so the failing configuration was reproduced
by inspection rather than by execution: the value that fails, the string
`format()` must have produced for the observed result, and the resulting bin
assignment are all shown above and were confirmed on x86_64 by feeding the
short literal back through the comparison. The fix removes the platform
dependence entirely rather than compensating for it.

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
* **macOS builder (aarch64, R release)**: to be verified before submission —
  this is the configuration that failed on 1.13.3, so it is the one that
  matters for this release.
* **win-builder (R-release and R-devel)**: to be verified before submission

---

## Dependencies

No change. No new `Imports`, `Depends` or `Suggests`.

---

## Downstream dependencies

This package currently has no reverse dependencies on CRAN.
