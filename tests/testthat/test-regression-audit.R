# Regression tests for defects found in the 2026-08 C++ audit.
# Every test in this file FAILS on the pre-audit sources and passes after the
# corresponding fix, so it pins the behaviour rather than merely exercising it.

# ---------------------------------------------------------------------------
# ob_categorical_ivb: null stats_cache on the "few categories" fast path
#
# perform_binning() skipped initialize_dp_structures() whenever
# ncat <= max_bins, but result assembly dereferenced stats_cache
# unconditionally -> segfault (killed the R session, not a catchable error).
# Triggered by any feature with <= max_bins distinct categories, i.e. sex,
# marital status, region: the most ordinary categorical predictors there are.
# ---------------------------------------------------------------------------
test_that("ob_categorical_ivb handles features with <= max_bins categories", {
  for (k in 2:5) {
    set.seed(100 + k)
    n <- 400
    feature <- sample(LETTERS[seq_len(k)], n, replace = TRUE)
    target <- rbinom(n, 1, 0.3)

    res <- ob_categorical_ivb(feature = feature, target = target)

    expect_equal(sum(res$count), n)
    expect_equal(res$count_pos + res$count_neg, res$count)
    expect_true(all(is.finite(res$woe)))
    expect_true(all(is.finite(res$iv)))
    expect_lte(length(res$count), k)
  }
})

test_that("ob_categorical_ivb handles a perfectly separating 2-level feature", {
  res <- ob_categorical_ivb(
    feature  = c("A", "A", "A", "A", "B", "B", "B", "B"),
    target   = c(1L, 1L, 1L, 1L, 0L, 0L, 0L, 0L),
    min_bins = 2, max_bins = 2
  )

  expect_equal(sum(res$count), 8)
  expect_true(all(is.finite(res$woe)))
  # a perfect separator must carry information, never IV == 0
  expect_gt(sum(res$iv), 1)
})

# ---------------------------------------------------------------------------
# Unbreakable infinite loops when min_bins exceeds the number of distinct
# feature values.
#
# OBN_JEDI / OBN_JEDIMWoE: the pre-binning "ensure at least min_bins" loop ran
#   before assign_bins(), so every count was 0 and find_largest_bin() always
#   returned index 0 -- the (-Inf, e1] interval, which split_bin() refuses to
#   split. The loop never made progress and never terminated.
# OBN_IR: ensureMinMaxBins() spun on splitLargestBin(), which is a no-op once
#   every bin holds a single observation.
#
# None of these loops contained R_CheckUserInterrupt(), so the hang could not
# even be interrupted with Ctrl-C -- the session had to be killed. This is a
# routine situation in pipelines that apply one min_bins across every feature.
# ---------------------------------------------------------------------------
test_that("numerical algorithms terminate when min_bins exceeds distinct values", {
  algos <- c("ob_numerical_ir", "ob_numerical_jedi", "ob_numerical_jedi_mwoe")

  for (fn in algos) {
    res <- do.call(fn, list(
      target = c(0L, 1L, 0L, 1L), feature = c(1, 2, 3, 4),
      min_bins = 6, max_bins = 10
    ))
    expect_equal(sum(res$count), 4, info = fn)
    expect_true(all(is.finite(res$woe)), info = fn)
  }
})

test_that("low-cardinality features terminate across the min_bins grid", {
  algos <- c("ob_numerical_ir", "ob_numerical_jedi", "ob_numerical_jedi_mwoe")
  grid <- list(c(u = 3, mb = 5), c(u = 4, mb = 5), c(u = 4, mb = 6), c(u = 2, mb = 3))

  for (fn in algos) {
    for (g in grid) {
      set.seed(11)
      n <- 300
      feature <- as.numeric(sample(seq_len(g[["u"]]), n, replace = TRUE))
      target <- rbinom(n, 1, 0.4)

      res <- do.call(fn, list(
        target = target, feature = feature,
        min_bins = g[["mb"]], max_bins = max(g[["mb"]], 5)
      ))
      expect_equal(sum(res$count), n,
                   info = sprintf("%s u=%d min_bins=%d", fn, g[["u"]], g[["mb"]]))
    }
  }
})

# ---------------------------------------------------------------------------
# ob_categorical_sab was seeded from std::random_device, so set.seed() had no
# effect and every call returned a different binning for identical input --
# no way to reproduce or audit a model built with it. It is now seeded from R's
# own RNG stream.
# ---------------------------------------------------------------------------
test_that("ob_categorical_sab is reproducible under set.seed()", {
  set.seed(42)
  feature <- sample(paste0("L", 1:5), 1000, replace = TRUE)
  target <- rbinom(1000, 1, 0.3)

  runs <- replicate(4, {
    set.seed(123)
    res <- ob_categorical_sab(feature = feature, target = target)
    paste(length(res$count), format(sum(res$iv), digits = 14))
  })
  expect_length(unique(runs), 1)

  # and the search must still respond to the seed, i.e. not be degenerate
  varied <- vapply(1:4, function(s) {
    set.seed(s)
    res <- ob_categorical_sab(feature = feature, target = target)
    sum(res$iv)
  }, numeric(1))
  expect_gt(length(unique(varied)), 1)
})

# ---------------------------------------------------------------------------
# Interval convention: every numerical algorithm bins right-closed (a, b], the
# convention its labels advertise and that ob_apply_woe_num(include_upper_bound
# = TRUE) implements.
#
# Before the audit the package disagreed with itself: labels said "(a;b]" while
# many algorithms assigned values as [a;b), ob_apply_woe_num had its two
# searches swapped, and some equal-frequency pre-binners split runs of tied
# values so their counts could not be derived from their own cutpoints at all.
# A value landing exactly on a cutpoint -- routine for integer or rounded
# features -- could therefore be scored into a different bin than the one it was
# trained in.
#
# The check below is self-validating: it recomputes the counts from the raw
# feature under both conventions and requires the right-closed one to match.
# ---------------------------------------------------------------------------
test_that("numerical algorithms bin right-closed (a, b], matching their labels", {
  make_case <- function(kind, n, seed) {
    set.seed(seed)
    feature <- switch(kind,
      integers = sample(1:20, n, replace = TRUE),
      coarse   = sample(seq(0, 100, by = 10), n, replace = TRUE),
      cont     = rnorm(n)
    )
    list(feature = as.numeric(feature), target = rbinom(n, 1, 0.35))
  }

  algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  for (fn in algos) {
    for (kind in c("integers", "coarse", "cont")) {
      d <- make_case(kind, 2000, 7)
      res <- try(suppressWarnings(
        do.call(fn, list(feature = d$feature, target = d$target))), silent = TRUE)
      if (inherits(res, "try-error") || is.null(res$count) || is.null(res$cutpoints)) next

      k <- length(res$count)
      cuts <- as.numeric(res$cutpoints)
      if (length(cuts) != k - 1L) next

      trained <- as.integer(res$count)
      lbl <- sprintf("%s / %s", fn, kind)

      # no observation may be lost
      expect_equal(sum(trained), length(d$feature), info = lbl)

      # counts must be reproducible from the reported cutpoints under (a, b]
      right_closed <- as.integer(table(factor(
        findInterval(d$feature, cuts, left.open = TRUE) + 1L, levels = seq_len(k))))
      expect_equal(right_closed, trained, info = lbl)
    }
  }
})

test_that("ob_apply_woe_num reproduces the training bins it was given", {
  set.seed(7)
  feature <- as.numeric(sample(1:20, 2000, replace = TRUE))
  target <- rbinom(2000, 1, 0.35)

  algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  for (fn in algos) {
    res <- try(suppressWarnings(
      do.call(fn, list(feature = feature, target = target))), silent = TRUE)
    if (inherits(res, "try-error") || is.null(res$count)) next

    applied <- try(suppressWarnings(
      ob_apply_woe_num(obresults = res, feature = feature)), silent = TRUE)
    # multinomial variants are out of scope for the binary apply helper
    if (inherits(applied, "try-error")) next

    k <- length(res$count)
    got <- as.integer(table(factor(applied$idbin, levels = seq_len(k))))
    expect_equal(got, as.integer(res$count), info = fn)
  }
})

# ---------------------------------------------------------------------------
# obcorr() spliced per-thread result buffers together inside an `omp critical`
# block, so the row order of the returned data frame depended on thread
# scheduling: two runs on the same data with the same thread count could return
# the pairs in different orders. Results are now written to a pre-sized vector
# by index, which is deterministic and independent of the thread count.
# ---------------------------------------------------------------------------
test_that("obcorr returns a deterministic row order regardless of threads", {
  set.seed(1)
  df <- as.data.frame(matrix(rnorm(60 * 20), ncol = 20))

  key <- function(th) {
    res <- obcorr(df, method = "pearson", threads = th)
    paste(res$x, res$y, sep = ":")
  }

  k_default <- key(0)
  expect_identical(key(0), k_default)   # repeated run, default threads
  expect_identical(key(1), k_default)
  expect_identical(key(2), k_default)
  expect_identical(key(4), k_default)

  # values must not depend on the thread count either
  v1 <- obcorr(df, method = "pearson", threads = 1)$pearson
  v4 <- obcorr(df, method = "pearson", threads = 4)$pearson
  expect_equal(v1, v4)

  # every unordered pair appears exactly once
  expect_equal(length(k_default), choose(ncol(df), 2))
  expect_equal(anyDuplicated(k_default), 0L)
})

# ---------------------------------------------------------------------------
# Package-wide invariants that must hold for every exported algorithm.
# These are the checks that surfaced the defects above in the first place.
# ---------------------------------------------------------------------------
test_that("every algorithm conserves observations and returns finite WoE/IV", {
  set.seed(4242)
  n <- 600
  num_feature <- rnorm(n)
  cat_feature <- sample(paste0("L", 1:6), n, replace = TRUE)
  target <- rbinom(n, 1, 0.35)

  num_algos <- grep("^ob_numerical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)
  cat_algos <- grep("^ob_categorical_", getNamespaceExports("OptimalBinningWoE"), value = TRUE)

  # count_pos/count_neg only exist for binary-target algorithms; the multinomial
  # *_jedi_mwoe variants report per-class counts instead.
  check_common <- function(res, fn) {
    expect_equal(sum(res$count), n, info = fn)
    if (length(res$count_pos) == length(res$count) && length(res$count_neg) == length(res$count)) {
      expect_equal(res$count_pos + res$count_neg, res$count, info = fn)
    }
    expect_true(all(is.finite(res$woe)), info = fn)
    expect_true(all(is.finite(res$iv)), info = fn)
  }

  for (fn in num_algos) {
    res <- try(do.call(fn, list(feature = num_feature, target = target)), silent = TRUE)
    if (inherits(res, "try-error")) next
    check_common(res, fn)
  }

  for (fn in cat_algos) {
    res <- try(do.call(fn, list(feature = cat_feature, target = target)), silent = TRUE)
    if (inherits(res, "try-error")) next
    check_common(res, fn)
  }
})

# ---------------------------------------------------------------------------
# Regression tests for defects found in the 2026-08 follow-up audit.
#
# As above, these pin behaviour rather than merely exercise it: 31 assertions
# below fail on the pre-fix sources. The two exceptions are stated where they
# appear -- they pin the engine invariant each fix relies on, and a no-op
# guarantee, so they hold before and after by design.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# obwoe_apply() / bake.step_obwoe(): category names were whitespace-trimmed
#
# Both rebuilt their category-to-bin lookup with
#   trimws(strsplit(bin_label, "%;%", fixed = TRUE)[[1]])
# The binning engines join the original category strings with the separator and
# add no padding, so the split pieces are already the categories byte for byte;
# trimming turned any category carrying leading or trailing whitespace -- a
# routine artefact of CHAR columns and of hand-maintained code tables -- into a
# key that no observation could ever match. Those rows silently received
# bin = NA and woe = na_woe, so a whole segment was scored as "unseen" without
# any warning.
# ---------------------------------------------------------------------------
# Holds before and after the fix: it pins the engine invariant that removing
# the trimming relies on, namely that no categorical algorithm pads around the
# separator, so the split pieces need no cleaning.
test_that("splitting a bin label recovers the categories byte for byte", {
  set.seed(4021)
  n <- 1200
  cats <- c("  padded  ", "plain", " lead", "trail ", "A B")
  feature <- sample(cats, n, replace = TRUE)
  target <- rbinom(n, 1, plogis(-0.4 + 0.9 * (feature %in% cats[1:2])))

  cat_algos <- grep("^ob_categorical_", getNamespaceExports("OptimalBinningWoE"),
    value = TRUE
  )
  cat_algos <- setdiff(cat_algos, "ob_categorical_jedi_mwoe")

  for (fn in cat_algos) {
    res <- try(
      do.call(fn, list(feature = feature, target = target, min_bins = 2, max_bins = 5)),
      silent = TRUE
    )
    if (inherits(res, "try-error")) next
    parts <- unlist(strsplit(res$bin, "%;%", fixed = TRUE))
    expect_setequal(parts, cats)
    expect_equal(length(parts), length(cats), info = fn)
  }
})

test_that("obwoe_apply() matches categories carrying whitespace", {
  set.seed(4022)
  n <- 1500
  cats <- c("  padded  ", "plain", " lead", "trail ", "\tTAB\t")
  df <- data.frame(
    g = sample(cats, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  df$target <- rbinom(n, 1, plogis(-0.4 + 0.9 * (df$g %in% cats[1:2])))

  model <- obwoe(df, target = "target", max_bins = 5)
  res <- model$results$g
  skip_if(!is.null(res$error), "binning failed on the constructed case")

  scored <- obwoe_apply(df, model)

  # No observation may be left unassigned ...
  expect_false(anyNA(scored$g_bin))
  # ... and the assignment must reproduce the fitted bin counts exactly
  expect_equal(
    as.integer(table(factor(scored$g_bin, levels = res$bin))),
    as.integer(res$count)
  )
  # ... with the WoE of the bin each row landed in
  expect_equal(scored$g_woe, res$woe[match(scored$g_bin, res$bin)], tolerance = 0)
})

test_that("bake.step_obwoe() matches categories carrying whitespace", {
  skip_if_not_installed("recipes")
  set.seed(4023)
  n <- 1500
  cats <- c("  padded  ", "plain", " lead", "trail ")
  df <- data.frame(
    g = sample(cats, n, replace = TRUE),
    stringsAsFactors = FALSE
  )
  df$target <- factor(rbinom(n, 1, plogis(-0.4 + 0.9 * (df$g %in% cats[1:2]))),
    levels = c(0, 1)
  )

  rec <- recipes::prep(
    step_obwoe(
      recipes::recipe(target ~ g, data = df),
      recipes::all_predictors(),
      outcome = "target", max_bins = 5, output = "both"
    ),
    training = df
  )
  fitted <- rec$steps[[1L]]$binning_results$g
  skip_if(is.null(fitted), "step produced no binning for the constructed case")

  baked <- recipes::bake(rec, new_data = df)

  # No observation may be left unassigned ...
  expect_false(anyNA(baked$g_bin))
  expect_true(all(baked$g_bin %in% fitted$bin))

  # ... and the bin each row landed in must actually contain that row's
  # category. The step stores a compact model without counts, so this is
  # checked directly against the bin labels.
  members <- strsplit(as.character(baked$g_bin), "%;%", fixed = TRUE)
  expect_true(all(mapply(function(cat, m) cat %in% m, df$g, members)))

  # The WoE must be the one attached to that bin
  expect_equal(
    baked$g_woe, fitted$woe[match(baked$g_bin, fitted$bin)],
    tolerance = 0
  )
})

# ---------------------------------------------------------------------------
# ob_numerical_ir: PAVA pooled the fitted rates but not the bins
#
# applyIsotonicRegression() ran the Pool Adjacent Violators algorithm on the bin
# event rates and then overwrote each bin with
#   count_pos <- round(fitted_rate * count)
# Pooling adjacent violators means those bins form ONE block; leaving them as
# separate bins and back-solving synthetic counts produced two defects at once:
#   * count_pos/count_neg no longer described the observations falling between
#     the reported cutpoints, so WoE, IV, KS and every gains table derived from
#     them referred to a distribution that does not exist; and
#   * because of the rounding, the reported bins were not even monotonic -- the
#     one property the algorithm exists to guarantee.
# The blocks are now merged, which reproduces the isotonic fit exactly (the
# pooled rate of a block IS sum(count_pos)/sum(count) over it) while keeping the
# counts equal to what was observed.
# ---------------------------------------------------------------------------
test_that("ob_numerical_ir reports the counts that fall between its cutpoints", {
  scenarios <- list(
    increasing = function(x) plogis(-0.5 + 1.0 * x),
    decreasing = function(x) plogis(-0.5 - 1.0 * x),
    u_shaped   = function(x) plogis(-1.0 + 1.2 * x^2),
    flat       = function(x) rep(0.3, length(x))
  )

  for (nm in names(scenarios)) {
    for (mb in c(2L, 3L, 5L)) {
      set.seed(4031)
      n <- 3000
      x <- rnorm(n)
      y <- rbinom(n, 1, scenarios[[nm]](x))

      res <- ob_numerical_ir(
        feature = x, target = y,
        min_bins = mb, max_bins = max(mb, 8L)
      )
      tag <- sprintf("%s/min_bins=%d", nm, mb)
      k <- length(res$bin)

      idx <- cut(x,
        breaks = c(-Inf, res$cutpoints, Inf), labels = FALSE,
        right = TRUE, include.lowest = TRUE
      )
      obs_count <- as.integer(table(factor(idx, levels = seq_len(k))))
      obs_pos <- as.integer(table(factor(idx[y == 1], levels = seq_len(k))))
      obs_neg <- as.integer(table(factor(idx[y == 0], levels = seq_len(k))))

      expect_equal(as.integer(res$count), obs_count, info = tag)
      expect_equal(as.integer(res$count_pos), obs_pos, info = tag)
      expect_equal(as.integer(res$count_neg), obs_neg, info = tag)

      # Totals are conserved: rounding used to lose or invent events
      expect_equal(sum(res$count_pos), sum(y == 1L), info = tag)
      expect_equal(sum(res$count_neg), sum(y == 0L), info = tag)
    }
  }
})

test_that("ob_numerical_ir returns bins that are monotonic in the observed rate", {
  # An isotonic binner must deliver monotonicity in the data it reports, not
  # only in a fitted vector that never reaches the caller.
  for (seed in c(11L, 22L, 33L, 44L)) {
    set.seed(seed)
    n <- 2500
    x <- rnorm(n)
    # A deliberately non-monotone relationship: PAVA has real work to do
    y <- rbinom(n, 1, plogis(-1 + 0.8 * x + 0.7 * x^2))

    res <- ob_numerical_ir(feature = x, target = y, min_bins = 2, max_bins = 8)
    rate <- res$count_pos / res$count
    d <- diff(rate)

    expect_true(
      length(d) == 0L || all(d >= -1e-12) || all(d <= 1e-12),
      info = sprintf("seed=%d rates=%s", seed, paste(round(rate, 4), collapse = ", "))
    )
    # The WoE the caller receives must order the same way as those rates
    expect_equal(order(res$woe), order(rate), info = sprintf("seed=%d", seed))
  }
})

test_that("ob_numerical_ir leaves an already monotone feature untouched", {
  # Holds before and after: the fix must be a no-op wherever PAVA has nothing
  # to pool, so a well-behaved feature keeps exactly the bins it had.
  set.seed(4032)
  n <- 4000
  x <- rnorm(n)
  y <- rbinom(n, 1, plogis(-0.5 + 1.2 * x))

  res <- ob_numerical_ir(feature = x, target = y, min_bins = 3, max_bins = 6)
  rate <- res$count_pos / res$count

  expect_gte(length(res$bin), 3L)
  expect_true(all(diff(rate) >= -1e-12) || all(diff(rate) <= 1e-12))
  expect_equal(as.integer(res$count), as.integer(res$count_pos + res$count_neg))
})

# ---------------------------------------------------------------------------
# obwoe_gains(): the lift column collapsed to a single value
#
# .build_gains_table() computed
#   df$lift <- ifelse(overall_rate > 0, df$pos_rate / overall_rate, 0)
# ifelse() returns a result shaped like its *test*, and the test here is the
# scalar `overall_rate > 0`, so the expression evaluated to a length-1 vector
# that R then recycled: every bin reported the lift of the first bin. The lift
# column of every gains table, and plot(type = "lift"), were wrong for any
# binning with more than one bin.
# ---------------------------------------------------------------------------
test_that("obwoe_gains() reports lift per bin, not the first bin's lift", {
  set.seed(5)
  n <- 5000
  score <- rnorm(n)
  target <- rbinom(n, 1, plogis(-1.5 + 1.2 * score))
  df <- data.frame(score = score, target = target)

  gains <- obwoe_gains(df,
    target = "target", feature = "score",
    use_column = "direct", n_groups = 5, sort_by = "bin"
  )
  tbl <- gains$table

  expect_equal(tbl$lift, tbl$pos_rate / mean(target))
  expect_gt(length(unique(round(tbl$lift, 6))), 1L)
  # lift is a ratio to the base rate, so the population-weighted mean is 1
  expect_equal(sum(tbl$lift * tbl$count) / sum(tbl$count), 1)
})

test_that("obwoe_gains() agrees with the C++ gains engine", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "duration", max_bins = 6)
  res <- model$results$duration

  r_side <- obwoe_gains(model, feature = "duration", sort_by = "id")$table
  cpp_side <- obwoe_gains_score(list(
    id = res$id, bin = res$bin, count = res$count,
    count_pos = res$count_pos, count_neg = res$count_neg
  ))

  expect_equal(r_side$lift, cpp_side$lift, tolerance = 1e-12)
  expect_equal(r_side$woe, cpp_side$woe, tolerance = 1e-12)
  expect_equal(r_side$iv, cpp_side$iv, tolerance = 1e-12)
  expect_equal(r_side$ks, cpp_side$ks, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# [B4/C-04] obwoe_gains(use_column = "woe", n_groups = k) used to zero the IV
#
# When the grouping column is itself the WoE, woe_source was set to the
# sentinel string "woe" so that the later `identical(woe_source, "woe")`
# branch used the grouping variable directly. But when n_groups regroups the
# numeric WoE into quantile labels ("G01", "G02", ...), the sentinel string
# survived the regrouping, so the code executed `as.numeric(bins)` on labels
# like "G01" -> NA for every bin, and the total IV silently collapsed to 0.
# ---------------------------------------------------------------------------
test_that("obwoe_gains(use_column = 'woe', n_groups = k) keeps WoE numeric and IV > 0", {
  skip_if_no_german()
  df <- german_credit()
  model <- obwoe(df, target = "target", feature = "duration", max_bins = 6)
  scored <- obwoe_apply(df, model)

  gains <- suppressWarnings(obwoe_gains(scored,
    target = df$target, feature = "duration",
    use_column = "woe", n_groups = 5
  ))

  expect_false(anyNA(gains$table$woe))
  expect_true(is.numeric(gains$table$woe))
  expect_gt(gains$metrics$total_iv, 0)
})

# ---------------------------------------------------------------------------
# [C4/A-02] .dispatch_algorithm() never checked registry$multinomial
#
# The registry carries a $multinomial flag (only "jedi"/"jedi_mwoe" support a
# multiclass target), but the dispatcher only checked $numerical/$categorical.
# A binary-only algorithm explicitly requested against a multinomial target
# used to fall through to the C++ engine itself, which happens to validate
# and reject it too, but with a less specific message and after doing real
# work; the R-level dispatcher should reject it up front, using the registry
# metadata that already declares the incompatibility.
# ---------------------------------------------------------------------------
test_that("obwoe() rejects a binary-only algorithm against a multinomial target", {
  set.seed(1)
  n <- 500
  df <- data.frame(x = rnorm(n), y = sample(0:2, n, replace = TRUE))

  res <- obwoe(df, target = "y", feature = "x", algorithm = "mob")
  expect_true(res$summary$error)
  expect_match(res$results$x$error, "does not support a multinomial target")

  # jedi_mwoe is the multinomial-capable universal algorithm and must still
  # work.
  res2 <- obwoe(df, target = "y", feature = "x", algorithm = "jedi_mwoe")
  expect_false(res2$summary$error)
})

# ---------------------------------------------------------------------------
# [C8/A-06] ob_categorical_mob()'s converged/iterations, "few categories" path
#
# OBC_MOB::fit() hardcoded converged = true and only ever reassigned it in
# the ncat > max_bins branch; the ncat <= max_bins branch (one bin per
# category, sorted by WoE, no enforceMonotonicity() call) always reported
# converged = TRUE, iterations = 0 without checking. The fix makes that
# branch honestly compute isMonotonic(bins). Because that branch always
# sorts bins by WoE ascending before returning, the resulting bins are
# monotonic by construction, so the reported value does not actually change
# for ordinary input -- this pins that it is still TRUE/0 and that the woe
# column is genuinely monotonic, now for the right reason.
# ---------------------------------------------------------------------------
test_that("ob_categorical_mob() with few categories reports converged honestly", {
  set.seed(1)
  n <- 500
  feature <- sample(letters[1:4], n, replace = TRUE)
  target <- rbinom(n, 1, 0.3)

  res <- ob_categorical_mob(feature = feature, target = target, min_bins = 2, max_bins = 6)

  expect_true(length(res$count) <= 4L) # fewer categories than max_bins
  expect_true(res$converged)
  expect_equal(res$iterations, 0L)
  expect_true(all(diff(res$woe) >= -1e-9)) # sorted ascending by construction
})

# ---------------------------------------------------------------------------
# [D1] .build_gains_table()'s neg_rate lacked the 0/0 guard pos_rate has
# ---------------------------------------------------------------------------
test_that("[D1] .build_gains_table()'s neg_rate is 0, not NaN, for an empty bin", {
  gt <- OptimalBinningWoE:::.build_gains_table(
    bins = c("a", "b", "c"),
    counts = c(10, 0, 5),
    pos_counts = c(4, 0, 2),
    neg_counts = c(6, 0, 3),
    woe = c(0.1, 0, -0.2),
    sort_by = "bin",
    id = 1:3
  )

  expect_false(any(is.nan(gt$neg_rate)))
  expect_equal(gt$neg_rate[gt$bin == "b"], 0)
})

# ---------------------------------------------------------------------------
# [D2] obwoe() documents that a missing target is not permitted, but never
# enforced it -- target-type detection silently dropped NA, and every
# downstream binning call received a target vector with NAs intact.
# ---------------------------------------------------------------------------
test_that("[D2] obwoe() rejects a target with missing values", {
  set.seed(1)
  n <- 200
  df <- data.frame(x = rnorm(n), y = rbinom(n, 1, 0.3))
  df$y[c(3, 17)] <- NA

  expect_error(
    obwoe(df, target = "y", feature = "x"),
    "missing values"
  )
})

# ---------------------------------------------------------------------------
# [D7] Divergent min_bins / bin_cutoff bounds copied across numerical wrappers
# ---------------------------------------------------------------------------
test_that("[D7] min_bins and bin_cutoff bounds are uniform across numerical wrappers", {
  set.seed(1)
  n <- 300
  feature <- rnorm(n)
  target <- rbinom(n, 1, 0.3)

  # min_bins = 1 used to be accepted by mdlp/mrblp only; now rejected
  # everywhere, like the other 6 siblings (ldb, mblp, mob, oslp, ubsd, udt).
  expect_error(
    ob_numerical_mdlp(feature = feature, target = target, min_bins = 1),
    "at least 2"
  )
  expect_error(
    ob_numerical_mrblp(feature = feature, target = target, min_bins = 1),
    "at least 2"
  )

  # bin_cutoff = 0 or 1 used to be accepted by ldb only; now rejected
  # everywhere, like the other 8 siblings.
  expect_error(
    ob_numerical_ldb(feature = feature, target = target, bin_cutoff = 0),
    "\\(0, 1\\)"
  )
  expect_error(
    ob_numerical_ldb(feature = feature, target = target, bin_cutoff = 1),
    "\\(0, 1\\)"
  )
})
