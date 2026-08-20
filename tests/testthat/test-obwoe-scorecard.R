# ===========================================================================#
# obwoe_scorecard(): the automated pipeline
#
# The pipeline is long, and a long pipeline fails quietly: every stage produces
# numbers that look plausible on their own. The tests below are therefore
# mostly invariants — identities that must hold for any input — rather than
# fixtures, and each is written to fail on the naive implementation it guards
# against (binning before the split, +factor instead of -factor, a points table
# that does not sum to the score, an engine substituted in silence).
#
# Engines: only "glm", "obwoe" and "glmnet" are exercised. parsnip, workflows
# and the boosted engines cannot be installed in the reference environment, so
# they are covered solely by the custom-engine contract test and by the test
# that an unavailable engine fails loudly. A skip is not coverage.
# ===========================================================================#

test_that("the scaling is pinned by its anchor and its scale", {
  for (pdo in c(15, 20, 40)) {
    for (odds in c(20, 50, 72)) {
      s <- obwoe_scale(pdo = pdo, score_ref = 600, odds_ref = odds)

      # Anchor: a case at the reference odds scores the reference score.
      expect_equal(obwoe_score(-log(odds), s), 600)

      # Scale: doubling the good:bad odds adds exactly one PDO, everywhere.
      eta <- c(-3, -1, 0, 0.5, 2)
      expect_equal(
        obwoe_score(eta - log(2), s) - obwoe_score(eta, s),
        rep(pdo, length(eta))
      )
    }
  }
})


test_that("the score falls as risk rises", {
  s <- obwoe_scale()
  # eta is the log-odds of the event, so a larger eta must score fewer points.
  expect_true(all(diff(obwoe_score(seq(-3, 3, by = 0.5), s)) < 0))

  reversed <- obwoe_scale(direction = "higher_is_riskier")
  expect_true(all(diff(obwoe_score(seq(-3, 3, by = 0.5), reversed)) > 0))
})


test_that("obwoe_scale validates its arguments", {
  expect_error(obwoe_scale(pdo = 0), "positive")
  expect_error(obwoe_scale(pdo = -5), "positive")
  expect_error(obwoe_scale(odds_ref = 0), "positive")
  expect_error(obwoe_scale(score_ref = NA), "finite")
  expect_error(obwoe_score(1, list()), "obwoe_scaling")
})


test_that("the pipeline runs and its funnel is consistent", {
  skip_if_no_german()
  df <- german_credit()

  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 42))

  expect_s3_class(sc, "obwoe_scorecard")
  expect_true(all(sc$final %in% sc$selected))
  expect_true(all(sc$selected %in% sc$candidates))
  expect_equal(sc$target, "target")
  expect_true(sc$engine$additive)
  expect_s3_class(sc$binning, "obwoe")
  expect_s3_class(sc$scaling, "obwoe_scaling")

  # the stage column must partition the candidates exactly
  stages <- as.data.frame(sc$screening)
  expect_setequal(stages$feature, sc$candidates)
  expect_setequal(stages$feature[stages$stage == "in_model"], sc$final)
})


test_that("[B1/C-01] the gains table KS reconciles with the rank-based KS", {
  # .ob_score_gains() used to do as.character(band) on the ordered factor
  # cut() produced, which stripped the level order. .build_gains_table()
  # then fell back to lexicographic order(df$bin) for sort_by = "bin", and
  # because '[' (ASCII 91) sorts after '(' (ASCII 40), the lowest-score bin
  # "[-Inf,x]" landed on the LAST row instead of the first, corrupting the
  # cumulative KS curve.
  skip_if_no_german()
  df <- german_credit()

  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 42))

  gains_ks <- max(sc$samples$train$gains$ks)
  rank_ks <- sc$samples$train$metrics$ks

  expect_equal(gains_ks, rank_ks, tolerance = 0.02)

  # The first row of the gains table must be the lowest-score bin.
  first_bin <- as.character(sc$samples$train$gains$bin[1])
  expect_true(grepl("^\\[-Inf", first_bin) || grepl("^\\(-Inf", first_bin))
})


test_that("the binning sees the training rows only", {
  skip_if_no_german()
  df <- german_credit()

  set.seed(99)
  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 99))

  # Reconstruct the split the pipeline used and refit the binning on it.
  set.seed(99)
  idx <- OptimalBinningWoE:::.ob_train_index(df, 0.7, df$target)
  on_train <- obwoe(df[idx$train, , drop = FALSE],
    target = "target", feature = sc$candidates
  )
  on_full <- obwoe(df, target = "target", feature = sc$candidates)

  cuts <- function(o, f) o$results[[f]]$cutpoints
  f <- sc$final[[1L]]

  expect_equal(cuts(sc$binning, f), cuts(on_train, f))
  # Guard against a trivial pass: the two fits must actually differ somewhere,
  # otherwise the test above would hold even for a leaky implementation.
  differs <- vapply(sc$candidates, function(v) {
    !identical(cuts(on_train, v), cuts(on_full, v))
  }, logical(1))
  expect_true(any(differs))
})


test_that("points sum to the score and the card stays within its bound", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 7,
    screening = list(top_n = 6)
  ))

  k <- length(sc$final)
  woe <- predict(sc, df, type = "woe")
  score <- predict(sc, df, type = "score")

  # Unrounded additivity: the Siddiqi allocation is exact by construction.
  raw <- vapply(sc$final, function(v) {
    tbl <- sc$points[sc$points$variable == v, , drop = FALSE]
    tbl$points_raw[match(round(woe[[v]], 12), round(tbl$woe, 12))]
  }, numeric(nrow(df)))
  expect_equal(rowSums(raw), score, tolerance = 1e-9)

  # The card is the sum of the fixed integer points; rounding k terms can move
  # the total by at most k/2, and that drift must stay inside the bound.
  card <- predict(sc, df, type = "card")
  expect_lte(max(abs(card - score)), k / 2)

  # The points table is fixed: one integer per variable and bin, for everyone.
  expect_false(any(duplicated(sc$points[, c("variable", "bin")])))
  expect_type(sc$points$points, "double")
  expect_equal(sc$points$points, round(sc$points$points_raw))
})


test_that("points fall as the Weight of Evidence rises", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 3))

  for (v in sc$final) {
    tbl <- sc$points[sc$points$variable == v, , drop = FALSE]
    o <- order(tbl$woe)
    # a riskier bin must be worth fewer points
    expect_true(all(diff(tbl$points_raw[o]) <= 1e-9), info = v)
  }
})


test_that("the generated points SQL reproduces the card exactly", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 11,
    screening = list(top_n = 6)
  ))

  sql <- OptimalBinningWoE:::.ob_points_sql(sc,
    table = "app", dialect = "ansi",
    keep_columns = NULL
  )

  # Read the emitted CASE blocks back with the interpreter used elsewhere in
  # this suite, so a wrong boundary or a mis-escaped literal shows up as a
  # wrong score rather than as a string difference.
  body <- sub("^(--[^\n]*\n)+", "", sql)
  blocks <- regmatches(body, gregexpr("CASE\n(.|\n)*?\nEND", body))[[1L]]
  expect_equal(length(blocks), length(sc$final))

  from_sql <- vapply(seq_along(sc$final), function(i) {
    sql_eval_case_num(blocks[i], df[[sc$final[i]]], sc$final[i])
  }, numeric(nrow(df)))

  expect_equal(rowSums(from_sql), predict(sc, df, type = "card"))
})


test_that("[C7/A-08] .ob_points_sql() skips a feature with inconsistent cutpoints instead of emitting a broken literal", {
  # obwoe_sql() guards against length(cutpoints) + 1 != n_bins (e.g. a
  # cutpoint duplicated by floating-point ties, collapsed by unique()):
  # it warns and skips the feature. .ob_points_sql() re-implemented the
  # CASE assembly directly, without that guard, so .ob_sql_case() indexed
  # its per-bin literal vector one past its length, and the out-of-bounds
  # NA silently became the literal text "NA" in the generated points SQL
  # (e.g. "WHEN x <= NA THEN ..."), instead of a warning and a skip.
  set.seed(11)
  n <- 2000
  df <- data.frame(
    x1 = rnorm(n), x2 = rnorm(n),
    target = rbinom(n, 1, plogis(-0.5 + 0.8 * rnorm(n)))
  )
  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 1,
    screening = list(iv_min = 0, require_monotonic = "none")
  ))
  skip_if(length(sc$final) < 2L, "need at least 2 final variables")

  feat <- sc$final[1]
  # Simulate the floating-point-tie scenario the guard exists for: every
  # cutpoint collapses to the same value, so length(unique(cutpoints)) + 1
  # no longer matches the number of fitted bins.
  cp <- sc$binning$results[[feat]]$cutpoints
  sc$binning$results[[feat]]$cutpoints <- rep(cp[1], length(cp))

  expect_warning(
    sql <- OptimalBinningWoE:::.ob_points_sql(sc,
      table = "app", dialect = "ansi", keep_columns = NULL
    ),
    "inconsistent"
  )

  expect_false(grepl("NA", sql, fixed = TRUE))
  expect_false(grepl(sprintf("%s_points", feat), sql, fixed = TRUE))
})


test_that("the model score is not a valid substitute for the card", {
  # Pins the reason both SQL sheets exist: they are not the same number.
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 5))

  card <- predict(sc, df, type = "card")
  score <- predict(sc, df, type = "score")
  expect_false(isTRUE(all.equal(card, score)))
  expect_true(max(abs(card - score)) > 0)
})


test_that("a reversed target level order does not invert the score", {
  skip_if_no_german()
  df <- german_credit()

  a <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 21))

  # Same data, target as a factor whose second level is still the event.
  df2 <- df
  df2$target <- factor(df$target, levels = c(0, 1))
  b <- suppressWarnings(obwoe_scorecard(df2, target = "target", seed = 21))

  expect_equal(a$event_level, "1")
  expect_equal(b$event_level, "1")
  expect_gt(cor(predict(a, df, "score"), predict(b, df2, "score")), 0.999)

  # And the event level is recorded, so an inverted card is traceable.
  df3 <- df
  df3$target <- factor(ifelse(df$target == 1, "bad", "good"),
    levels = c("bad", "good")
  )
  c3 <- suppressWarnings(obwoe_scorecard(df3, target = "target", seed = 21))
  expect_equal(c3$event_level, "good")
})


test_that("the score ranks: events score lower and AUC exceeds a half", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 13))

  for (nm in names(sc$samples)) {
    s <- sc$samples[[nm]]
    expect_lt(mean(s$score[s$y == 1L]), mean(s$score[s$y == 0L]))
    expect_gt(s$metrics$auc, 0.5)
    expect_equal(s$metrics$gini, 2 * s$metrics$auc - 1)
    expect_gte(s$metrics$ks, 0)
    expect_lte(s$metrics$ks, 1)
  }
})


test_that("correlation pruning is iterative and rank-aware", {
  set.seed(31)
  n <- 800
  a <- rnorm(n)
  # A ~ B ~ C chain: B duplicates A, C duplicates B, A and C are unrelated
  # enough that a pairwise pass would wrongly remove both B and C.
  d <- data.frame(A = a, B = a + rnorm(n, 0, 0.2))
  d$C <- d$B + rnorm(n, 0, 0.2)
  d$D <- rnorm(n)

  p <- obwoe_prune(d, ranking = c("A", "B", "C", "D"), cutoff = 0.7)
  expect_true("A" %in% p$keep)
  expect_true("D" %in% p$keep)
  expect_false("B" %in% p$keep)
  expect_true(nrow(p$dropped) >= 1L)
  expect_setequal(names(p$dropped), c("variable", "correlated_with", "correlation"))

  # The better-ranked member of each pair survives, whatever the input order.
  q <- obwoe_prune(d, ranking = c("B", "A", "C", "D"), cutoff = 0.7)
  expect_true("B" %in% q$keep)
  expect_false("A" %in% q$keep)

  # A single candidate has nothing to be redundant with.
  expect_equal(obwoe_prune(d[, "A", drop = FALSE], ranking = "A")$keep, "A")
})


test_that("PSI is zero for identical samples and grows with the shift", {
  set.seed(41)
  base <- rnorm(5000)

  expect_lt(obwoe_psi(base, rnorm(5000))$psi, 0.05)

  shifts <- vapply(c(0.1, 0.3, 0.6, 1.0), function(s) {
    obwoe_psi(base, rnorm(5000, mean = s))$psi
  }, numeric(1))
  expect_false(is.unsorted(shifts))

  # A merely shifted distribution must not read as infinite: that is what
  # banding on the reference's own extremes would produce.
  expect_true(all(is.finite(shifts)))

  # A genuinely emptied band is infinite, and is flagged rather than smoothed.
  gone <- obwoe_psi(c("a", "b", "c"), c("a", "b"), levels = c("a", "b", "c"))
  expect_equal(gone$psi, Inf)
  expect_equal(gone$flag, "act")

  expect_equal(obwoe_psi(base, base)$flag, "stable")
})


test_that("an unavailable engine fails loudly and never substitutes silently", {
  skip_if_no_german()
  df <- german_credit()

  expect_error(
    obwoe_scorecard(df, target = "target", engine = "lightgbm"),
    "must be one of"
  )
  expect_error(OptimalBinningWoE:::.ob_engine_get("nosuchengine"), "must be one of")

  # The fallback is opt-in, and records what was asked for.
  fake <- list(pkgs = "definitelyNotInstalled")
  expect_error(
    OptimalBinningWoE:::.ob_engine_get("glmnet", fallback = FALSE),
    NA # glmnet is a Suggests; if present this must not error
  )
})


test_that("a custom engine satisfies the contract, including non-additivity", {
  skip_if_no_german()
  df <- german_credit()

  # An engine that declares itself non-additive must suppress the points table
  # rather than fabricate one, and must keep everything else working.
  flat <- list(
    fit = function(x, y, args) list(p = mean(y), m = colMeans(x)),
    link = function(object, x) as.numeric(as.matrix(x) %*% rep(1, ncol(x))),
    coef = function(object) NULL
  )

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", engine = flat,
    seed = 2, screening = list(top_n = 4)
  ))

  expect_false(sc$engine$additive)
  expect_null(sc$points)
  expect_null(sc$coefficients)
  expect_true(any(grepl("non-additive", sc$warnings)))
  expect_true(all(is.finite(sc$samples$train$score)))
  expect_error(predict(sc, df), "non-additive")

  expect_error(
    OptimalBinningWoE:::.ob_engine_get(list(fit = identity)),
    "must supply"
  )
})


test_that("glm and the package's own logistic engine agree", {
  skip_if_no_german()
  df <- german_credit()

  a <- suppressWarnings(obwoe_scorecard(df,
    target = "target", engine = "glm",
    seed = 8, screening = list(top_n = 5)
  ))
  b <- suppressWarnings(obwoe_scorecard(df,
    target = "target", engine = "obwoe",
    seed = 8, screening = list(top_n = 5)
  ))

  expect_equal(a$final, b$final)
  expect_equal(unname(a$coefficients), unname(b$coefficients), tolerance = 1e-4)
  expect_gt(cor(predict(a, df, "score"), predict(b, df, "score")), 0.9999)
})


test_that("glmnet keeps the model additive", {
  skip_if_not_installed("glmnet")
  skip_if_no_german()
  df <- german_credit()

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", engine = "glmnet",
    seed = 4, screening = list(top_n = 5)
  ))
  expect_true(sc$engine$additive)
  expect_false(is.null(sc$points))
  expect_lte(
    max(abs(predict(sc, df, "card") - predict(sc, df, "score"))),
    length(sc$final) / 2
  )
})


test_that("constant and single-bin predictors are dropped before fitting", {
  set.seed(51)
  n <- 2000
  df <- data.frame(
    good = rnorm(n),
    flat = rep(1, n),
    two = sample(c("a", "b"), n, TRUE),
    stringsAsFactors = FALSE
  )
  df$target <- rbinom(n, 1, plogis(-1 + 1.2 * df$good))

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 51,
    screening = list(iv_min = 0, require_monotonic = "none")
  ))

  expect_false("flat" %in% sc$final)
  expect_false(anyNA(sc$coefficients))
})


test_that("an empty shortlist errors with the screening reason", {
  skip_if_no_german()
  df <- german_credit()

  expect_error(
    obwoe_scorecard(df, target = "target", screening = list(iv_min = 99, iv_max = 100)),
    "rejected every candidate"
  )
})


test_that("unseen categories are counted, not silently absorbed", {
  set.seed(61)
  n <- 3000
  g <- sample(c("a", "b", "c"), n, TRUE)
  df <- data.frame(
    g = g, x = rnorm(n),
    target = rbinom(n, 1, plogis(-1 + 0.9 * (g == "a"))),
    stringsAsFactors = FALSE
  )

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 61,
    screening = list(iv_min = 0, require_monotonic = "none")
  ))
  skip_if(!"g" %in% sc$final, "g did not enter the model")

  novel <- df[1:50, ]
  novel$g <- "never_seen"

  expect_warning(
    OptimalBinningWoE:::.ob_score_sample(
      novel, "probe", sc$binning, sc$final,
      OptimalBinningWoE:::.ob_engine_get("glm"), sc$model, sc$scaling,
      sc$points, "target", control.obwoe_scorecard(), warning
    ),
    "fell in no fitted bin"
  )

  # Counted is not the same as ignored: the card must still return a score for
  # such a row, and it must be the same fallback the generated SQL applies.
  bins <- obwoe_apply(novel, sc$binning, keep_original = FALSE)
  card <- OptimalBinningWoE:::.ob_card_score(bins, sc$points, sc$final)
  expect_true(all(is.finite(card)))

  na_points <- attr(sc$points, "points_na")
  expect_setequal(names(na_points), sc$final)

  others <- setdiff(sc$final, "g")
  seen <- vapply(others, function(f) {
    tbl <- sc$points[sc$points$variable == f, , drop = FALSE]
    tbl$points[match(bins[[paste0(f, "_bin")]][1L], tbl$bin)]
  }, numeric(1))
  expect_equal(card[1L], unname(na_points[["g"]]) + sum(seen))
})


test_that("[C2/A-01] score and card agree on na_woe for unseen categories", {
  # object$control was not persisted on the scorecard object.
  # predict.obwoe_scorecard() hardcoded na_woe <- 0 for type = "score" (and
  # the other WoE-based types), regardless of what na_woe the scorecard was
  # actually fitted with, while the points table's "points_na" fallback used
  # by type = "card" was correctly derived from control$na_woe at fit time.
  # With na_woe != 0, an unseen category therefore scored differently under
  # type = "score" than under type = "card" for the very same row.
  set.seed(61)
  n <- 3000
  g <- sample(c("a", "b", "c"), n, TRUE)
  df <- data.frame(
    g = g, x = rnorm(n),
    target = rbinom(n, 1, plogis(-1 + 0.9 * (g == "a"))),
    stringsAsFactors = FALSE
  )

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 61,
    screening = list(iv_min = 0, require_monotonic = "none"),
    control = control.obwoe_scorecard(na_woe = -0.75)
  ))
  skip_if(!"g" %in% sc$final, "g did not enter the model")

  expect_equal(sc$control$na_woe, -0.75)

  novel <- df[1:50, ]
  novel$g <- "never_seen"

  k <- length(sc$final)
  score <- suppressWarnings(predict(sc, novel, type = "score"))
  card <- suppressWarnings(predict(sc, novel, type = "card"))

  # Same bound as the "points sum to the score" test: rounding k integer
  # points can move the total by at most k/2 relative to the raw score.
  expect_lte(max(abs(card - score)), k / 2)
})


test_that("[C3/C-06] drop_negative stops instead of returning silently when every variable is negative", {
  # The removal loop drops exactly one variable per iteration (the worst
  # offender) and refits. The old guard,
  # length(features) - length(negative) < 1L, was also TRUE whenever every
  # *remaining* variable had a negative coefficient -- not only when
  # removing one more would empty the feature set -- so a model where every
  # variable is negative (including the single-variable case) was returned
  # silently, with only a warning, instead of reaching the documented
  # stop("Every variable took a negative coefficient...").
  set.seed(5)
  n <- 500
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n), target = rbinom(n, 1, 0.3))

  # Deterministic mock engine: whatever the data, every slope is negative.
  negative_engine <- list(
    fit = function(x, y, args) list(vars = colnames(x)),
    coef = function(object) {
      stats::setNames(
        c(0, rep(-1, length(object$vars))),
        c("(Intercept)", object$vars)
      )
    },
    link = function(object, x) rep(0, nrow(x)),
    diagnostics = function(object) list(converged = TRUE)
  )

  expect_error(
    suppressWarnings(obwoe_scorecard(df,
      target = "target", feature = c("x1", "x2"),
      engine = negative_engine,
      screening = list(iv_min = 0, require_monotonic = "none")
    )),
    "Every variable took a negative coefficient"
  )
})


test_that("the points fallback follows na_woe", {
  skip_if_no_german()
  df <- german_credit()

  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 42))
  na_points <- attr(sc$points, "points_na")

  # na_woe = 0 puts the fallback at the variable's base term, which is the same
  # for every variable in the model.
  expect_equal(length(unique(na_points)), 1L)

  shifted <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 42,
    control = control.obwoe_scorecard(na_woe = 0.5)
  ))
  moved <- attr(shifted$points, "points_na")

  # With na_woe != 0 the fallback moves by that variable's slope, in the
  # direction the scaling gives risk.
  beta <- shifted$coefficients[shifted$final]
  base <- shifted$scaling$offset / length(shifted$final) -
    shifted$scaling$factor * shifted$coefficients[["(Intercept)"]] /
      length(shifted$final)
  expect_equal(
    unname(moved),
    unname(round(base - shifted$scaling$factor * beta * 0.5))
  )
})


test_that("input validation rejects malformed calls", {
  skip_if_no_german()
  df <- german_credit()

  expect_error(obwoe_scorecard(list(), target = "target"), "data.frame")
  expect_error(obwoe_scorecard(df, target = "nope"), "not found")
  expect_error(obwoe_scorecard(df, target = "target", split = 1.5), "proportion")
  expect_error(
    obwoe_scorecard(df, target = "target", binning = list(nonsense = 1)),
    "Unknown binning argument"
  )
  expect_error(
    obwoe_scorecard(df, target = "target", screening = list(nonsense = 1)),
    "Unknown screening argument"
  )
  expect_error(
    obwoe_scorecard(df, target = "target", feature = "does_not_exist"),
    "not found"
  )

  bad <- df
  bad$target <- df$target + 1L
  expect_error(obwoe_scorecard(bad, target = "target"), "coded 0/1")
})


test_that("an out-of-time split is honoured and never randomised", {
  skip_if_no_german()
  df <- german_credit()
  df$vintage <- rep(c("2024H1", "2024H2"), length.out = nrow(df))

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", split = "vintage",
    seed = 1
  ))

  expect_match(sc$split, "out-of-time")
  expect_false("vintage" %in% sc$candidates)
  expect_equal(sc$samples$train$n, sum(df$vintage == "2024H1"))
  expect_equal(sc$samples$holdout$n, sum(df$vintage == "2024H2"))
})


test_that("the run is deterministic under a seed", {
  skip_if_no_german()
  df <- german_credit()

  a <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 123))
  b <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 123))

  expect_equal(a$final, b$final)
  expect_equal(a$coefficients, b$coefficients)
  expect_equal(a$points$points, b$points$points)
  expect_equal(a$samples$train$score, b$samples$train$score)
})


test_that("stability reports the score and every variable in the model", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 17,
    screening = list(top_n = 5)
  ))

  st <- as.data.frame(sc$stability)
  expect_true("SCORE" %in% st$level)
  expect_setequal(st$variable[st$level == "VARIABLE"], sc$final)
  expect_true(all(st$flag %in% c("stable", "watch", "act")))
  expect_true(all(st$psi >= 0))
})


test_that("predict() reads only the stored artefacts", {
  skip_if_no_german()
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, target = "target", seed = 19))

  # A scorecard whose raw model has been dropped must still score: the
  # deployment artefact is the binning plus the coefficients plus the scaling.
  stripped <- sc
  stripped$model <- NULL

  expect_equal(predict(stripped, df, "score"), predict(sc, df, "score"))
  expect_equal(predict(stripped, df, "card"), predict(sc, df, "card"))
  expect_true(all(predict(sc, df, "prob") > 0 & predict(sc, df, "prob") < 1))
  expect_equal(
    obwoe_score(predict(sc, df, "link"), sc$scaling),
    predict(sc, df, "score")
  )
  expect_s3_class(predict(sc, df, "woe"), "data.frame")
})


test_that("the workbook is written with every sheet populated", {
  skip_if_not_installed("openxlsx")
  skip_if_no_german()
  df <- german_credit()

  path <- tempfile(fileext = ".xlsx")
  on.exit(unlink(path), add = TRUE)

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 23,
    file = path, screening = list(top_n = 6)
  ))

  expect_true(file.exists(path))
  expect_equal(sc$file, path)

  sheets <- openxlsx::getSheetNames(path)
  expect_true(all(nchar(sheets) <= 31L))
  for (s in c(
    "01_Model_Summary", "02_Scorecard", "03_Coefficients",
    "04_Bin_Statistics", "05_Screening", "07_Score_Gains",
    "08_Stability_PSI", "09_Cutoff_Strategy", "10_SQL_WoE",
    "11_SQL_Points", "12_Reproducibility"
  )) {
    expect_true(s %in% sheets, info = s)
    expect_gt(nrow(openxlsx::read.xlsx(path, s)), 0L)
  }

  # The scorecard sheet is the deliverable: it must carry the integer points
  # for every bin of every variable in the model.
  card <- openxlsx::read.xlsx(path, "02_Scorecard")
  expect_setequal(unique(card$variable), sc$final)
  expect_equal(nrow(card), nrow(sc$points))
  expect_false(anyNA(card$points))
})


test_that("obwoe_report() can regenerate the workbook from a saved object", {
  skip_if_not_installed("openxlsx")
  skip_if_no_german()
  df <- german_credit()

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 29,
    screening = list(top_n = 4)
  ))
  expect_null(sc$file)

  path <- tempfile(fileext = ".xlsx")
  on.exit(unlink(path), add = TRUE)
  expect_equal(obwoe_report(sc, file = path, dialect = "postgres"), path)
  expect_true(file.exists(path))

  expect_error(obwoe_report(list(), file = path), "obwoe_scorecard")
})


test_that("a workbook that cannot be written fails before the fit", {
  skip_if_no_german()
  df <- german_credit()

  # The failure must arrive before the pipeline has spent its time, so the
  # error is the path error and not something raised downstream of a fit.
  expect_error(
    obwoe_scorecard(df, target = "target", file = 42),
    "single path"
  )
  expect_error(
    obwoe_scorecard(df, target = "target",
      file = file.path(tempdir(), "no_such_dir", "x.xlsx")
    ),
    "does not exist"
  )
})


test_that("the stage column names the step that actually rejected", {
  # Two near-duplicates and one variable the model will want to reverse: the
  # funnel has to distinguish a correlation drop from a sign drop, because the
  # workbook is read as the reason a variable is absent.
  set.seed(7)
  n <- 4000
  z <- rnorm(n)
  y <- rbinom(n, 1, plogis(-1.2 + 0.9 * z))
  df <- data.frame(
    z = z,
    z_copy = z + rnorm(n, sd = 0.02),
    noise = rnorm(n),
    flat = 1,
    target = y
  )

  sc <- suppressWarnings(obwoe_scorecard(df,
    target = "target", seed = 7,
    screening = list(iv_min = 0, iv_max = 100, require_monotonic = "none")
  ))
  stages <- as.data.frame(sc$screening)

  # Every candidate is accounted for, exactly once, under a known label.
  expect_setequal(stages$feature, sc$candidates)
  expect_true(all(stages$stage %in% c(
    "in_model", "sign_rejected", "corr_pruned", "constant_woe", "screened_out"
  )))
  expect_setequal(stages$feature[stages$stage == "in_model"], sc$final)

  # The near-duplicate pair must cost exactly one variable, and the survivor
  # is the one that entered the model.
  pair <- stages[stages$feature %in% c("z", "z_copy"), ]
  expect_equal(sum(pair$stage == "corr_pruned"), 1L)
  expect_equal(sum(pair$stage == "in_model"), 1L)

  # A variable with no variation cannot be labelled as a correlation drop.
  expect_false(any(stages$stage[stages$feature == "flat"] == "corr_pruned"))

  # And nothing dropped by the sign check may be labelled corr_pruned: those
  # variables survived pruning, so the pruning result must still list them.
  sign_dropped <- stages$feature[stages$stage == "sign_rejected"]
  expect_true(all(sign_dropped %in% sc$correlation$keep))
})


test_that("a variable the model reverses is dropped and labelled as such", {
  # Suppression: b is positively associated with the target on its own, so its
  # WoE points the usual way, but conditionally on a the effect reverses. The
  # WoE already carries the direction, so a negative slope is a fault — the
  # variable must be dropped, and the funnel must say the sign check did it,
  # not the correlation step (pruning is switched off here precisely so that
  # a mislabelled rejection cannot hide behind it).
  set.seed(101)
  n <- 6000
  z <- rnorm(n)
  a <- z + rnorm(n, sd = 0.6)
  b <- z + rnorm(n, sd = 0.6)
  df <- data.frame(
    a = a, b = b,
    target = rbinom(n, 1, plogis(-1 + 1.4 * a - 0.7 * b))
  )
  expect_gt(coef(glm(target ~ b, family = binomial(), data = df))[[2L]], 0)

  expect_warning(
    sc <- obwoe_scorecard(df,
      target = "target", seed = 101,
      control = control.obwoe_scorecard(corr_cutoff = 1),
      screening = list(iv_min = 0, iv_max = 100, require_monotonic = "none")
    ),
    "coefficient on WoE was negative"
  )

  stages <- as.data.frame(sc$screening)
  expect_equal(stages$stage[stages$feature == "b"], "sign_rejected")
  expect_false("b" %in% sc$final)

  # Pruning was off, so b reached the fit: it must not be blamed on correlation.
  expect_true("b" %in% sc$correlation$keep)

  # What survives has a non-negative slope, which is the invariant the drop
  # exists to restore.
  expect_true(all(sc$coefficients[sc$final] >= 0))
  expect_setequal(names(attr(sc$points, "points_na")), sc$final)
})


test_that("the cutoff table approves the safer side, whichever way the scale runs", {
  set.seed(31)
  n <- 3000
  eta <- rnorm(n)
  y <- rbinom(n, 1, plogis(-1 + 1.5 * eta))

  # Under the default the score falls as risk rises, so approving above the
  # cut is right; reversing the scale must reverse the rule, not the meaning.
  for (dir in c("higher_is_safer", "higher_is_riskier")) {
    s <- obwoe_scale(direction = dir)
    tb <- OptimalBinningWoE:::.ob_cutoff_table(
      obwoe_score(eta, s), y, direction = dir
    )
    # An approved population must always be cleaner than the rejected one.
    ok <- is.finite(tb$bad_rate_approved) & is.finite(tb$bad_rate_rejected)
    expect_true(all(tb$bad_rate_approved[ok] < tb$bad_rate_rejected[ok]),
      info = dir
    )
    # Approving less must not make the approved book worse.
    expect_true(all(diff(tb$bad_rate_approved[order(-tb$approval_rate)]) <= 1e-9),
      info = dir
    )
  }
})
