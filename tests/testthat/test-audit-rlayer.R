# Regression and invariant tests for the R-level interface: obwoe(),
# obwoe_apply(), step_obwoe(), obwoe_select(), obwoe_sql(), obwoe_psi() and the
# scorecard pipeline.

# A per-row reference for the categorical lookup of obwoe_apply(): every
# category of every bin is a key, a key listed twice resolves to the LAST bin,
# NA goes to the first bin holding a missing-value token, else to na_woe.
ref_apply_cat <- function(x, bins, woe, sep = "%;%", na_woe = 0) {
  parts <- OptimalBinningWoE:::.ob_split_categories(bins, sep)
  na_bin <- which(vapply(parts, function(p) any(p %in% c("NA", "Missing", "")), TRUE))
  out_bin <- rep(NA_character_, length(x))
  out_woe <- rep(na_woe, length(x))
  for (r in seq_along(x)) {
    if (is.na(x[r])) {
      if (length(na_bin)) {
        out_bin[r] <- bins[na_bin[1]]
        out_woe[r] <- woe[na_bin[1]]
      }
      next
    }
    for (i in seq_along(parts)) {
      if (x[r] %in% parts[[i]]) {
        out_bin[r] <- bins[i]
        out_woe[r] <- woe[i]
      }
    }
  }
  list(bin = out_bin, woe = as.numeric(out_woe))
}

cat_data <- function(n = 1500, seed = 1, lev = c(letters[1:8], "NA", " pad ")) {
  set.seed(seed)
  v <- sample(lev, n, TRUE)
  v[sample(n, 40)] <- NA
  code <- match(v, lev)
  code[is.na(code)] <- 0L
  y <- rbinom(n, 1, plogis(-1 + 0.35 * (code %% 4)))
  data.frame(v = v, target = y, stringsAsFactors = FALSE)
}

test_that(".ob_split_categories inverts the join, empty pieces included", {
  sp <- OptimalBinningWoE:::.ob_split_categories
  expect_identical(
    sp(c("a", "a%;%b", "", "a%;%", "%;%a", "a%;%%;%b", NA), "%;%"),
    list("a", c("a", "b"), "", c("a", ""), c("", "a"), c("a", "", "b"), NA_character_)
  )
  expect_identical(sp(c("x|y", "z|"), "|"), list(c("x", "y"), c("z", "")))
})

test_that("obwoe_apply categorical lookup matches the per-row reference", {
  df <- cat_data()
  fit <- obwoe(df, "target", algorithm = "cm", max_bins = 4)
  expect_false(fit$summary$error)
  nd <- df[1:300, ]
  nd$v[1:3] <- c("UNSEEN", NA, "NA")
  out <- obwoe_apply(nd, fit, na_woe = -7)
  ref <- ref_apply_cat(nd$v, fit$results$v$bin, fit$results$v$woe, na_woe = -7)
  expect_identical(out$v_bin, ref$bin)
  expect_identical(out$v_woe, ref$woe)
  expect_identical(out$v_woe[1], -7)
  expect_true(is.character(out$v_bin) && is.double(out$v_woe))
})

test_that("a category listed in two bins resolves to the last bin", {
  df <- cat_data(seed = 2)
  fit <- obwoe(df, "target", algorithm = "cm", max_bins = 3)
  r <- fit$results$v
  k <- length(r$bin)
  expect_gte(k, 2L)
  dup <- strsplit(r$bin[1], "%;%", fixed = TRUE)[[1]][1]
  fit$results$v$bin[k] <- paste0(r$bin[k], "%;%", dup)
  out <- obwoe_apply(data.frame(v = dup), fit)
  expect_identical(out$v_bin, fit$results$v$bin[k])
  expect_identical(out$v_woe, as.numeric(r$woe[k]))
})

test_that("the empty-string category is scored with its fitted bin", {
  # Before the fix, a bin label equal to "" (or ending in the separator) lost
  # its empty category when split, so "" was scored as na_woe by
  # obwoe_apply(), fell to ELSE in obwoe_sql() and to na_woe in bake().
  set.seed(3)
  n <- 2000
  v <- sample(c("", "a", "b", "c", "d"), n, replace = TRUE)
  y <- rbinom(n, 1, ifelse(v == "", 0.6, ifelse(v == "a", 0.1, 0.3)))
  df <- data.frame(v = v, target = y, stringsAsFactors = FALSE)
  fit <- obwoe(df, "target", algorithm = "cm")
  expect_false(fit$summary$error)
  bins <- fit$results$v$bin
  has_empty <- vapply(
    OptimalBinningWoE:::.ob_split_categories(bins, "%;%"),
    function(p) "" %in% p, TRUE
  )
  expect_true(any(has_empty))

  out <- obwoe_apply(df, fit, na_woe = -99)
  expect_false(anyNA(out$v_bin))
  expect_false(any(out$v_woe == -99))
  e <- which(has_empty)[1]
  expect_true(all(out$v_bin[df$v == ""] == bins[e]))
  expect_true(all(out$v_woe[df$v == ""] == fit$results$v$woe[e]))

  sql <- as.character(obwoe_sql(fit, style = "case"))
  expect_match(sql, "v = ''|v IN \\(.*''", all = FALSE)

  sel <- as.data.frame(obwoe_select(fit, detail = "full"))
  expect_identical(
    sel$n_categories[sel$bin == bins[e]],
    length(OptimalBinningWoE:::.ob_split_categories(bins[e], "%;%")[[1]])
  )
})

test_that("obwoe_apply handles a data frame with no rows", {
  set.seed(4)
  df <- data.frame(x = rnorm(400), v = sample(c("a", "b", "c"), 400, TRUE))
  df$target <- rbinom(400, 1, plogis(df$x))
  fit <- obwoe(df, "target", algorithm = "cm")
  # force a single-bin numerical result (no cut points)
  fit$results$x$cutpoints <- numeric(0)
  fit$results$x$bin <- fit$results$x$bin[1]
  fit$results$x$woe <- fit$results$x$woe[1]
  out <- expect_no_error(obwoe_apply(df[0, ], fit))
  expect_identical(nrow(out), 0L)
  expect_identical(out$x_bin, character(0))
  expect_identical(out$x_woe, numeric(0))
  expect_identical(out$v_bin, character(0))
  expect_identical(out$v_woe, numeric(0))
  one <- obwoe_apply(df[1:3, ], fit)
  expect_identical(one$x_woe, rep(as.numeric(fit$results$x$woe[1]), 3))
})

test_that("obwoe() summary has one typed row per feature", {
  set.seed(5)
  n <- 600
  df <- data.frame(
    a = rnorm(n), b = sample(1:5, n, TRUE),
    c = sample(c("x", "y", "z"), n, TRUE), d = factor(sample(c("u", "v"), n, TRUE))
  )
  df$target <- rbinom(n, 1, plogis(df$a))
  fit <- expect_no_warning(obwoe(df, "target"))
  s <- fit$summary
  expect_identical(s$feature, c("a", "b", "c", "d"))
  expect_identical(
    vapply(s, class, ""),
    c(feature = "character", type = "character", algorithm = "character",
      n_bins = "integer", total_iv = "numeric", converged = "logical",
      iterations = "integer", error = "logical")
  )
  expect_identical(.row_names_info(s), -4L)
  # errors are recorded as rows, not dropped
  fit2 <- obwoe(df, "target", algorithm = "mdlp")
  expect_identical(fit2$summary$error, c(FALSE, FALSE, TRUE, TRUE))
  expect_identical(nrow(fit2$summary), 4L)
})

test_that("step_obwoe honours a custom bin separator at bake time", {
  skip_if_not_installed("recipes")
  set.seed(6)
  n <- 3000
  lev <- paste0("k", 1:12)
  v <- sample(lev, n, TRUE)
  y <- rbinom(n, 1, plogis(-1 + 0.25 * (match(v, lev) %% 5)))
  df <- data.frame(v = v, target = factor(y), stringsAsFactors = FALSE)
  rec <- recipes::recipe(target ~ v, data = df)
  rec <- step_obwoe(rec, v,
    outcome = "target", algorithm = "cm", max_bins = 3,
    control = list(bin_separator = "|"), na_woe = -99
  )
  p <- recipes::prep(rec, training = df)
  br <- p$steps[[1]]$binning_results$v
  expect_true(any(grepl("|", br$bin, fixed = TRUE)))
  expect_false(any(grepl("|", br$cat_map_keys, fixed = TRUE)))
  baked <- recipes::bake(p, new_data = df)
  expect_false(any(baked$v == -99))
  # agrees with obwoe_apply() on the same binning
  fit <- obwoe(data.frame(v = v, target = y), "target",
    algorithm = "cm", max_bins = 3, control = control.obwoe(bin_separator = "|")
  )
  ap <- obwoe_apply(df, fit)
  expect_equal(baked$v, ap$v_woe)
  # the SQL of the prepped recipe and of the step split on the same separator
  for (obj in list(p, p$steps[[1]])) {
    sql <- as.character(obwoe_sql(obj, style = "case"))
    expect_match(sql, "IN \\(", all = FALSE)
    expect_false(grepl("|", sql, fixed = TRUE))
  }
})

test_that("obwoe_sql and obwoe_select default to the model's separator", {
  set.seed(7)
  n <- 3000
  lev <- paste0("k", 1:12)
  v <- sample(lev, n, TRUE)
  y <- rbinom(n, 1, plogis(-1 + 0.25 * (match(v, lev) %% 5)))
  fit <- obwoe(data.frame(v = v, target = y), "target",
    algorithm = "cm", max_bins = 3, control = control.obwoe(bin_separator = "|")
  )
  expect_true(any(grepl("|", fit$results$v$bin, fixed = TRUE)))
  sql <- as.character(obwoe_sql(fit, style = "case"))
  expect_match(sql, "IN \\(", all = FALSE)
  expect_false(grepl("|", sql, fixed = TRUE))
  sel <- as.data.frame(obwoe_select(fit, detail = "full"))
  expect_identical(sum(sel$n_categories), 12L)
  # an explicit separator still wins
  sel2 <- as.data.frame(obwoe_select(fit, detail = "full", bin_separator = "%;%"))
  expect_identical(sum(sel2$n_categories), length(fit$results$v$bin))
})

test_that("obwoe_select survives an infinite WoE", {
  set.seed(8)
  df <- data.frame(x = rnorm(800))
  df$target <- rbinom(800, 1, plogis(df$x))
  fit <- obwoe(df, "target")
  fit$results$x$woe[1] <- -Inf
  sel <- expect_no_error(obwoe_select(fit))
  expect_true(is.logical(sel$woe_monotonic))
  fit$results$x$woe[length(fit$results$x$woe)] <- Inf
  expect_no_error(obwoe_select(fit, detail = "full"))
})

test_that("obwoe_psi band counts match the table() reference", {
  ref_psi <- function(b, c_, lev) {
    sh <- function(v) {
      tb <- table(factor(as.character(v), levels = lev))
      as.numeric(tb) / sum(as.numeric(tb))
    }
    p <- sh(b)
    q <- sh(c_)
    cb <- (p - q) * log(p / q)
    cb[p == 0 & q == 0] <- 0
    cb
  }
  set.seed(9)
  b <- rnorm(3000)
  c_ <- c(rnorm(2000, 0.3), NA)
  r <- obwoe_psi(b, c_, n_groups = 7)
  br <- unique(c(-Inf, stats::quantile(b, seq(0, 1, length.out = 8))[2:7], Inf))
  lev <- levels(cut(b, br, include.lowest = TRUE))
  expect_identical(r$table$psi_band, ref_psi(
    cut(b, br, include.lowest = TRUE), cut(c_, br, include.lowest = TRUE), lev
  ))
  cb <- sample(c("a", "b", "c", NA), 500, TRUE)
  cc <- sample(c("a", "b", "d"), 300, TRUE)
  r2 <- obwoe_psi(cb, cc)
  expect_identical(r2$table$psi_band, ref_psi(cb, cc, sort(unique(c(cb, cc)))))
  r3 <- obwoe_psi(cb, cc, levels = c("a", "b", "c", "d", "e"))
  expect_identical(r3$table$psi_band, ref_psi(cb, cc, c("a", "b", "c", "d", "e")))
})

test_that("scoring does not warn about variables the scorecard does not use", {
  df <- german_credit()
  names(df)[names(df) == "target"] <- "default"
  sc <- suppressWarnings(obwoe_scorecard(df, "default", seed = 1))
  unused <- setdiff(sc$candidates, sc$final)
  expect_gt(length(unused), 0L)
  nd <- df[1:60, setdiff(names(df), unused), drop = FALSE]
  for (ty in c("score", "card", "link", "prob", "woe")) {
    expect_no_warning(predict(sc, nd, type = ty))
  }
  expect_identical(predict(sc, nd), predict(sc, df[1:60, ]))
  expect_identical(predict(sc, nd, type = "card"), predict(sc, df[1:60, ], type = "card"))
})

test_that("a clean binning pipeline raises no warnings", {
  df <- german_credit()
  fit <- expect_no_warning(obwoe(df, "target"))
  expect_no_warning(obwoe_apply(df, fit))
  expect_no_warning(obwoe_select(fit, detail = "full"))
  expect_no_warning(obwoe_sql(fit))
  expect_no_warning(obwoe_gains(fit))
  expect_no_warning(summary(fit))
})

test_that("apply and bake agree on numerical bins, including cut points", {
  skip_if_not_installed("recipes")
  set.seed(10)
  n <- 1000
  df <- data.frame(x = round(rnorm(n), 1))
  df$target <- rbinom(n, 1, plogis(df$x))
  fit <- obwoe(df, "target", algorithm = "mob")
  cp <- fit$results$x$cutpoints
  nd <- data.frame(x = c(cp, cp + 1e-9, -Inf, Inf, NA))
  ap <- obwoe_apply(nd, fit, na_woe = -5)
  k <- findInterval(nd$x, cp, left.open = TRUE) + 1L
  expect_identical(ap$x_bin[1:(2 * length(cp) + 2)], fit$results$x$bin[k[1:(2 * length(cp) + 2)]])
  expect_identical(ap$x_woe[nrow(nd)], -5)
  dff <- df
  dff$target <- factor(dff$target)
  p <- recipes::prep(step_obwoe(recipes::recipe(target ~ x, data = dff), x,
    outcome = "target", algorithm = "mob", na_woe = -5, max_bins = 7
  ), training = dff)
  expect_equal(recipes::bake(p, new_data = nd)$x, ap$x_woe)
})

# ---------------------------------------------------------------------------
# Branch coverage of the interface layer: argument checks, degenerate inputs,
# print/summary/plot methods and the less common pipeline exits.
# ---------------------------------------------------------------------------

small_fit_data <- function(n = 500, seed = 21) {
  set.seed(seed)
  d <- data.frame(
    a = rnorm(n), b = sample(1:6, n, TRUE),
    c = sample(c("x", "y", "z", "w"), n, TRUE),
    stringsAsFactors = FALSE
  )
  d$target <- rbinom(n, 1, plogis(1.2 * d$a + 0.3 * (d$c == "x")))
  d
}

test_that("obwoe() rejects malformed arguments with clear messages", {
  d <- small_fit_data()
  expect_error(obwoe(d, target = 1), "single character string")
  expect_error(obwoe(d, "target", feature = 1), "NULL or a character vector")
  expect_error(obwoe(d, "target", feature = "nope"), "not found: nope")
  expect_error(obwoe(d["target"], "target"), "No feature columns")
  expect_error(obwoe(d, "target", control = 3), "control.obwoe")
  expect_error(obwoe(d, "target", min_bins = 1), "at least 2")
  expect_error(obwoe(d, "target", min_bins = 5, max_bins = 3), ">= 'min_bins'")
  d1 <- d
  d1$target[1] <- NA
  expect_error(obwoe(d1, "target"), "missing values")
  d2 <- d
  d2$target <- 0L
  expect_error(obwoe(d2, "target"), "binary \\(0/1\\) or multinomial")
  expect_error(control.obwoe(bin_cutoff = 1), "between 0 and 1")
  # a plain list is accepted as control
  f <- obwoe(d, "target", feature = "a", control = list(bin_cutoff = 0.1))
  expect_s3_class(f$control, "obwoe_control")
  expect_identical(f$control$bin_cutoff, 0.1)
})

test_that("obwoe() records dispatch failures per feature", {
  d <- small_fit_data()
  f <- obwoe(d, "target", algorithm = "nosuch")
  expect_true(all(f$summary$error))
  expect_match(f$results$a$error, "Unknown algorithm")
  f <- obwoe(d, "target", algorithm = "gmb")
  expect_match(f$results$a$error, "does not support numerical")
  f <- obwoe(d, "target", feature = "a", algorithm = "fast_mdlp")
  expect_false(f$summary$error)
  d$target[1:30] <- 2L
  f <- obwoe(d, "target", feature = "a", algorithm = "mob")
  expect_match(f$results$a$error, "multinomial target")
  local_mocked_bindings(.get_algorithm_registry = function() {
    list(zzz = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE))
  })
  f <- obwoe(small_fit_data(), "target", feature = "a", algorithm = "zzz")
  expect_match(f$results$a$error, "Function 'ob_numerical_zzz' not found")
})

test_that("internal helpers cover their degenerate inputs", {
  iv <- OptimalBinningWoE:::.ob_iv_from_counts
  expect_equal(iv(c(10, 20), c(30, 5)), {
    p <- (c(10, 20) + 0.5) / 31
    q <- (c(30, 5) + 0.5) / 36
    sum((p - q) * log(p / q))
  })
  expect_identical(iv(c(1, NA), c(2, 3)), NA_real_)
  expect_identical(iv(c(0, 0), c(2, 3)), NA_real_)
  expect_identical(OptimalBinningWoE:::.ob_bin_separator(list()), "%;%")
  expect_identical(OptimalBinningWoE:::.ob_split_categories("a%;%", ""), strsplit("a%;%", ""))
  alg <- obwoe_algorithms()
  expect_true(all(c("algorithm", "numerical", "categorical", "multinomial") %in% names(alg)))
  expect_identical(nrow(alg), length(OptimalBinningWoE:::.get_algorithm_registry()))
})

test_that("print, summary and plot methods cover every branch", {
  pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  set.seed(22)
  n <- 800
  d <- data.frame(matrix(rnorm(n * 7), n))
  d$strong <- rnorm(n)
  d$cat <- sample(c("p", "q"), n, TRUE)
  d$target <- rbinom(n, 1, plogis(2.5 * d$strong))
  f <- obwoe(d, "target", algorithm = "mdlp") # 'cat' errors
  out <- capture.output(print(f))
  expect_true(any(grepl("errors", out)))
  expect_true(any(grepl("and .* more", out)))
  s <- summary(f, sort_by = "feature", decreasing = TRUE)
  expect_identical(s$feature_summary$feature, sort(s$feature_summary$feature))
  expect_true("Error" %in% as.character(s$feature_summary$iv_class))
  expect_true(any(grepl("Median IV", capture.output(print(s)))))
  s2 <- summary(f, sort_by = "n_bins")
  expect_s3_class(s2, "summary.obwoe")
  # every Siddiqi class, incl. "Strong"
  cls <- OptimalBinningWoE:::summary.obwoe(structure(list(
    summary = data.frame(
      feature = letters[1:6], type = "numerical", algorithm = "x",
      n_bins = 2L, total_iv = c(0.01, 0.05, 0.2, 0.4, 0.9, NA),
      error = c(rep(FALSE, 5), TRUE), stringsAsFactors = FALSE
    ), target = "t", target_type = "binary"
  ), class = "obwoe"))
  expect_identical(
    as.character(cls$feature_summary$iv_class),
    c("Suspicious", "Strong", "Medium", "Weak", "Unpredictive", "Error")
  )
  # no IV measured anywhere / nothing successful
  noiv <- structure(list(
    summary = data.frame(
      feature = c("a", "b"), type = "numerical", algorithm = "x",
      n_bins = NA_integer_, total_iv = NA_real_, error = c(FALSE, TRUE),
      stringsAsFactors = FALSE
    ), target = "t", target_type = "binary"
  ), class = "obwoe")
  s3 <- summary(noiv)
  expect_true(is.na(s3$aggregate$total_iv_sum))
  expect_true(any(grepl("Total IV: NA", capture.output(print(s3)))))
  expect_true(any(grepl("Mean Bins: NA", capture.output(print(s3)))))
  noiv$summary$error <- TRUE
  expect_identical(summary(noiv)$aggregate$n_successful, 0)
  expect_message(plot(noiv), "No successful")
  noiv$summary$error <- c(FALSE, TRUE)
  expect_message(plot(noiv), "No finite Information Value")

  expect_null(plot(f, type = "iv", top_n = 3))
  expect_null(plot(f, type = "woe"))
  expect_message(plot(f, type = "woe", feature = c(paste0("X", 1:7), "cat")), "first 6")
  expect_null(plot(f, type = "woe", feature = c("strong", "cat")))
  expect_null(plot(f, type = "woe", feature = "strong"))
  expect_null(plot(f, type = "bins"))
  expect_message(plot(f, type = "bins", feature = c("strong", "X1")), "first feature only")
  expect_message(plot(f, type = "bins", feature = "cat"), "no valid results")
})

test_that("obwoe_apply() refuses what it cannot score and flags odd cut points", {
  d <- small_fit_data()
  f <- obwoe(d, "target")
  bad <- f
  bad$summary$error <- TRUE
  expect_error(obwoe_apply(d, bad), "No successful binning")
  expect_error(obwoe_apply(as.list(d), f), "must be a data.frame")
  expect_error(obwoe_apply(d, unclass(f)), "class 'obwoe'")
  expect_error(obwoe_apply(d["target"], f), "None of the binned features")
  expect_warning(out <- obwoe_apply(d[c("a", "target")], f), "Features not in data")
  expect_identical(names(out), c("target", "a", "a_bin", "a_woe"))
  mc <- f
  mc$target_type <- "multinomial"
  expect_error(obwoe_apply(d, mc), "Multiclass")
  # duplicated cut points that no longer match the bin count: fallback mapping
  dup <- f
  r <- dup$results$a
  k <- length(r$bin)
  dup$results$a$cutpoints <- c(r$cutpoints[1], r$cutpoints)
  dup$results$a$bin <- c(r$bin, "extra")
  dup$results$a$woe <- c(r$woe, 0)
  expect_warning(out <- obwoe_apply(d, dup, na_woe = -1), "Cutpoint deduplication")
  expect_true(all(out$a_woe == mean(dup$results$a$woe)))
  expect_match(out$a_bin[which.min(d$a)], "^\\(-Inf")
  expect_match(out$a_bin[which.max(d$a)], "Inf\\)$")
})

test_that("obwoe_gains() resolves its inputs or explains why it cannot", {
  d <- small_fit_data()
  f <- obwoe(d, "target")
  ap <- obwoe_apply(d, f)
  expect_error(obwoe_gains(list()), "must be an 'obwoe' object or a 'data.frame'")
  expect_error(obwoe_gains(f, feature = "nope"), "no valid binning")
  none <- f
  none$summary$total_iv <- NA_real_
  expect_error(obwoe_gains(none), "Cannot pick a feature")
  expect_error(obwoe_gains(ap), "'target' argument is required")
  expect_error(obwoe_gains(ap, target = 1:3), "length must match")
  g <- obwoe_gains(ap, target = "target")
  expect_identical(g$feature, "a")
  g <- obwoe_gains(ap[c("target", "a_woe")], target = d$target)
  expect_identical(g$feature, "a")
  expect_error(obwoe_gains(ap[c("target", "a")], target = d$target), "No '_bin' or '_woe'")
  g <- obwoe_gains(ap[c("target", "a_woe")], target = "target", feature = "a", use_column = "auto")
  expect_equal(g$table$woe, sort(unique(ap$a_woe)))
  g <- obwoe_gains(ap[c("target", "a")], target = "target", feature = "a", n_groups = 5)
  expect_identical(g$n_bins, 5L)
  expect_error(
    obwoe_gains(ap[c("target", "a")], target = "target", feature = "zz"),
    "Could not automatically locate"
  )
  expect_error(
    obwoe_gains(ap, target = "target", feature = "zz", use_column = "bin"),
    "Grouping column 'zz_bin' not found"
  )
  g <- obwoe_gains(ap[c("target", "a")], target = "target", feature = "a", use_column = "bin")
  expect_equal(g$n_obs, nrow(d))
  expect_warning(
    obwoe_gains(data.frame(t = d$target, s = 1), target = "t", feature = "s",
      use_column = "direct", n_groups = 4
    ),
    "Insufficient unique values"
  )
  g <- obwoe_gains(ap, target = "target", feature = "c", sort_by = "bin")
  expect_identical(g$table$bin, sort(g$table$bin))
  for (sb in c("woe", "event_rate")) {
    expect_s3_class(obwoe_gains(f, feature = "c", sort_by = sb), "obwoe_gains")
  }
  pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  for (ty in c("cumulative", "ks", "lift", "woe_iv")) expect_null(plot(g, type = ty))
  expect_true(any(grepl("KS Statistic", capture.output(print(g)))))
})

test_that("obwoe_gains() orders numeric groups by value, not as text", {
  # Before the fix the groups of a numeric column were sorted as strings
  # ("1", "10", "11", "12", "2", ...), so KS and AUC were accumulated in an
  # order unrelated to the score: KS 25.7% here instead of 50.0%.
  set.seed(1)
  n <- 5000
  x <- sample(1:12, n, TRUE)
  y <- rbinom(n, 1, plogis(-3 + 0.4 * x))
  d <- data.frame(x = x, t = y)
  g <- obwoe_gains(d, target = "t", feature = "x", use_column = "direct")
  expect_identical(g$table$bin, as.character(1:12))
  gf <- obwoe_gains(transform(d, x = factor(x)), target = "t", feature = "x", use_column = "direct")
  expect_equal(g$metrics[c("ks", "gini", "auc", "total_iv")], gf$metrics[c("ks", "gini", "auc", "total_iv")])
  expect_equal(g$metrics$ks, 49.95525, tolerance = 1e-6)
  w <- data.frame(t = y, x_woe = c(-3.1, -2, -1.8, 0.1)[x %% 4 + 1])
  gw <- obwoe_gains(w, target = "t", feature = "x", use_column = "woe")
  expect_identical(gw$table$woe, c(-3.1, -2, -1.8, 0.1))
})

test_that("engine resolution: custom engines and missing packages", {
  expect_error(OptimalBinningWoE:::.ob_engine_get(list(fit = identity)), "missing")
  expect_error(OptimalBinningWoE:::.ob_engine_get("nosuch"), "must be one of")
  cu <- OptimalBinningWoE:::.ob_engine_get(list(fit = 1, link = 2, coef = 3))
  expect_identical(cu$used, "custom")
  local_mocked_bindings(.ob_engine_registry = function() {
    list(glm = list(pkgs = character(0), additive = TRUE), ghost = list(pkgs = "notAPackage.xyz"))
  })
  expect_error(OptimalBinningWoE:::.ob_engine_get("ghost"), "not installed")
  expect_warning(
    e <- OptimalBinningWoE:::.ob_engine_get("ghost", fallback = TRUE),
    "falling back"
  )
  expect_identical(c(e$requested, e$used), c("ghost", "glm"))
})

test_that("step_obwoe() argument checks, degenerate columns and print paths", {
  skip_if_not_installed("recipes")
  d <- small_fit_data()
  d$target <- factor(d$target)
  rec <- recipes::recipe(target ~ ., data = d)
  expect_error(step_obwoe(rec, a, outcome = "target", max_bins = "x"), "max_bins")
  expect_error(step_obwoe(rec, a, outcome = "target", bin_cutoff = "x"), "bin_cutoff")
  st <- step_obwoe(rec, a, outcome = "target")
  expect_error(recipes::prep(st, training = d[setdiff(names(d), "target")]))
  # numeric 0/1 outcome with algorithm = "auto"
  d2 <- small_fit_data()
  p <- recipes::prep(step_obwoe(recipes::recipe(target ~ ., data = d2), a, c,
    outcome = "target", output = "both"
  ), training = d2)
  expect_identical(p$steps[[1]]$algorithm, "jedi")
  b <- recipes::bake(p, new_data = d2)
  expect_identical(
    names(b),
    c("a", "a_woe", "a_bin", "b", "c", "c_woe", "c_bin", "target")
  )
  # a feature whose obwoe() call fails outright is skipped with a warning
  d3 <- d2
  d3$target[1] <- NA
  expect_warning(
    p3 <- recipes::prep(step_obwoe(recipes::recipe(target ~ ., data = d3), a,
      outcome = "target"
    ), training = d3),
    "Failed to bin variable 'a'"
  )
  expect_length(p3$steps[[1]]$binning_results, 0L)
  expect_match(capture.output(print(p3$steps[[1]])), "0 features", all = FALSE)
  # single-bin numerical binning at bake time
  st <- p$steps[[1]]
  st$binning_results$a$cutpoints <- NULL
  st$binning_results$a$bin <- "(-Inf;+Inf)"
  st$binning_results$a$woe <- 0.25
  nd <- d2[1:4, ]
  nd$a[2] <- NA
  bb <- recipes::bake(st, new_data = tibble::as_tibble(nd))
  expect_identical(bb$a_woe, c(0.25, st$na_woe, 0.25, 0.25))
  expect_identical(bb$a_bin, c("(-Inf;+Inf)", NA, "(-Inf;+Inf)", "(-Inf;+Inf)"))
  st$binning_results$a$cutpoints <- NA_real_
  bb <- recipes::bake(st, new_data = tibble::as_tibble(nd))
  expect_identical(bb$a_woe, c(0.25, st$na_woe, 0.25, 0.25))
  # output = "both" on the last column of the frame
  last <- d2[c("target", "a")]
  pl <- recipes::prep(step_obwoe(recipes::recipe(target ~ a, data = last), a,
    outcome = "target", output = "both"
  ), training = last)
  expect_identical(
    names(recipes::bake(pl$steps[[1]], new_data = tibble::as_tibble(last))),
    c("target", "a", "a_woe", "a_bin")
  )
  # print: IV from per-bin iv, IV not reported, long term list
  st$binning_results$a$total_iv <- NA_real_
  expect_match(capture.output(print(st)), "total IV=", all = FALSE)
  st$binning_results <- lapply(st$binning_results, function(r) {
    r$total_iv <- NA_real_
    r$iv <- rep(NA_real_, length(r$bin))
    r
  })
  expect_match(capture.output(print(st)), "not reported", all = FALSE)
  long <- step_obwoe(rec, a, b, c, outcome = "target")$steps[[1]]
  expect_match(capture.output(print(long, width = 5)), "\\.\\.\\.", all = FALSE)
})

test_that("obwoe_select() error rows, strict monotonicity and the base-R path", {
  d <- small_fit_data()
  f <- obwoe(d, "target")
  expect_error(obwoe_select(f, iv_max = NA), "'iv_max'")
  empty <- f
  empty$results <- list()
  expect_error(obwoe_select(empty), "no binning results")
  g <- f
  g$results$a$count_pos <- NULL
  g$results$b$woe <- matrix(0, 2, 2)
  g$results$c$count <- g$results$c$count[-1]
  s <- as.data.frame(obwoe_select(g, monotonicity = "strict"))
  expect_true(all(s$error))
  expect_match(s$error_msg[s$feature == "a"], "lacks binary bin counts")
  expect_match(s$error_msg[s$feature == "b"], "Multinomial")
  expect_match(s$error_msg[s$feature == "c"], "Inconsistent bin vectors")
  full <- obwoe_select(g, detail = "full")
  expect_identical(nrow(full), 3L)
  local_mocked_bindings(obwoe_gains_score = function(...) stop("boom"))
  s <- as.data.frame(obwoe_select(f))
  expect_true(all(grepl("Gains table computation failed", s$error_msg)))
  expect_identical(OptimalBinningWoE:::.ob_monotonicity(c(1, NA))$monotonic, NA)
  expect_null(OptimalBinningWoE:::.ob_rbind(list(NULL)))
  local_mocked_bindings(.ob_has_dt = function() FALSE)
  rb <- OptimalBinningWoE:::.ob_rbind(list(list(x = 1:2), data.frame(x = 3L)))
  expect_identical(rb$x, 1:3)
  expect_false(inherits(OptimalBinningWoE:::.ob_as_table(data.frame(x = 1)), "data.table"))
})

test_that("obwoe_sql() argument checks and rarely taken branches", {
  skip_if_not_installed("recipes")
  d <- small_fit_data()
  f <- obwoe(d, "target")
  expect_error(obwoe_sql(f, view_name = NA_character_), "view_name")
  empty <- f
  empty$results <- list()
  expect_error(obwoe_sql(empty), "no binning results")
  nob <- f
  nob$results$a$bin <- character(0)
  expect_warning(obwoe_sql(nob), "has no bins")
  mc <- f
  mc$results$c$woe <- cbind(mc$results$c$woe, mc$results$c$woe)
  expect_error(obwoe_sql(mc, features = "c", class_index = 3), "between 1 and 2")
  expect_match(obwoe_sql(mc, features = "c", class_index = 2), "CASE")
  one <- f
  one$results$a$cutpoints <- numeric(0)
  one$results$a$bin <- one$results$a$bin[1]
  one$results$a$woe <- one$results$a$woe[1]
  expect_match(obwoe_sql(one, features = "a", style = "case"), "ELSE")
  pc <- capture.output(print(obwoe_sql(f, style = "case")))
  expect_true(any(grepl("^-- a_woe", pc)))
  df2 <- d
  df2$target <- factor(df2$target)
  rec <- recipes::recipe(target ~ ., data = df2)
  expect_error(obwoe_sql(recipes::prep(recipes::step_center(rec, a), training = df2)), "no step_obwoe")
  twice <- step_obwoe(step_obwoe(rec, a, outcome = "target"), a, outcome = "target")
  expect_error(obwoe_sql(recipes::prep(twice, training = df2)), "more than one step")
  st <- recipes::prep(step_obwoe(rec, a, outcome = "target"), training = df2)$steps[[1]]
  expect_match(obwoe_sql(st), "Algorithm\\(s\\): jedi")
  st$algorithm <- NULL
  expect_match(obwoe_sql(st), "Algorithm\\(s\\): unknown")
})

test_that("score helpers: scaling print, pruning inputs and PSI branches", {
  expect_output(print(obwoe_scale()), "points double the odds")
  expect_error(obwoe_prune(data.frame(a = 1), ranking = character(0)), "non-empty")
  expect_error(obwoe_prune(data.frame(a = 1), ranking = "a", cutoff = 2), "cutoff")
  expect_error(obwoe_prune(1:3, ranking = "a"), "data.frame of numeric")
  expect_error(
    obwoe_prune(data.frame(x = "a", y = "b"), ranking = "a"),
    "correlation column"
  )
  pr <- obwoe_prune(data.frame(x = c("a", "a"), y = c("b", "c"), r = c(0.9, 0.1)),
    ranking = c("a", "b", "c")
  )
  expect_identical(pr$keep, c("a", "c"))
  expect_identical(obwoe_prune(data.frame(a = 1:3), ranking = "a")$keep, "a")
  expect_error(obwoe_psi(numeric(0), 1), "non-empty")
  expect_identical(obwoe_psi(rnorm(5), 1:5, n_groups = 1L)$flag, "stable")
  set.seed(23)
  expect_identical(obwoe_psi(rnorm(4000), rnorm(4000, 0.35))$flag, "watch")
  # missing and repeated levels used to abort with unrelated errors
  p <- obwoe_psi(c("a", "b", NA), c("a", "b"), levels = c("a", "b", NA, "a"))
  expect_identical(p$table$band, c("a", "b"))
  expect_identical(p$psi, 0)
  expect_error(obwoe_psi("a", "b", levels = NA), "at least one non-missing")
})

test_that("obwoe_scorecard() argument checks and split variants", {
  df <- german_credit()
  names(df)[names(df) == "target"] <- "default"
  expect_error(obwoe_scorecard(list(), "default"), "must be a data.frame")
  expect_error(obwoe_scorecard(df, c("a", "b")), "single column name")
  expect_error(obwoe_scorecard(df, "default", control = 1), "control.obwoe_scorecard")
  expect_error(obwoe_scorecard(df, "default", binning = 1), "must be lists")
  expect_error(obwoe_scorecard(df, "default", file = "/nonexistent_dir_xyz/a.xlsx"), "does not exist")
  expect_error(obwoe_scorecard(df, "default", split = "nosuchcol_or_bad"), "'split' must be")
  expect_error(obwoe_scorecard(transform(df, s = 1), "default", split = "s"), "single value")
  expect_error(obwoe_scorecard(df, "default", split = c(0, 5000)), "outside the data")
  expect_error(
    obwoe_scorecard(df, "default", split = df$default == 1),
    "only one class"
  )
  expect_error(
    obwoe_scorecard(df, "default", validation = list(df[1:10, ])),
    "named list"
  )
  expect_error(
    obwoe_scorecard(df, "default", validation = list(v = df[1:50, names(df) != "default"])),
    "missing column"
  )
  vna <- df[1:50, ]
  vna$default[1] <- NA
  expect_error(obwoe_scorecard(df, "default", validation = list(v = vna)), "missing value")
  bad <- df
  bad$default <- factor(ifelse(bad$default == 1, "b", ifelse(seq_len(nrow(bad)) %% 2, "g", "h")))
  expect_error(obwoe_scorecard(bad, "default"), "exactly two levels")
  local_mocked_bindings(.ob_has_openxlsx = function() FALSE)
  expect_error(obwoe_scorecard(df, "default", file = tempfile(fileext = ".xlsx")), "openxlsx")
})

test_that("obwoe_scorecard() pipeline exits and diagnostics", {
  df <- german_credit()
  names(df)[names(df) == "target"] <- "default"
  df$default <- as.logical(df$default)
  # logical target, explicit index split, a list control, constant after WoE,
  # a failed candidate and the thin-sample warning all in one run
  df$flat <- 1
  df$broken <- NA_real_
  idx <- seq(1, nrow(df), by = 2)
  w <- character()
  sc <- withCallingHandlers(
    obwoe_scorecard(df, "default",
      split = idx, control = list(corr_cutoff = 0.99, drop_negative = FALSE, max_abs_coef = 0.1),
      screening = list(iv_min = 0.0001, require_monotonic = "none", allow_degenerate = TRUE)
    ),
    warning = function(x) {
      w <<- c(w, conditionMessage(x))
      invokeRestart("muffleWarning")
    }
  )
  expect_identical(sc$event_level, "TRUE")
  expect_identical(sc$split, "explicit row index")
  expect_true(any(grepl("could not be binned", w)))
  expect_true(any(grepl("Coefficient\\(s\\) above", w)))
  out <- capture.output(print(sc))
  expect_true(any(grepl("Funnel", out)))
  # no hold-out, logical split, stability absent
  sc2 <- suppressWarnings(obwoe_scorecard(german_credit(), "target", split = NULL))
  expect_null(sc2$stability)
  expect_true(any(grepl("No hold-out", sc2$warnings)))
  sc3 <- suppressWarnings(obwoe_scorecard(german_credit(), "target",
    split = rep(c(TRUE, FALSE), 500)
  ))
  expect_identical(sc3$split, "explicit logical index")
  # an old object without $control scores with na_woe = 0
  old <- sc3
  old$control <- NULL
  expect_equal(predict(old, german_credit()[1:5, ]), predict(sc3, german_credit()[1:5, ]))
  # non-additive and non-converging engines
  glm_e <- OptimalBinningWoE:::.ob_engine_get("glm")
  nonadd <- list(fit = glm_e$fit, link = glm_e$link, coef = function(object) NULL)
  sc4 <- suppressWarnings(obwoe_scorecard(german_credit(), "target", engine = nonadd, seed = 2))
  expect_null(sc4$points)
  expect_error(predict(sc4, german_credit()), "non-additive")
  nc <- list(fit = glm_e$fit, link = glm_e$link, coef = glm_e$coef,
    diagnostics = function(object) list(converged = FALSE))
  expect_error(obwoe_scorecard(german_credit(), "target", engine = nc), "did not converge")
  expect_error(
    obwoe_scorecard(german_credit(), "target",
      engine = list(fit = glm_e$fit, link = glm_e$link, coef = function(o) {
        cf <- glm_e$coef(o)
        cf[-1] <- -abs(cf[-1])
        cf
      })
    ),
    "Every variable took a negative coefficient"
  )
  # degenerate score metrics
  m <- OptimalBinningWoE:::.ob_score_metrics(1:3, c(1L, 1L, 1L))
  expect_true(is.na(m$auc))
})

test_that("obwoe_report() writes every sheet and checks its inputs", {
  skip_if_not_installed("openxlsx")
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, "target", seed = 3,
    validation = list(recent = df[1:200, ])
  ))
  f <- tempfile(fileext = ".xlsx")
  on.exit(unlink(f), add = TRUE)
  expect_error(obwoe_report(list(), f), "obwoe_scorecard")
  expect_error(obwoe_report(sc, 1), "single path")
  expect_identical(obwoe_report(sc, f, keep_columns = "id"), f)
  expect_true(file.exists(f))
  expect_error(
    obwoe_report(sc, f, control = control.obwoe_scorecard(overwrite = FALSE)),
    "overwrite is FALSE"
  )
  # an object saved before 1.13.1/1.13.6 (no control, no band breaks), from a
  # non-additive engine (no points table)
  old <- sc
  old$control <- NULL
  old$band_breaks <- NULL
  old$points <- NULL
  expect_identical(obwoe_report(old, f), f)
  local_mocked_bindings(.ob_has_openxlsx = function() FALSE)
  expect_error(obwoe_report(sc, f), "openxlsx")
})

test_that("points SQL and cutoff table reject what they cannot represent", {
  df <- german_credit()
  sc <- suppressWarnings(obwoe_scorecard(df, "target", seed = 4))
  sql <- OptimalBinningWoE:::.ob_points_sql(sc, "t", "ansi", keep_columns = "id")
  expect_match(sql, "SELECT\nid,")
  broken <- sc
  num <- sc$final[vapply(sc$final, function(v) identical(sc$binning$results[[v]]$type, "numerical"), TRUE)]
  expect_gt(length(num), 0L)
  for (v in num) broken$binning$results[[v]]$cutpoints <- numeric(0)
  cat_final <- setdiff(sc$final, num)
  broken$final <- num
  expect_error(
    suppressWarnings(OptimalBinningWoE:::.ob_points_sql(broken, "t", "ansi")),
    "No variable produced a valid points SQL"
  )
  ct <- OptimalBinningWoE:::.ob_cutoff_table
  expect_error(ct(c(1, NA), c(0, 1)), "score\\(s\\) are missing")
  expect_error(ct(c(1, 2), c(0, NA)), "target value\\(s\\) are missing")
})

test_that("remaining fallbacks: IV from counts, all-NA cut points, SQL skips", {
  d <- small_fit_data()
  # an algorithm reporting neither total_iv nor a per-bin iv
  local_mocked_bindings(.dispatch_algorithm = function(...) {
    list(bin = c("lo", "hi"), woe = c(-0.5, 0.5), count = c(10L, 12L),
      count_pos = c(2L, 8L), count_neg = c(8L, 4L))
  })
  f <- obwoe(d, "target", feature = "a")
  expect_equal(
    f$summary$total_iv,
    OptimalBinningWoE:::.ob_iv_from_counts(c(2L, 8L), c(8L, 4L))
  )
})

test_that("cut points that are all missing, or inconsistent with the bins", {
  d <- small_fit_data()
  f <- obwoe(d, "target")
  na_cp <- f
  na_cp$results$a$cutpoints <- NA_real_
  expect_warning(out <- obwoe_apply(d, na_cp), "All cutpoints were duplicates")
  expect_true(all(out$a_woe == na_cp$results$a$woe[1]))
  dup <- f
  dup$results$a$cutpoints <- rep(dup$results$a$cutpoints[1], 2)
  expect_warning(s <- obwoe_sql(dup, features = c("a", "c")), "inconsistent with 1 distinct cut")
  expect_false(grepl("a_woe", s))
})

test_that("plot colours cover a missing and a strong IV", {
  pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  x <- structure(list(summary = data.frame(
    feature = c("a", "b", "c"), type = "numerical", algorithm = "x",
    n_bins = 2L, total_iv = c(NA, 0.4, 0.05), error = FALSE,
    stringsAsFactors = FALSE
  )), class = "obwoe")
  expect_null(plot(x, type = "iv"))
})

test_that("scorecard: thin sample warning and constant WoE after transform", {
  df <- german_credit()
  set.seed(24)
  noise <- as.data.frame(matrix(rnorm(nrow(df) * 37), nrow(df)))
  big <- cbind(df, noise)
  sc <- NULL
  w <- character()
  sc <- withCallingHandlers(
    obwoe_scorecard(big, "target", seed = 5, control = list(corr_cutoff = 1),
      binning = list(algorithm = "ewb"),
      screening = list(iv_min = 0.001, require_monotonic = "none", allow_degenerate = TRUE)
    ),
    warning = function(x) {
      w <<- c(w, conditionMessage(x))
      invokeRestart("muffleWarning")
    }
  )
  expect_true(any(grepl("an Information\\s+Value floor is a weak filter", w)))
  real <- OptimalBinningWoE:::.ob_woe_matrix
  local_mocked_bindings(.ob_woe_matrix = function(data, binning, features, na_woe) {
    out <- real(data, binning, features, na_woe)
    out[[1]] <- 0
    out
  })
  sc2 <- suppressWarnings(obwoe_scorecard(df, "target", seed = 5))
  expect_true(any(grepl("constant after the WoE transform", sc2$warnings)))
  expect_true("constant_woe" %in% as.data.frame(sc2$screening)$stage)
  local_mocked_bindings(.ob_woe_matrix = function(data, binning, features, na_woe) {
    out <- real(data, binning, features, na_woe)
    out[] <- 0
    out
  })
  expect_error(suppressWarnings(obwoe_scorecard(df, "target", seed = 5)), "Every screened variable is constant")
})

test_that("step_obwoe() reports an outcome missing from the training data", {
  skip_if_not_installed("recipes")
  d <- small_fit_data()
  d$target <- factor(d$target)
  rec <- step_obwoe(recipes::recipe(target ~ ., data = d), a, outcome = "zzz")
  expect_error(recipes::prep(rec, training = d), "Outcome column 'zzz' not found")
})

test_that("a numerical binner's trailing NA bin is honoured by apply, SQL and select", {
  # ob_numerical_udt() gives missing values a bin of their own. Before the fix
  # obwoe_apply() counted it as a bin/interval mismatch, warned, and scored
  # every row at the mean WoE; obwoe_sql() dropped the feature.
  set.seed(2)
  n <- 2000
  x <- rnorm(n)
  x[sample(n, 100)] <- NA
  y <- rbinom(n, 1, plogis(ifelse(is.na(x), 1, x)))
  d <- data.frame(x = x, target = y)
  f <- obwoe(d, "target", algorithm = "udt")
  r <- f$results$x
  k <- length(r$bin)
  expect_identical(r$bin[k], "NA")
  expect_identical(k, length(r$cutpoints) + 2L)
  out <- expect_no_warning(obwoe_apply(d, f, na_woe = -99))
  expect_true(all(out$x_bin[is.na(x)] == "NA"))
  expect_true(all(out$x_woe[is.na(x)] == r$woe[k]))
  idx <- findInterval(x[!is.na(x)], r$cutpoints, left.open = TRUE) + 1L
  expect_identical(out$x_bin[!is.na(x)], r$bin[idx])
  expect_identical(out$x_woe[!is.na(x)], as.numeric(r$woe[idx]))
  sql <- expect_no_warning(as.character(obwoe_sql(f, style = "case")))
  expect_match(sql, sprintf("x IS NULL THEN %s", OptimalBinningWoE:::.ob_sql_num(r$woe[k])))
  sql0 <- as.character(obwoe_sql(f, style = "case", null_to_na_bin = FALSE, na_value = 7))
  expect_match(sql0, "x IS NULL THEN 7")
  sel <- as.data.frame(obwoe_select(f, detail = "full"))
  expect_identical(sel$bin_upper, c(sort(r$cutpoints), Inf, NA))
  expect_identical(
    OptimalBinningWoE:::.ob_numeric_na_bin(c("a", "b"), numeric(0)),
    NA_integer_
  )
})

test_that("the points SQL scores NULL at the missing-value bin, as the card does", {
  df <- german_credit()
  set.seed(3)
  cat_col <- "credit_history"
  df[[cat_col]][sample(nrow(df), 60)] <- NA
  sc <- suppressWarnings(obwoe_scorecard(df, "target", seed = 6,
    screening = list(iv_min = 0.0001, require_monotonic = "none", allow_degenerate = TRUE),
    control = list(corr_cutoff = 1, drop_negative = FALSE)
  ))
  skip_if_not(cat_col %in% sc$final)
  pts <- sc$points[sc$points$variable == cat_col, ]
  na_row <- which(vapply(strsplit(pts$bin, "%;%", fixed = TRUE), function(p) "NA" %in% p, TRUE))
  expect_length(na_row, 1L)
  sql <- OptimalBinningWoE:::.ob_points_sql(sc, "t", "ansi")
  expect_match(sql, sprintf("%s IS NULL THEN %s", cat_col, pts$points[na_row]), fixed = TRUE)
  nd <- df[is.na(df[[cat_col]]), ][1:3, ]
  card <- predict(sc, nd, type = "card")
  bins <- obwoe_apply(nd, sc$binning, keep_original = FALSE)
  expect_true(all(bins[[paste0(cat_col, "_bin")]] == pts$bin[na_row]))
  expect_identical(card, OptimalBinningWoE:::.ob_card_score(bins, sc$points, sc$final))
})
