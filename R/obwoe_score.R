# ============================================================================ #
# Score scaling, points allocation, redundancy pruning and stability           #
# ============================================================================ #

#' @title Points-to-Double-the-Odds Score Scaling
#'
#' @description
#' Computes the two constants that turn a model's log-odds into scorecard
#' points: the \emph{factor} that fixes how many points double the odds, and
#' the \emph{offset} that anchors a reference score to a reference odds.
#'
#' @param pdo Numeric. Points to Double the Odds. Default \code{20}.
#' @param score_ref Numeric. The reference score. Default \code{600}.
#' @param odds_ref Numeric. The good-to-bad odds at \code{score_ref}, expressed
#'   as a ratio (\code{50} means 50 non-events per event). Default \code{50}.
#' @param direction Character string fixing which end of the scale is safe:
#'   \code{"higher_is_safer"} (default, the scorecard convention) or
#'   \code{"higher_is_riskier"}.
#'
#' @return An object of class \code{"obwoe_scaling"}: a list with \code{pdo},
#'   \code{score_ref}, \code{odds_ref}, \code{direction}, \code{factor} and
#'   \code{offset}.
#'
#' @details
#' Scorecards are scaled on the \strong{good-to-bad} odds:
#'
#' \deqn{\mathrm{Score} = \mathrm{Offset} + \mathrm{Factor}\cdot
#'   \ln(\mathrm{odds}_{good:bad}), \qquad
#'   \mathrm{Factor} = \frac{\mathrm{PDO}}{\ln 2}, \qquad
#'   \mathrm{Offset} = \mathrm{Score}_0 - \mathrm{Factor}\cdot\ln(\mathrm{Odds}_0)}
#'
#' Every model in this package predicts the log-odds of the \strong{event}
#' (the bad), which is the other direction, so the deployed form is
#'
#' \deqn{\mathrm{Score} = \mathrm{Offset} - \mathrm{Factor}\cdot \eta}
#'
#' The minus sign is the whole point: a bin with a high Weight of Evidence is a
#' risky bin, and a risky bin must score \emph{fewer} points.
#'
#' Two identities pin the scaling completely, and \code{\link{obwoe_score}}
#' is tested against both:
#' \itemize{
#'   \item \strong{Anchor.} A case whose odds are \code{odds_ref} scores exactly
#'     \code{score_ref}.
#'   \item \strong{Scale.} Doubling the good-to-bad odds adds exactly
#'     \code{pdo} points, at every point of the scale.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' @seealso \code{\link{obwoe_score}} to apply the scaling,
#'   \code{\link{obwoe_scorecard}} for the full pipeline.
#'
#' @examples
#' scaling <- obwoe_scale(pdo = 20, score_ref = 600, odds_ref = 50)
#' scaling
#'
#' # a case at the reference odds scores the reference score
#' obwoe_score(-log(50), scaling)
#'
#' # halving the odds of being good costs exactly one PDO
#' obwoe_score(-log(25), scaling)
#'
#' @export
obwoe_scale <- function(pdo = 20,
                        score_ref = 600,
                        odds_ref = 50,
                        direction = c("higher_is_safer", "higher_is_riskier")) {
  direction <- match.arg(direction)

  if (!is.numeric(pdo) || length(pdo) != 1L || !is.finite(pdo) || pdo <= 0) {
    stop("'pdo' must be a single positive number.")
  }
  if (!is.numeric(score_ref) || length(score_ref) != 1L || !is.finite(score_ref)) {
    stop("'score_ref' must be a single finite number.")
  }
  if (!is.numeric(odds_ref) || length(odds_ref) != 1L || !is.finite(odds_ref) ||
    odds_ref <= 0) {
    stop("'odds_ref' must be a single positive number.")
  }

  factor_ <- pdo / log(2)

  out <- list(
    pdo = pdo,
    score_ref = score_ref,
    odds_ref = odds_ref,
    direction = direction,
    factor = factor_,
    offset = score_ref - factor_ * log(odds_ref)
  )
  class(out) <- "obwoe_scaling"
  out
}


#' @method print obwoe_scaling
#' @export
print.obwoe_scaling <- function(x, ...) {
  cat("Scorecard scaling\n")
  cat(sprintf(
    "  %g points double the odds; %g points at %g:1 good:bad\n",
    x$pdo, x$score_ref, x$odds_ref
  ))
  cat(sprintf("  factor = %.6f, offset = %.6f\n", x$factor, x$offset))
  cat(sprintf("  score = offset - factor * link   (%s)\n", x$direction))
  invisible(x)
}


#' @title Turn Log-Odds into Scorecard Points
#'
#' @description
#' Applies a \code{\link{obwoe_scale}} scaling to the linear predictor of a
#' model that predicts the log-odds of the event.
#'
#' @param link Numeric vector of log-odds of the event, as returned by
#'   \code{predict(fit, type = "link")}.
#' @param scaling An \code{"obwoe_scaling"} object from \code{\link{obwoe_scale}}.
#' @param round Logical. Round to whole points? Default \code{FALSE}; the
#'   unrounded score is what the additivity identity holds for.
#'
#' @return Numeric vector of scores, of the same length as \code{link}.
#'
#' @seealso \code{\link{obwoe_scale}}
#'
#' @examples
#' scaling <- obwoe_scale()
#' obwoe_score(c(-log(50), -log(100), 0), scaling)
#'
#' @export
obwoe_score <- function(link, scaling, round = FALSE) {
  if (!inherits(scaling, "obwoe_scaling")) {
    stop("'scaling' must be an 'obwoe_scaling' object from obwoe_scale().")
  }
  link <- as.numeric(link)

  sign_ <- if (identical(scaling$direction, "higher_is_safer")) -1 else 1
  out <- scaling$offset + sign_ * scaling$factor * link

  if (isTRUE(round)) round(out) else out
}


#' @title Internal: Siddiqi Points Allocation
#'
#' @description
#' Distributes a fitted additive model over its bins as a fixed points table.
#'
#' With \eqn{k} variables in the model, an intercept \eqn{\alpha} and slopes
#' \eqn{\beta_j} on the Weight of Evidence,
#'
#' \deqn{\mathrm{points}_{ij} = \frac{\mathrm{Offset}}{k} -
#'   \mathrm{Factor}\left(\beta_j\,\mathrm{WoE}_{ij} + \frac{\alpha}{k}\right)}
#'
#' so that summing one row per variable reproduces the score exactly:
#' \eqn{\sum_j \mathrm{points}_{ij} = \mathrm{Offset} -
#' \mathrm{Factor}(\alpha + \sum_j \beta_j \mathrm{WoE}_{ij}) =
#' \mathrm{Offset} - \mathrm{Factor}\,\eta}.
#'
#' @param binning An \code{"obwoe"} object.
#' @param coefficients Named numeric with \code{"(Intercept)"} first.
#' @param features Character vector of the variables in the model, in the order
#'   their coefficients appear.
#' @param scaling An \code{"obwoe_scaling"} object.
#' @param na_woe Weight of Evidence assigned to a value that falls in no fitted
#'   bin. The per-variable points that follow from it are attached to the result
#'   as the \code{"points_na"} attribute, so that the R card score and the
#'   generated SQL fall back on one and the same number.
#' @param bin_separator Separator inside merged categorical bin labels.
#'
#' @return A \code{data.frame} with one row per variable and bin, carrying a
#'   named numeric \code{"points_na"} attribute.
#'
#' @keywords internal
.ob_points <- function(binning, coefficients, features, scaling, na_woe = 0,
                       bin_separator = "%;%") {
  alpha <- unname(coefficients[["(Intercept)"]])
  beta <- coefficients[setdiff(names(coefficients), "(Intercept)")]
  k <- length(features)

  sign_ <- if (identical(scaling$direction, "higher_is_safer")) -1 else 1
  base_per_var <- scaling$offset / k + sign_ * scaling$factor * alpha / k

  blocks <- lapply(seq_along(features), function(j) {
    feat <- features[j]
    res <- binning$results[[feat]]
    woe <- as.numeric(res$woe)
    b <- unname(beta[[feat]])

    parts <- strsplit(as.character(res$bin), bin_separator, fixed = TRUE)
    cats <- vapply(parts, paste, character(1), collapse = ", ")

    cp <- res$cutpoints
    lower <- rep(NA_real_, length(woe))
    upper <- rep(NA_real_, length(woe))
    if (identical(res$type, "numerical") && !is.null(cp) && length(cp) > 0L) {
      cp <- sort(unique(as.numeric(cp)))
      if (length(cp) + 1L == length(woe)) {
        lower <- c(-Inf, cp)
        upper <- c(cp, Inf)
      }
    }

    raw <- base_per_var + sign_ * scaling$factor * b * woe

    data.frame(
      variable = feat,
      var_order = j,
      bin_id = seq_along(woe),
      bin = as.character(res$bin),
      bin_lower = lower,
      bin_upper = upper,
      n_categories = if (identical(res$type, "categorical")) {
        as.integer(vapply(parts, length, integer(1)))
      } else {
        NA_integer_
      },
      categories = if (identical(res$type, "categorical")) cats else NA_character_,
      count = as.integer(res$count),
      pos = as.integer(res$count_pos),
      neg = as.integer(res$count_neg),
      pos_rate = as.numeric(res$count_pos) / as.numeric(res$count),
      woe = woe,
      iv_bin = as.numeric(res$iv),
      beta = b,
      points_raw = raw,
      points = round(raw),
      stringsAsFactors = FALSE
    )
  })

  out <- as.data.frame(.ob_rbind(blocks), stringsAsFactors = FALSE)

  # Per-variable point range: the committee's test of whether any one variable
  # can dominate a decision on its own.
  rng <- stats::aggregate(points ~ variable, data = out, FUN = range)
  rng <- data.frame(
    variable = rng$variable,
    points_var_min = rng$points[, 1],
    points_var_max = rng$points[, 2],
    stringsAsFactors = FALSE
  )
  out <- merge(out, rng, by = "variable", all.x = TRUE, sort = FALSE)
  out <- out[order(out$var_order, out$bin_id), , drop = FALSE]
  rownames(out) <- NULL

  # Points for a value in no fitted bin: the same base term every bin of the
  # variable carries, plus whatever na_woe is worth under that variable's slope.
  attr(out, "points_na") <- stats::setNames(
    round(base_per_var + sign_ * scaling$factor *
      unname(beta[features]) * as.numeric(na_woe)),
    features
  )
  out
}


#' @title Prune Redundant Variables by Correlation
#'
#' @description
#' Greedy, iterative removal of variables that carry the same information as a
#' better-ranked one. Information Value ranks variables one at a time; two
#' variables can both be strong and say the same thing, which a model on Weight
#' of Evidence shows as an unstable or sign-flipped coefficient.
#'
#' @param x A \code{data.frame} of numeric columns to be compared — normally
#'   the WoE-transformed predictors, which is the space the model sees — or a
#'   pre-computed pairwise table from \code{\link{obcorr}} with columns
#'   \code{x}, \code{y} and a correlation column.
#' @param ranking Character vector of variable names, best first. Variables
#'   absent from \code{ranking} are treated as ranked last.
#' @param cutoff Numeric in \eqn{(0, 1]}. Absolute correlation at or above
#'   which two variables are considered redundant. Default \code{0.70}.
#' @param method Character string passed to \code{\link{obcorr}} when \code{x}
#'   is a \code{data.frame} of columns. Default \code{"pearson"}.
#'
#' @return A list with:
#'   \describe{
#'     \item{\code{keep}}{Character vector of surviving variables.}
#'     \item{\code{dropped}}{\code{data.frame} of \code{variable},
#'       \code{correlated_with} and \code{correlation} — one row per removal,
#'       in the order the removals happened.}
#'     \item{\code{pairs}}{The pairwise table, with an \code{abs_corr} column.}
#'     \item{\code{cutoff}}{The cutoff used.}
#'   }
#'
#' @details
#' The pass is iterative on purpose. Evaluating every pair independently
#' removes both \eqn{B} and \eqn{C} from a chain \eqn{A \sim B \sim C} even
#' when \eqn{B} and \eqn{C} are unrelated to each other: once \eqn{B} is gone,
#' the pair \eqn{(B, C)} no longer exists. Each iteration therefore drops one
#' variable — the worst-ranked member of the strongest surviving pair — and
#' recomputes what is left.
#'
#' @seealso \code{\link{obcorr}}, \code{\link{obwoe_select}}
#'
#' @examples
#' set.seed(1)
#' n <- 500
#' a <- rnorm(n)
#' df <- data.frame(a = a, b = a + rnorm(n, 0, 0.2), c = rnorm(n))
#'
#' # b duplicates a; a is ranked better, so b goes
#' obwoe_prune(df, ranking = c("a", "b", "c"), cutoff = 0.7)
#'
#' @export
obwoe_prune <- function(x, ranking, cutoff = 0.70, method = "pearson") {
  if (!is.character(ranking) || length(ranking) == 0L) {
    stop("'ranking' must be a non-empty character vector of variable names.")
  }
  if (!is.numeric(cutoff) || length(cutoff) != 1L || is.na(cutoff) ||
    cutoff <= 0 || cutoff > 1) {
    stop("'cutoff' must be a single number in (0, 1].")
  }

  is_pairs <- is.data.frame(x) && all(c("x", "y") %in% names(x))

  if (is_pairs) {
    pairs <- x
    corr_col <- setdiff(names(pairs), c("x", "y"))[1L]
    if (is.na(corr_col)) {
      stop("The pairwise table must carry a correlation column besides 'x' and 'y'.")
    }
    pairs$abs_corr <- abs(as.numeric(pairs[[corr_col]]))
  } else {
    if (!is.data.frame(x)) {
      stop("'x' must be a data.frame of numeric columns or an obcorr() table.")
    }
    # A single column has no pair to be redundant with.
    if (ncol(x) < 2L) {
      return(list(
        keep = names(x), dropped = .ob_empty_drops(),
        pairs = NULL, cutoff = cutoff
      ))
    }
    pairs <- obcorr(x, method = method)
    corr_col <- setdiff(names(pairs), c("x", "y"))[1L]
    pairs$abs_corr <- abs(as.numeric(pairs[[corr_col]]))
  }

  # Rank position: unlisted variables sort last, so they are dropped first.
  rank_of <- function(v) {
    p <- match(v, ranking)
    ifelse(is.na(p), length(ranking) + 1L, p)
  }

  alive <- unique(c(pairs$x, pairs$y))
  alive <- union(intersect(ranking, alive), alive)
  drops <- .ob_empty_drops()

  repeat {
    live <- pairs[pairs$x %in% alive & pairs$y %in% alive &
      pairs$abs_corr >= cutoff, , drop = FALSE]
    if (nrow(live) == 0L) break

    # Strongest surviving pair; drop its worse-ranked member.
    i <- which.max(live$abs_corr)
    a <- live$x[i]
    b <- live$y[i]
    loser <- if (rank_of(a) > rank_of(b)) a else b
    winner <- if (identical(loser, a)) b else a

    drops <- rbind(drops, data.frame(
      variable = loser, correlated_with = winner,
      correlation = live[[corr_col]][i], stringsAsFactors = FALSE
    ))
    alive <- setdiff(alive, loser)
  }

  rownames(drops) <- NULL
  list(
    keep = ranking[ranking %in% alive],
    dropped = drops,
    pairs = pairs,
    cutoff = cutoff
  )
}


#' @keywords internal
.ob_empty_drops <- function() {
  data.frame(
    variable = character(0), correlated_with = character(0),
    correlation = numeric(0), stringsAsFactors = FALSE
  )
}


#' @title Population Stability Index
#'
#' @description
#' Compares how a variable, or a score, is distributed between two samples —
#' typically the development window and a later vintage.
#'
#' @param base Vector observed in the reference sample.
#' @param compare Vector observed in the sample being monitored.
#' @param levels Character vector of the categories or bin labels to compare
#'   over. Required when the inputs are not numeric; defaults to the union of
#'   the values seen.
#' @param breaks Numeric vector of cut points used to band numeric inputs.
#'   \code{NULL} (default) bands them at the deciles of \code{base}.
#' @param n_groups Integer. Number of bands when \code{breaks} is \code{NULL}.
#'   Default \code{10}.
#'
#' @return A list with \code{psi} (the total), \code{table} (the contribution
#'   of each band) and \code{flag}, one of \code{"stable"}, \code{"watch"} or
#'   \code{"act"} at the conventional 0.10 and 0.25 thresholds.
#'
#' @details
#' \deqn{\mathrm{PSI} = \sum_i (p_i - q_i)\,\ln\frac{p_i}{q_i}}
#'
#' which is the same symmetrised Kullback-Leibler divergence that defines
#' Information Value, applied to two vintages of one variable instead of to two
#' classes of one sample. The conventional reading is \eqn{<0.10} stable,
#' \eqn{0.10}–\eqn{0.25} worth watching, \eqn{>0.25} act.
#'
#' A band that is populated in one sample and empty in the other makes the
#' divergence infinite. That is reported as \code{Inf} rather than smoothed
#' away, because a segment that has vanished is exactly what monitoring exists
#' to catch; the per-band table shows which one.
#'
#' @seealso \code{\link{obwoe_scorecard}}
#'
#' @examples
#' set.seed(1)
#' obwoe_psi(rnorm(1000), rnorm(1000))
#' obwoe_psi(rnorm(1000), rnorm(1000, mean = 0.6))
#'
#' @export
obwoe_psi <- function(base, compare, levels = NULL, breaks = NULL,
                      n_groups = 10L) {
  if (length(base) == 0L || length(compare) == 0L) {
    stop("'base' and 'compare' must both be non-empty.")
  }

  numeric_input <- is.numeric(base) && is.numeric(compare)

  if (numeric_input) {
    if (is.null(breaks)) {
      # Interior quantiles only. Including the 0% and 100% points and then
      # wrapping the result in -Inf/Inf would open two bands that hold nothing
      # but the reference sample's own extremes, and any value of the compared
      # sample beyond them would read as an emptied band, i.e. an infinite PSI
      # for a distribution that has merely shifted.
      probs <- seq(0, 1, length.out = n_groups + 1L)
      probs <- probs[-c(1L, length(probs))]
      breaks <- stats::quantile(base, probs = probs, na.rm = TRUE)
    }
    breaks <- unique(c(-Inf, breaks[is.finite(breaks)], Inf))
    if (length(breaks) < 3L) {
      # A constant reference cannot be banded; there is nothing to compare.
      return(list(
        psi = 0, flag = "stable",
        table = data.frame(
          band = "all", pct_base = 1, pct_compare = 1, psi_band = 0,
          stringsAsFactors = FALSE
        )
      ))
    }
    b <- cut(base, breaks = breaks, include.lowest = TRUE)
    c_ <- cut(compare, breaks = breaks, include.lowest = TRUE)
    lev <- levels(b)
  } else {
    b <- as.character(base)
    c_ <- as.character(compare)
    lev <- if (is.null(levels)) sort(unique(c(b, c_))) else as.character(levels)
  }

  share <- function(v) {
    tb <- table(factor(as.character(v), levels = lev))
    as.numeric(tb) / sum(as.numeric(tb))
  }

  p <- share(b)
  q <- share(c_)

  # An emptied band is a genuine Inf, not something to smooth away.
  contrib <- (p - q) * log(p / q)
  contrib[p == 0 & q == 0] <- 0

  total <- sum(contrib)

  flag <- if (!is.finite(total) || total > 0.25) {
    "act"
  } else if (total > 0.10) {
    "watch"
  } else {
    "stable"
  }

  list(
    psi = total,
    flag = flag,
    table = data.frame(
      band = lev,
      pct_base = p,
      pct_compare = q,
      psi_band = contrib,
      stringsAsFactors = FALSE
    )
  )
}
