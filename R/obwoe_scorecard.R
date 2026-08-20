# ============================================================================ #
# obwoe_scorecard(): the automated scorecard pipeline                          #
# ============================================================================ #

#' @title Control Parameters for the Scorecard Pipeline
#'
#' @description
#' Settings for the stages of \code{\link{obwoe_scorecard}} that have no home
#' in \code{\link{control.obwoe}} or \code{\link{obwoe_select}}.
#'
#' @param corr_cutoff Numeric in \eqn{(0, 1]}. Absolute correlation in the WoE
#'   space at or above which two variables are treated as redundant. Default
#'   \code{0.70}. Set to \code{1} to disable pruning.
#' @param corr_method Character string passed to \code{\link{obcorr}}. Default
#'   \code{"pearson"}.
#' @param pdo,score_ref,odds_ref,direction Passed to \code{\link{obwoe_scale}}.
#' @param n_groups Integer. Number of score bands in the gains and stability
#'   tables. Default \code{10}.
#' @param na_woe Numeric. WoE given to a value that falls in no fitted bin —
#'   an unseen category, or a missing value where the binner built no missing
#'   bin. Default \code{0}, which is the population-average contribution.
#' @param drop_negative Logical. Refit without any variable whose coefficient
#'   on WoE comes out negative? Default \code{TRUE}. A negative coefficient
#'   means the variable is fighting the rest of the model, and a scorecard with
#'   one cannot be signed off.
#' @param max_abs_coef Numeric. Coefficients above this in absolute value are
#'   flagged as separation. Default \code{15}.
#' @param min_events_per_variable Numeric. Events per variable in the model
#'   below which the fit is flagged as thin. Default \code{20}, the usual
#'   rule of thumb.
#' @param engine_fallback Logical. If the requested engine's package is not
#'   installed, fall back to \code{"glm"} with a warning instead of failing?
#'   Default \code{FALSE}.
#' @param overwrite Logical. Overwrite \code{file} if it exists? Default
#'   \code{TRUE}.
#' @param digits Integer. Rounding used in the workbook. Default \code{6}.
#'
#' @return An object of class \code{"obwoe_scorecard_control"}.
#'
#' @seealso \code{\link{obwoe_scorecard}}
#'
#' @examples
#' control.obwoe_scorecard(pdo = 25, score_ref = 700, corr_cutoff = 0.8)
#'
#' @export
control.obwoe_scorecard <- function(corr_cutoff = 0.70,
                                    corr_method = "pearson",
                                    pdo = 20,
                                    score_ref = 600,
                                    odds_ref = 50,
                                    direction = "higher_is_safer",
                                    n_groups = 10L,
                                    na_woe = 0,
                                    drop_negative = TRUE,
                                    max_abs_coef = 15,
                                    min_events_per_variable = 20,
                                    engine_fallback = FALSE,
                                    overwrite = TRUE,
                                    digits = 6L) {
  ctrl <- list(
    corr_cutoff = corr_cutoff,
    corr_method = corr_method,
    pdo = pdo,
    score_ref = score_ref,
    odds_ref = odds_ref,
    direction = direction,
    n_groups = as.integer(n_groups),
    na_woe = na_woe,
    drop_negative = isTRUE(drop_negative),
    max_abs_coef = max_abs_coef,
    min_events_per_variable = min_events_per_variable,
    engine_fallback = isTRUE(engine_fallback),
    overwrite = isTRUE(overwrite),
    digits = as.integer(digits)
  )
  class(ctrl) <- "obwoe_scorecard_control"
  ctrl
}


#' @keywords internal
.ob_msg <- function(verbose, fmt, ...) {
  if (isTRUE(verbose)) message(sprintf(fmt, ...))
}


#' @title Internal: Resolve the Target into a 0/1 Event Vector
#'
#' @description
#' Records which level was treated as the event, because getting this wrong
#' inverts the score while leaving every other statistic — IV, KS, Gini, the
#' coefficient signs — completely unchanged.
#'
#' @param y The raw target column.
#'
#' @return A list with \code{y} (integer 0/1) and \code{event_level}.
#'
#' @keywords internal
.ob_resolve_target <- function(y) {
  if (is.factor(y)) {
    lev <- levels(y)
    if (length(lev) != 2L) {
      stop(sprintf(
        "The target must have exactly two levels; found %d.", length(lev)
      ))
    }
    return(list(y = as.integer(y) - 1L, event_level = lev[2L]))
  }

  if (is.logical(y)) {
    return(list(y = as.integer(y), event_level = "TRUE"))
  }

  yi <- as.integer(y)
  vals <- sort(unique(yi[!is.na(yi)]))
  if (!identical(vals, c(0L, 1L))) {
    stop(sprintf(
      "A numeric target must be coded 0/1; found %s.",
      paste(utils::head(vals, 5), collapse = ", ")
    ))
  }
  list(y = yi, event_level = "1")
}


#' @title Automated Scorecard Pipeline
#'
#' @description
#' Runs a credit or fraud scorecard end to end in one call: optimal binning,
#' variable screening by Information Value and bin ordering, redundancy pruning
#' in the Weight of Evidence space, model fitting, and score scaling to points.
#' Optionally writes the whole thing to a multi-sheet \code{.xlsx} workbook that
#' a model committee, a validation team and a deployment engineer can each read.
#'
#' @param data A \code{data.frame} holding the development sample.
#' @param target Character string naming the target column. Binary; a factor's
#'   \strong{second} level is the event, and which level that was is recorded in
#'   the result.
#' @param feature Character vector of candidate predictors. \code{NULL}
#'   (default) uses every column except the target, the split column and
#'   anything named in \code{exclude}.
#' @param exclude Character vector of columns to keep out of the model —
#'   identifiers, dates, and above all fields populated after the outcome was
#'   known. Default \code{NULL}.
#' @param split How the development sample is divided. One of:
#'   \describe{
#'     \item{a number in \eqn{(0, 1)}}{proportion of rows used for training,
#'       sampled at random with the event rate preserved (default \code{0.7});}
#'     \item{a column name}{an out-of-time split: the earliest value of that
#'       column trains, the rest validates;}
#'     \item{an integer or logical vector}{the training rows, given explicitly;}
#'     \item{\code{NULL}}{no split — everything trains, and every reported
#'       metric is in-sample. Flagged as such throughout.}
#'   }
#' @param validation Optional named list of extra \code{data.frame}s to score
#'   and report on. They never influence any fitted quantity.
#' @param file Optional path to a \code{.xlsx} file. \code{NULL} (default)
#'   computes everything and writes nothing; pass the result to
#'   \code{\link{obwoe_report}} later to produce the workbook.
#' @param binning Named list of arguments for \code{\link{obwoe}} — typically
#'   \code{min_bins}, \code{max_bins}, \code{algorithm}, \code{control}.
#' @param screening Named list of arguments for \code{\link{obwoe_select}} —
#'   typically \code{iv_min}, \code{iv_max}, \code{require_monotonic},
#'   \code{min_bin_pct}, \code{top_n}.
#' @param engine Character string naming the model engine — \code{"glm"}
#'   (default), \code{"obwoe"} (the package's own C++ logistic regression) or
#'   \code{"glmnet"} — or a list supplying \code{fit}, \code{link} and
#'   \code{coef} of your own. See Details.
#' @param engine_args Named list forwarded to the engine's fitting call.
#' @param control An \code{"obwoe_scorecard_control"} object from
#'   \code{\link{control.obwoe_scorecard}}.
#' @param seed Optional integer seed, for a reproducible split.
#' @param verbose Logical. Report progress? Default \code{FALSE}.
#'
#' @return An object of class \code{"obwoe_scorecard"}: a list with
#'   \describe{
#'     \item{\code{call}, \code{version}, \code{built_on}, \code{seed}}{provenance}
#'     \item{\code{target}, \code{event_level}, \code{candidates},
#'       \code{selected}, \code{final}}{the funnel, as character vectors}
#'     \item{\code{binning}}{the \code{"obwoe"} object, fitted on the training
#'       rows only}
#'     \item{\code{screening}, \code{screening_bins}}{\code{\link{obwoe_select}}
#'       at both levels of detail, plus a \code{stage} column naming the step
#'       that rejected each variable: \code{"in_model"},
#'       \code{"sign_rejected"} (negative WoE coefficient),
#'       \code{"corr_pruned"}, \code{"constant_woe"} (one WoE value after the
#'       transform) or \code{"screened_out"} (failed the IV or ordering rules)}
#'     \item{\code{correlation}}{the \code{\link{obwoe_prune}} result}
#'     \item{\code{control}}{the resolved \code{"obwoe_scorecard_control"} this
#'       object was actually built with (as of 1.13.1) -- \code{predict()} and
#'       \code{\link{obwoe_report}} read \code{na_woe} from here so every
#'       output agrees on the fallback for an unseen category or an
#'       unmodelled missing value}
#'     \item{\code{engine}}{name, requested versus used, and whether additive}
#'     \item{\code{model}}{the raw fitted object — engine-specific and outside
#'       the supported interface}
#'     \item{\code{coefficients}, \code{diagnostics}}{the additive fit and its
#'       sign, separation and convergence checks}
#'     \item{\code{scaling}}{the \code{\link{obwoe_scale}} constants}
#'     \item{\code{points}}{the fixed points table, one row per variable and bin}
#'     \item{\code{samples}}{per sample: scores, gains table, headline metrics}
#'     \item{\code{stability}}{PSI for the score and for every variable in the
#'       model}
#'     \item{\code{warnings}}{everything the pipeline wants a human to read}
#'   }
#'
#' @details
#' \subsection{The split comes first}{
#'
#' Binning is \strong{supervised}: the cut points and the WoE are both chosen
#' using the target. Fitting them on rows that are later used to measure
#' performance is leakage, and it is the failure this function is built to make
#' impossible — the split happens before anything is fitted, the binning sees
#' the training rows only, and every other sample is transformed through
#' \code{\link{obwoe_apply}}. Screening statistics are computed on the training
#' rows for the same reason.
#' }
#'
#' \subsection{The binning is fitted once}{
#'
#' One \code{"obwoe"} object drives screening, correlation, the model, the
#' points, the stability report and the generated SQL. Refitting the binning
#' for the model — with, say, a different \code{bin_cutoff} — produces a
#' workbook whose evidence describes a transform the deployed model never saw,
#' and every number in it looks internally consistent.
#' }
#'
#' \subsection{Points}{
#'
#' With \eqn{k} variables in the model, an intercept \eqn{\alpha} and slopes
#' \eqn{\beta_j},
#'
#' \deqn{\mathrm{points}_{ij} = \frac{\mathrm{Offset}}{k} -
#'   \mathrm{Factor}\left(\beta_j\,\mathrm{WoE}_{ij} + \frac{\alpha}{k}\right)}
#'
#' so the points of the bins an applicant falls into sum to their score. The
#' table is \strong{fixed}: each bin rounds to one integer, the same for
#' everyone, because a bin whose value depended on who was being scored would
#' not be a scorecard. Rounding \eqn{k} terms moves the total by at most
#' \eqn{k/2} points against the unrounded model score; that drift is measured
#' and reported rather than hidden, and the card — the sum of the integers — is
#' the deployed definition.
#' }
#'
#' \subsection{Engines}{
#'
#' An engine is three functions: \code{fit(x, y, args)}, \code{link(object, x)}
#' returning the log-odds of the event, and \code{coef(object)} returning the
#' additive coefficients \strong{or \code{NULL}}. Returning \code{NULL} declares
#' the model non-additive: gains, stability and score bands are still produced,
#' but no points table is, because a model that is not a sum of per-variable
#' terms cannot be decomposed into one. Supplying \code{engine} as a list of
#' those three functions is how a gradient-boosted or \pkg{tidymodels} fit is
#' plugged in without the package depending on either.
#' }
#'
#' \subsection{What it refuses to do}{
#'
#' The pipeline stops, rather than producing a plausible workbook, when the
#' shortlist is empty, when the fit does not converge, or when the requested
#' engine's package is missing. It drops constant predictors before fitting and,
#' by default, refits without any variable whose WoE coefficient comes out
#' negative. Everything it does silently absorb is listed in \code{$warnings}
#' and written to the workbook.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring and Its
#' Applications. \emph{SIAM}. \doi{10.1137/1.9780898718317}
#'
#' Anderson, R. (2007). The Credit Scoring Toolkit. \emph{Oxford University
#' Press}.
#'
#' @seealso
#' \code{\link{obwoe_report}} to write the workbook,
#' \code{\link{predict.obwoe_scorecard}} to score new data,
#' \code{\link{obwoe_select}} for the screening rules,
#' \code{\link{obwoe_sql}} for the deployment SQL,
#' \code{\link{obwoe_scale}} and \code{\link{obwoe_psi}}.
#'
#' @examples
#' \donttest{
#' german <- read.csv(
#'   gzfile(system.file("extdata", "germancredit.csv.gz",
#'     package = "OptimalBinningWoE"
#'   )),
#'   stringsAsFactors = FALSE
#' )
#' german$default <- 1L - german$credit_risk
#' german$credit_risk <- NULL
#'
#' sc <- obwoe_scorecard(german, target = "default", seed = 42)
#' sc
#'
#' head(sc$points[, c("variable", "bin", "woe", "points")])
#' sc$samples$train$metrics
#'
#' # write the workbook
#' # obwoe_report(sc, file = "scorecard.xlsx")
#' }
#'
#' @importFrom stats aggregate setNames
#' @export
obwoe_scorecard <- function(data,
                            target,
                            feature = NULL,
                            exclude = NULL,
                            split = 0.7,
                            validation = NULL,
                            file = NULL,
                            binning = list(),
                            screening = list(),
                            engine = "glm",
                            engine_args = list(),
                            control = control.obwoe_scorecard(),
                            seed = NULL,
                            verbose = FALSE) {
  call <- match.call()
  warn <- character(0)
  add_warning <- function(msg) warn <<- c(warn, msg)

  # ------------------------------------------------------------------------ #
  # 1. Validation
  # ------------------------------------------------------------------------ #
  if (!is.data.frame(data)) stop("'data' must be a data.frame.")
  if (!is.character(target) || length(target) != 1L) {
    stop("'target' must be a single column name.")
  }
  if (!target %in% names(data)) {
    stop(sprintf("Target column '%s' not found in data.", target))
  }
  if (!inherits(control, "obwoe_scorecard_control")) {
    if (is.list(control)) {
      control <- do.call(control.obwoe_scorecard, control)
    } else {
      stop("'control' must be created by control.obwoe_scorecard().")
    }
  }
  if (!is.list(binning) || !is.list(screening) || !is.list(engine_args)) {
    stop("'binning', 'screening' and 'engine_args' must be lists.")
  }
  .ob_check_names(binning, obwoe, "binning")
  .ob_check_names(screening, obwoe_select, "screening")

  # The workbook is written last but is checked first: a missing package or an
  # unwritable path is worth knowing before the binning and the fit are run,
  # not after they have been computed and are about to be discarded.
  if (!is.null(file)) {
    if (!is.character(file) || length(file) != 1L || is.na(file)) {
      stop("'file' must be a single path, or NULL.")
    }
    if (!.ob_has_openxlsx()) {
      stop(paste(
        "Writing the workbook needs the 'openxlsx' package.",
        "Install it with install.packages(\"openxlsx\"), or use file = NULL",
        "and call obwoe_report() later."
      ))
    }
    dir <- dirname(file)
    if (!dir.exists(dir)) {
      stop(sprintf("Directory '%s' does not exist, so '%s' cannot be written.",
        dir, file
      ))
    }
    if (file.access(dir, mode = 2L) != 0L) {
      stop(sprintf("Directory '%s' is not writable.", dir))
    }
  }

  if (!is.null(seed)) set.seed(as.integer(seed))

  tgt <- .ob_resolve_target(data[[target]])
  y_all <- tgt$y
  if (anyNA(y_all)) stop("The target contains missing values.")

  split_col <- if (is.character(split) && length(split) == 1L &&
    split %in% names(data)) {
    split
  } else {
    NULL
  }

  candidates <- if (is.null(feature)) {
    setdiff(names(data), c(target, exclude, split_col))
  } else {
    setdiff(feature, c(target, exclude))
  }
  missing_cols <- setdiff(candidates, names(data))
  if (length(missing_cols) > 0L) {
    stop(sprintf(
      "Candidate column(s) not found: %s",
      paste(missing_cols, collapse = ", ")
    ))
  }
  if (length(candidates) == 0L) stop("No candidate predictors to screen.")

  # ------------------------------------------------------------------------ #
  # 2. Split, before anything is fitted
  # ------------------------------------------------------------------------ #
  idx <- .ob_train_index(data, split, y_all)
  train <- data[idx$train, , drop = FALSE]
  holdout <- if (length(idx$test) > 0L) data[idx$test, , drop = FALSE] else NULL

  y_train <- y_all[idx$train]
  if (length(unique(y_train)) < 2L) {
    stop("The training rows contain only one class.")
  }
  .ob_msg(verbose, "Split: %d training rows, %d held out.",
    nrow(train), if (is.null(holdout)) 0L else nrow(holdout))

  if (is.null(holdout) && is.null(validation)) {
    add_warning(paste(
      "No hold-out sample: every reported metric is in-sample and optimistic,",
      "because the bins, the shortlist and the coefficients all come from",
      "these rows."
    ))
  }

  # ------------------------------------------------------------------------ #
  # 3. Binning, on the training rows only, once
  # ------------------------------------------------------------------------ #
  .ob_msg(verbose, "Binning %d candidates...", length(candidates))
  bin_args <- utils::modifyList(
    list(data = train, target = target, feature = candidates), binning
  )
  binning_fit <- do.call(obwoe, bin_args)

  n_failed <- sum(binning_fit$summary$error)
  if (n_failed > 0L) {
    add_warning(sprintf(
      "%d candidate(s) could not be binned and cannot enter the model.", n_failed
    ))
  }

  # ------------------------------------------------------------------------ #
  # 4. Screening
  # ------------------------------------------------------------------------ #
  sel_args <- utils::modifyList(list(obj = binning_fit), screening)
  screening_tbl <- as.data.frame(do.call(obwoe_select, sel_args),
    stringsAsFactors = FALSE
  )
  screening_bins <- as.data.frame(
    do.call(obwoe_select, utils::modifyList(sel_args, list(detail = "full"))),
    stringsAsFactors = FALSE
  )

  shortlist <- screening_tbl$feature[screening_tbl$selected]
  .ob_msg(verbose, "Screening kept %d of %d.", length(shortlist), length(candidates))

  # Pure noise clears a low IV floor by chance when the sample is small
  # relative to the number of candidates. Say so rather than let it pass.
  events <- sum(y_train == 1L)
  if (nrow(train) < 250 * length(candidates) / 20 && length(shortlist) > 0L) {
    add_warning(sprintf(
      paste(
        "%d training rows for %d candidates: at this ratio an Information",
        "Value floor is a weak filter and uninformative variables can clear it",
        "by chance. Confirm the shortlist on the hold-out sample."
      ),
      nrow(train), length(candidates)
    ))
  }

  if (length(shortlist) == 0L) {
    stop(sprintf(
      paste0(
        "Screening rejected every candidate, so there is no model to fit. ",
        "Reasons: %s. Relax the rules in 'screening' (see ?obwoe_select)."
      ),
      paste(names(sort(table(screening_tbl$reason), decreasing = TRUE)),
        collapse = "; "
      )
    ))
  }

  # ------------------------------------------------------------------------ #
  # 5. WoE transform, then redundancy pruning in that space
  # ------------------------------------------------------------------------ #
  woe_train <- .ob_woe_matrix(train, binning_fit, shortlist, control$na_woe)

  screened <- shortlist

  constant <- names(woe_train)[vapply(woe_train, function(v) {
    length(unique(v[is.finite(v)])) < 2L
  }, logical(1))]
  if (length(constant) > 0L) {
    add_warning(sprintf(
      "Dropped before fitting, constant after the WoE transform: %s.",
      paste(constant, collapse = ", ")
    ))
    shortlist <- setdiff(shortlist, constant)
    woe_train <- woe_train[, shortlist, drop = FALSE]
  }
  if (length(shortlist) == 0L) {
    stop("Every screened variable is constant after the WoE transform.")
  }

  pruning <- if (control$corr_cutoff < 1 && length(shortlist) > 1L) {
    obwoe_prune(woe_train,
      ranking = shortlist, cutoff = control$corr_cutoff,
      method = control$corr_method
    )
  } else {
    list(
      keep = shortlist, dropped = .ob_empty_drops(),
      pairs = NULL, cutoff = control$corr_cutoff
    )
  }
  final <- pruning$keep
  .ob_msg(verbose, "Correlation pruning kept %d of %d.", length(final), length(shortlist))

  if (length(final) == 0L) stop("Correlation pruning removed every variable.")

  # ------------------------------------------------------------------------ #
  # 6. Fit
  # ------------------------------------------------------------------------ #
  eng <- .ob_engine_get(engine, fallback = control$engine_fallback)

  if (events < control$min_events_per_variable * length(final)) {
    add_warning(sprintf(
      paste(
        "%d events for %d variables (%.1f per variable): below the",
        "%g-events-per-variable rule of thumb, so the coefficients are",
        "unstable."
      ),
      events, length(final), events / length(final),
      control$min_events_per_variable
    ))
  }

  offered <- final

  fitted <- .ob_fit_checked(
    eng, woe_train[, final, drop = FALSE], y_train, engine_args,
    control, add_warning
  )
  final <- fitted$features
  coefs <- fitted$coefficients

  # ------------------------------------------------------------------------ #
  # 7. Scale and decompose into points
  # ------------------------------------------------------------------------ #
  scaling <- obwoe_scale(
    pdo = control$pdo, score_ref = control$score_ref,
    odds_ref = control$odds_ref, direction = control$direction
  )

  points <- NULL
  if (!is.null(coefs)) {
    points <- .ob_points(binning_fit, coefs, final, scaling,
      na_woe = control$na_woe
    )
  } else {
    add_warning(paste(
      "The engine declared itself non-additive, so no points table was",
      "produced. Scores, gains and stability are still available."
    ))
  }

  # ------------------------------------------------------------------------ #
  # 8. Score every sample
  # ------------------------------------------------------------------------ #
  samples <- list(train = train)
  if (!is.null(holdout)) samples$holdout <- holdout
  if (!is.null(validation)) {
    if (!is.list(validation) || is.null(names(validation)) ||
      any(!nzchar(names(validation)))) {
      stop("'validation' must be a named list of data.frames.")
    }
    samples <- c(samples, validation)
  }

  scored <- lapply(names(samples), function(nm) {
    .ob_score_sample(
      samples[[nm]], nm, binning_fit, final, eng, fitted$model, scaling,
      points, target, control, add_warning
    )
  })
  names(scored) <- names(samples)

  band_breaks <- .ob_score_breaks(scored$train$score, control$n_groups)
  scored <- lapply(scored, function(s) {
    s$gains <- .ob_score_gains(s$score, s$y, band_breaks)
    s$metrics <- .ob_score_metrics(s$score, s$y)
    s
  })

  # ------------------------------------------------------------------------ #
  # 9. Stability against the training sample
  # ------------------------------------------------------------------------ #
  stability <- .ob_stability(scored, binning_fit, final, band_breaks)

  # ------------------------------------------------------------------------ #
  # 10. Assemble
  # ------------------------------------------------------------------------ #
  # Why a variable is not in the model is the whole point of this table, so
  # each exit is labelled with the stage that actually rejected it. Assigned
  # in pipeline order, so the last stage a variable reached is the one shown.
  f <- screening_tbl$feature
  stage <- rep("screened_out", length(f))
  stage[f %in% constant] <- "constant_woe"
  stage[f %in% setdiff(screened, c(constant, offered))] <- "corr_pruned"
  stage[f %in% setdiff(offered, final)] <- "sign_rejected"
  stage[f %in% final] <- "in_model"
  screening_tbl$stage <- stage

  out <- list(
    call = call,
    version = list(
      package = as.character(utils::packageVersion("OptimalBinningWoE")),
      R = paste(R.version$major, R.version$minor, sep = ".")
    ),
    built_on = Sys.time(),
    seed = seed,
    target = target,
    event_level = tgt$event_level,
    candidates = candidates,
    selected = shortlist,
    final = final,
    split = idx$description,
    binning = binning_fit,
    screening = .ob_as_table(screening_tbl),
    screening_bins = .ob_as_table(screening_bins),
    correlation = pruning,
    control = control,
    engine = list(
      name = eng$name, requested = eng$requested, used = eng$used,
      additive = !is.null(coefs), args = engine_args
    ),
    model = fitted$model,
    coefficients = coefs,
    diagnostics = fitted$diagnostics,
    scaling = scaling,
    points = points,
    samples = scored,
    stability = stability,
    warnings = warn
  )
  class(out) <- "obwoe_scorecard"

  if (!is.null(file)) {
    out$file <- obwoe_report(out, file = file, control = control)
  }

  for (w in warn) warning(w, call. = FALSE)
  out
}


#' @keywords internal
.ob_check_names <- function(args, fun, label) {
  if (length(args) == 0L) {
    return(invisible(NULL))
  }
  allowed <- names(formals(fun))
  bad <- setdiff(names(args), allowed)
  if (length(bad) > 0L) {
    stop(sprintf(
      "Unknown %s argument(s): %s. Accepted: %s.",
      label, paste(sQuote(bad), collapse = ", "),
      paste(setdiff(allowed, c("obj", "data", "target", "feature", "...")),
        collapse = ", "
      )
    ))
  }
  invisible(NULL)
}


#' @title Internal: Choose the Training Rows
#' @keywords internal
.ob_train_index <- function(data, split, y) {
  n <- nrow(data)

  if (is.null(split)) {
    return(list(
      train = seq_len(n), test = integer(0),
      description = "none (all rows used for training)"
    ))
  }

  if (is.character(split) && length(split) == 1L && split %in% names(data)) {
    v <- data[[split]]
    lv <- sort(unique(as.character(v)))
    if (length(lv) < 2L) {
      stop(sprintf("Split column '%s' has a single value.", split))
    }
    tr <- which(as.character(v) == lv[1L])
    return(list(
      train = tr, test = setdiff(seq_len(n), tr),
      description = sprintf(
        "out-of-time on '%s': train = %s, holdout = %s",
        split, lv[1L], paste(lv[-1L], collapse = ", ")
      )
    ))
  }

  if (is.logical(split) && length(split) == n) {
    tr <- which(split)
    return(list(
      train = tr, test = setdiff(seq_len(n), tr),
      description = "explicit logical index"
    ))
  }

  if (is.numeric(split) && length(split) > 1L) {
    tr <- unique(as.integer(split))
    if (any(tr < 1L | tr > n)) stop("'split' holds row numbers outside the data.")
    return(list(
      train = tr, test = setdiff(seq_len(n), tr),
      description = "explicit row index"
    ))
  }

  if (!is.numeric(split) || length(split) != 1L || split <= 0 || split >= 1) {
    stop("'split' must be a proportion in (0, 1), a column name, a row index, or NULL.")
  }

  # Stratified so the event rate is the same on both sides of the split.
  tr <- unlist(lapply(split(seq_len(n), y), function(rows) {
    sample(rows, max(1L, floor(split * length(rows))))
  }), use.names = FALSE)

  list(
    train = sort(tr), test = setdiff(seq_len(n), tr),
    description = sprintf("stratified random, %.0f%% training", 100 * split)
  )
}


#' @title Internal: WoE Design Matrix
#' @keywords internal
.ob_woe_matrix <- function(data, binning, features, na_woe) {
  w <- obwoe_apply(data, binning, keep_original = FALSE, na_woe = na_woe)
  out <- w[, paste0(features, "_woe"), drop = FALSE]
  names(out) <- features
  out
}


#' @title Internal: Fit With Sign, Separation and Convergence Checks
#' @keywords internal
.ob_fit_checked <- function(eng, x, y, engine_args, control, add_warning) {
  features <- names(x)
  repeat {
    model <- eng$fit(x[, features, drop = FALSE], y, engine_args)
    coefs <- eng$coef(model)
    diag <- eng$diagnostics(model)

    if (is.null(coefs)) {
      return(list(
        model = model, coefficients = NULL, features = features,
        diagnostics = diag
      ))
    }

    if (isFALSE(diag$converged)) {
      stop(paste(
        "The model did not converge. This usually means a predictor separates",
        "the target perfectly; check the screening for a leaked field."
      ))
    }

    slopes <- coefs[setdiff(names(coefs), "(Intercept)")]

    if (any(abs(slopes) > control$max_abs_coef)) {
      big <- names(slopes)[abs(slopes) > control$max_abs_coef]
      add_warning(sprintf(
        paste(
          "Coefficient(s) above %g in absolute value (%s): this is the",
          "signature of separation, not of a strong predictor."
        ),
        control$max_abs_coef, paste(big, collapse = ", ")
      ))
    }

    # On WoE predictors every slope should be positive: a WoE of +1 is one more
    # log-odds of risk. A negative one means the variable is fighting the model.
    negative <- names(slopes)[slopes < 0]
    if (length(negative) == 0L || !control$drop_negative ||
      length(features) - length(negative) < 1L) {
      if (length(negative) > 0L) {
        add_warning(sprintf(
          paste(
            "Negative coefficient on the WoE of %s: the variable opposes the",
            "rest of the model, usually through residual correlation."
          ),
          paste(negative, collapse = ", ")
        ))
      }
      diag$sign_ok <- stats::setNames(slopes >= 0, names(slopes))
      return(list(
        model = model, coefficients = coefs, features = features,
        diagnostics = diag
      ))
    }

    # Drop the worst offender and refit, so the reported model is signable.
    worst <- negative[which.min(slopes[negative])]
    add_warning(sprintf(
      "Refitted without %s: its coefficient on WoE was negative (%.4f).",
      worst, slopes[[worst]]
    ))
    features <- setdiff(features, worst)
    if (length(features) == 0L) {
      stop("Every variable took a negative coefficient; no model can be fitted.")
    }
  }
}


#' @title Internal: Score One Sample
#' @keywords internal
.ob_score_sample <- function(df, name, binning, features, eng, model, scaling,
                             points, target, control, add_warning) {
  missing_cols <- setdiff(c(features, target), names(df))
  if (length(missing_cols) > 0L) {
    stop(sprintf(
      "Sample '%s' is missing column(s): %s",
      name, paste(missing_cols, collapse = ", ")
    ))
  }

  w <- .ob_woe_matrix(df, binning, features, control$na_woe)

  # Count values that landed in no fitted bin. WoE 0 is the population average,
  # which is a defensible fallback but is not "no effect" once an intercept is
  # in the model, so the size of that segment has to be visible.
  bins <- obwoe_apply(df, binning, keep_original = FALSE, na_woe = control$na_woe)
  unseen <- vapply(features, function(f) {
    sum(is.na(bins[[paste0(f, "_bin")]]))
  }, integer(1))
  if (any(unseen > 0L)) {
    hit <- names(unseen)[unseen > 0L]
    add_warning(sprintf(
      "Sample '%s': %s fell in no fitted bin and were scored at na_woe = %g (%s).",
      name, paste(sprintf("%d value(s) of %s", unseen[hit], hit), collapse = "; "),
      control$na_woe, "unseen category or unbinned missing"
    ))
  }

  link <- eng$link(model, w)
  score <- obwoe_score(link, scaling)

  card <- NULL
  drift <- NA_real_
  if (!is.null(points)) {
    card <- .ob_card_score(bins, points, features)
    finite <- is.finite(card - score)
    drift <- if (any(finite)) max(abs(card - score)[finite]) else NA_real_
  }

  list(
    name = name,
    n = nrow(df),
    y = .ob_resolve_target(df[[target]])$y,
    link = link,
    score = score,
    card_score = card,
    max_points_drift = drift,
    unseen = unseen,
    bins = bins[, paste0(features, "_bin"), drop = FALSE]
  )
}


#' @title Internal: Score From the Fixed Points Table
#'
#' @details
#' A value in no fitted bin scores that variable's \code{"points_na"} fallback
#' rather than \code{NA}, which is what the deployed SQL does and what a
#' production scorer has to do: a single unseen category cannot void the whole
#' application's score.
#'
#' @keywords internal
.ob_card_score <- function(bins, points, features,
                           na_points = attr(points, "points_na")) {
  cols <- lapply(features, function(f) {
    tbl <- points[points$variable == f, , drop = FALSE]
    p <- tbl$points[match(bins[[paste0(f, "_bin")]], tbl$bin)]
    fallback <- if (is.null(na_points)) NA_real_ else unname(na_points[[f]])
    p[is.na(p)] <- fallback
    p
  })
  rowSums(do.call(cbind, cols))
}


#' @keywords internal
.ob_score_breaks <- function(score, n_groups) {
  probs <- seq(0, 1, length.out = n_groups + 1L)
  probs <- probs[-c(1L, length(probs))]
  unique(c(-Inf, stats::quantile(score, probs = probs, na.rm = TRUE), Inf))
}


#' @keywords internal
.ob_score_gains <- function(score, y, breaks) {
  band <- cut(score, breaks = breaks, include.lowest = TRUE)
  # Keep `band` as the ordered factor cut() produced. Converting to
  # character here loses the level order, so .build_gains_table() falls
  # back to lexicographic order(df$bin) (ASCII '[' = 91 sorts after '('
  # = 40), which pushes the lowest-score bin "[-Inf,x]" to the last row
  # and silently corrupts the cumulative KS curve.
  df <- data.frame(band = band, target = y, stringsAsFactors = FALSE)
  g <- obwoe_gains(df,
    target = "target", feature = "band",
    use_column = "direct", sort_by = "bin"
  )
  g$table
}


#' @keywords internal
.ob_score_metrics <- function(score, y) {
  # Discrimination of the continuous score: rank-based AUC, no binning loss.
  n1 <- sum(y == 1L)
  n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) {
    return(list(n = length(y), events = n1, event_rate = NA_real_,
      auc = NA_real_, gini = NA_real_, ks = NA_real_))
  }
  r <- rank(score)
  # score is high-for-safe, so the event's AUC is the mirrored statistic
  auc <- 1 - (sum(r[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)

  o <- order(score)
  f1 <- cumsum(y[o] == 1L) / n1
  f0 <- cumsum(y[o] == 0L) / n0

  list(
    n = length(y), events = n1, event_rate = n1 / length(y),
    auc = auc, gini = 2 * auc - 1, ks = max(abs(f1 - f0))
  )
}


#' @title Internal: Stability Against the Training Sample
#'
#' @description
#' PSI for the score and for every variable in the model, comparing each
#' sample back to the training rows. Variables are compared on their bin
#' shares — the bins are what the model consumes, the comparison works for
#' numerical and categorical variables alike, and a shift that does not cross
#' a cut point is a shift the model never sees.
#'
#' @keywords internal
.ob_stability <- function(scored, binning, features, breaks) {
  base <- scored[[1L]]
  others <- names(scored)[-1L]
  if (length(others) == 0L) {
    return(NULL)
  }

  rows <- lapply(others, function(nm) {
    cmp <- scored[[nm]]

    s <- obwoe_psi(base$score, cmp$score, breaks = breaks)
    score_row <- data.frame(
      sample = nm, level = "SCORE", variable = "score",
      psi = s$psi, flag = s$flag, stringsAsFactors = FALSE
    )

    var_rows <- lapply(features, function(f) {
      col <- paste0(f, "_bin")
      lev <- as.character(binning$results[[f]]$bin)
      v <- obwoe_psi(base$bins[[col]], cmp$bins[[col]], levels = lev)
      data.frame(
        sample = nm, level = "VARIABLE", variable = f,
        psi = v$psi, flag = v$flag, stringsAsFactors = FALSE
      )
    })

    .ob_rbind(c(list(score_row), var_rows))
  })

  out <- as.data.frame(.ob_rbind(rows), stringsAsFactors = FALSE)
  out <- out[order(out$sample, out$level != "SCORE", -out$psi), , drop = FALSE]
  rownames(out) <- NULL
  .ob_as_table(out)
}


#' @method print obwoe_scorecard
#' @export
print.obwoe_scorecard <- function(x, ...) {
  cat("Scorecard\n")
  cat(strrep("=", 60), "\n")
  cat(sprintf("Target      : %s (event = %s)\n", x$target, x$event_level))
  cat(sprintf("Split       : %s\n", x$split))
  cat(sprintf(
    "Funnel      : %d candidates -> %d screened -> %d in model\n",
    length(x$candidates), length(x$selected), length(x$final)
  ))
  cat(sprintf(
    "Engine      : %s (%s)\n", x$engine$name,
    if (x$engine$additive) "additive, points available" else "not additive"
  ))
  cat(sprintf(
    "Scaling     : PDO %g, %g points at %g:1\n",
    x$scaling$pdo, x$scaling$score_ref, x$scaling$odds_ref
  ))

  cat("\nPerformance:\n")
  perf <- do.call(rbind, lapply(names(x$samples), function(nm) {
    m <- x$samples[[nm]]$metrics
    data.frame(
      sample = nm, n = m$n, events = m$events,
      ks = round(m$ks, 4), gini = round(m$gini, 4), auc = round(m$auc, 4),
      stringsAsFactors = FALSE
    )
  }))
  print(perf, row.names = FALSE)

  if (!is.null(x$points)) {
    d <- vapply(x$samples, function(s) s$max_points_drift, numeric(1))
    drift <- if (any(is.finite(d))) max(d[is.finite(d)]) else NA_real_
    cat(sprintf(
      "\nPoints      : %d rows; max card-vs-model drift %.2f (bound %.1f)\n",
      nrow(x$points), drift, length(x$final) / 2
    ))
  }

  if (!is.null(x$file)) cat(sprintf("Workbook    : %s\n", x$file))

  if (length(x$warnings) > 0L) {
    cat(sprintf("\n%d warning(s); see $warnings.\n", length(x$warnings)))
  }
  invisible(x)
}


#' @title Score New Data With a Fitted Scorecard
#'
#' @description
#' Applies a fitted scorecard to new data. Reads only the binning, the
#' coefficients and the scaling — never the raw fitted model — so a scorecard
#' saved with \code{saveRDS()} keeps scoring after the engine's package is gone.
#'
#' @param object An \code{"obwoe_scorecard"} object.
#' @param new_data A \code{data.frame} carrying the model's variables.
#' @param type One of \code{"score"} (default), \code{"card"} (the sum of the
#'   integer points, i.e. what the deployed table gives), \code{"link"} (log-odds
#'   of the event), \code{"prob"} or \code{"woe"}.
#' @param ... Ignored.
#'
#' @return A numeric vector, or a \code{data.frame} when \code{type = "woe"}.
#'
#' @seealso \code{\link{obwoe_scorecard}}
#'
#' @examples
#' \donttest{
#' german <- read.csv(
#'   gzfile(system.file("extdata", "germancredit.csv.gz",
#'     package = "OptimalBinningWoE"
#'   )),
#'   stringsAsFactors = FALSE
#' )
#' german$default <- 1L - german$credit_risk
#' german$credit_risk <- NULL
#'
#' sc <- obwoe_scorecard(german, target = "default", seed = 1)
#' summary(predict(sc, german))
#' }
#'
#' @method predict obwoe_scorecard
#' @export
predict.obwoe_scorecard <- function(object,
                                    new_data,
                                    type = c("score", "card", "link", "prob", "woe"),
                                    ...) {
  type <- match.arg(type)
  if (!is.data.frame(new_data)) stop("'new_data' must be a data.frame.")

  if (is.null(object$coefficients)) {
    stop(paste(
      "This scorecard was fitted with a non-additive engine, so it cannot be",
      "re-scored from the stored coefficients. Use the engine's own predict()",
      "on $model."
    ))
  }

  # [C-02/A-01] Read the na_woe the scorecard was actually built with
  # (stored in object$control as of 1.13.1), instead of silently hardcoding
  # 0. type = "score" and type = "card" both go through this same value, so
  # they now agree on unseen categories/missing values instead of each
  # falling back to a different default. Objects saved by an older package
  # version carry no $control and keep the historical 0 fallback.
  na_woe <- if (!is.null(object$control) && !is.null(object$control$na_woe)) {
    object$control$na_woe
  } else {
    0
  }
  w <- .ob_woe_matrix(new_data, object$binning, object$final, na_woe)
  if (identical(type, "woe")) {
    return(w)
  }

  if (identical(type, "card")) {
    if (is.null(object$points)) stop("This scorecard has no points table.")
    bins <- obwoe_apply(new_data, object$binning,
      keep_original = FALSE, na_woe = na_woe
    )
    return(.ob_card_score(bins, object$points, object$final))
  }

  cf <- object$coefficients
  link <- as.numeric(cf[["(Intercept)"]] +
    as.matrix(w[, object$final, drop = FALSE]) %*%
      cf[object$final])

  switch(type,
    "link" = link,
    "prob" = stats::plogis(link),
    "score" = obwoe_score(link, object$scaling)
  )
}
