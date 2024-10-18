utils::globalVariables(c(
  ":=", "target", "count_pos", "count_neg", "woe", "original_index", "bin", "..va",
  "min_bins", "max_bins", "total_iv", "..", "%||%", "va", "::", ":::", "%||%", ".",
  "..features", "iv", "feature", "is_monotonic", "total_zero_pos", "total_zero_neg",
  "total_bins", "head", "numeric_features", "categorical_features"
))

#' Pipe operator
#'
#' See \code{magrittr::\link[magrittr:pipe]{\%>\%}} for details.
#'
#' @name %>%
#' @rdname pipe
#' @keywords internal
#' @export
#' @importFrom magrittr %>%
#' @usage lhs \%>\% rhs
#' @param lhs A value or the magrittr placeholder.
#' @param rhs A function call using the magrittr semantics.
#' @return The result of calling `rhs(lhs)`.
NULL


## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL

## usethis namespace: start
#' @useDynLib OptimalBinningWoE, .registration = TRUE
## usethis namespace: end
NULL

## usethis namespace: start
#' @importFrom stats as.formula binomial model.frame model.response
#' @importFrom stats plogis predict printCoefmat terms update update.formula
## usethis namespace: end
NULL
