#' Optimal Binning for Categorical Variables using Order-Restricted Bayesian Evidence
#'
#' Groups the categories of a feature so that the grouping and the number of
#' groups maximize the Bayesian evidence of a constant event rate per group.
#' A category keeps a group of its own only when the data support it. Reports
#' posterior-mean WoE with credible intervals.
#'
#' @param feature A character vector or factor of categories. Missing values
#'   form the category \code{"NA"}.
#' @param target A binary vector (0/1 or logical) with the same length as
#'   \code{feature}. Missing values are an error.
#' @param min_bins Minimum number of bins (default: 2). It is honored whenever
#'   a grouping with that many bins satisfies \code{bin_cutoff} and has
#'   strictly increasing WoE; otherwise the largest feasible number of bins is
#'   returned. With \code{min_bins = 1} a single bin is returned when the
#'   evidence does not support any split.
#' @param max_bins Maximum number of bins (default: 10). The number of bins is
#'   selected by the evidence between \code{min_bins} and \code{max_bins}.
#' @param bin_cutoff Minimum fraction of the rows in a bin, in \eqn{[0, 1)}
#'   (default: 0.05). It is a constraint of the optimization. With
#'   \code{bin_cutoff = 0} rare categories are merged only by the evidence.
#' @param max_n_prebins Maximum number of atoms, the candidate units the bins
#'   are built from (default: 200). A feature with at most this many
#'   categories is searched over every boundary between categories in the
#'   order of their event rates. Above it, categories adjacent in that order
#'   are grouped into atoms of about equal frequency. Values above 1000 are
#'   treated as 1000, and the limit is lowered further when \code{max_bins} is
#'   large, to bound memory.
#' @param bin_separator String placed between the categories of a merged bin
#'   in its label (default: \code{"\%;\%"}). A warning is issued when it
#'   occurs inside a category name.
#' @param prior_strength Weight of the prior, in pseudo-events of the minority
#'   class (default: 1). Larger values shrink rare categories more strongly
#'   toward the overall event rate.
#' @param credible_level Probability content of the WoE credible intervals, in
#'   \eqn{(0, 1)} (default: 0.95).
#'
#' @return A list with one element per bin in the vectors below, in increasing
#'   order of WoE:
#' \describe{
#'   \item{id}{Bin identifiers (1-based).}
#'   \item{bin}{Bin labels: the categories joined by \code{bin_separator}.}
#'   \item{woe}{Posterior-mean Weight of Evidence.}
#'   \item{iv}{Information Value contribution,
#'     \eqn{(p_i - q_i) \, \mathrm{WoE}_i}, with \eqn{p_i} and \eqn{q_i} the
#'     observed shares of events and non-events.}
#'   \item{count, count_pos, count_neg}{Rows, events and non-events.}
#'   \item{event_rate}{Observed event rate.}
#'   \item{woe_se}{Posterior standard deviation of the WoE.}
#'   \item{woe_lower, woe_upper}{Limits of the equal-tailed credible interval
#'     of the WoE.}
#' }
#' and the following:
#' \describe{
#'   \item{total_iv}{Sum of \code{iv}.}
#'   \item{total_iv_corrected}{\code{total_iv} minus the first-order sampling
#'     bias of a plug-in IV, floored at zero.}
#'   \item{log_evidence}{Log evidence of the selected grouping.}
#'   \item{log_evidence_by_bins}{Log evidence of the best grouping with
#'     \eqn{1, 2, \ldots} bins; \code{NA} where no grouping is feasible.}
#'   \item{order_probability}{Posterior probability that the event rates of
#'     the bins are in the reported order.}
#'   \item{converged}{Always \code{TRUE}.}
#'   \item{iterations}{Number of bin counts evaluated.}
#' }
#'
#' @details
#' \strong{Model.} The event rates of the groups are independent
#' \eqn{\mathrm{Beta}(a, b)} with mean equal to the overall event rate
#' \eqn{\pi_0} and \eqn{\min(a, b)} equal to \code{prior_strength}. A group
#' with \eqn{n_i} rows and \eqn{e_i} events has log evidence
#' \deqn{f(n_i, e_i) = \log \frac{B(a + e_i,\; b + n_i - e_i)}{B(a, b)} .}
#' With \eqn{L} categories, a grouping into \eqn{K} groups has log prior
#' \eqn{-\log L - L \log K}: the number of groups is uniform on
#' \eqn{1, \ldots, L} and each category is assigned to one of the \eqn{K}
#' groups with equal probability. Every additional group therefore costs each
#' category, and the cost grows with the number of categories.
#'
#' \strong{Search.} Categories are ordered by their posterior-mean event rate
#' \eqn{(a + e)/(a + b + n)}, which shrinks rare categories toward
#' \eqn{\pi_0} before they are ordered. Ties are broken by count and then by
#' name, so the result does not depend on the row order. The engine then
#' finds, by dynamic programming, the grouping into bins contiguous in that
#' order that maximizes \eqn{\sum_i f(n_i, e_i)} plus the log prior, subject
#' to \code{bin_cutoff}, the limits on the number of bins and strictly
#' increasing posterior WoE. The search is exact among groupings contiguous in
#' that order; groupings that are not contiguous are not examined.
#'
#' \strong{Rare categories.} Merging two groups changes the evidence by
#' \eqn{f(n_g + n_h, e_g + e_h) - f(n_g, e_g) - f(n_h, e_h)} plus the change
#' in the log prior, which favors fewer groups. A category with few rows
#' rarely carries enough evidence to pay for a group of its own, so with
#' \code{bin_cutoff = 0} it is merged by the evidence, not by a frequency
#' threshold.
#'
#' \strong{WoE.} As in \code{\link{ob_numerical_orbe}}: with
#' \eqn{A_i = a + e_i} and \eqn{B_i = b + n_i - e_i},
#' \deqn{\mathrm{WoE}_i = \psi(A_i) - \psi(B_i) - \log\frac{\pi_0}{1 - \pi_0},}
#' with posterior variance \eqn{\psi'(A_i) + \psi'(B_i)}. The credible
#' intervals are conditional on the selected bins.
#'
#' @references
#' \enc{Boullé}{Boulle}, M. (2005). A Bayes optimal approach for partitioning the values of
#' categorical attributes. \emph{Journal of Machine Learning Research}, 6,
#' 1431-1452.
#'
#' @seealso \code{\link{ob_numerical_orbe}} for numerical features and
#'   \code{\link{ob_categorical_dp}} for dynamic programming on the
#'   Information Value.
#'
#' @examples
#' set.seed(123)
#' levels <- c(paste0("region_", 1:8), "rare_a", "rare_b")
#' prob <- c(rep(0.1, 8), 0.01, 0.01)
#' feature <- sample(levels, 3000, replace = TRUE, prob = prob)
#' rate <- c(0.05, 0.06, 0.10, 0.11, 0.20, 0.21, 0.35, 0.36, 0.15, 0.15)
#' target <- rbinom(3000, 1, rate[match(feature, levels)])
#'
#' # bin_cutoff = 0 leaves the fate of the rare categories to the evidence
#' fit <- ob_categorical_orbe(feature, target, bin_cutoff = 0)
#'
#' data.frame(
#'   bin = fit$bin, count = fit$count,
#'   woe = round(fit$woe, 3),
#'   lower = round(fit$woe_lower, 3),
#'   upper = round(fit$woe_upper, 3)
#' )
#'
#' @export
ob_categorical_orbe <- function(feature, target, min_bins = 2, max_bins = 10,
                                bin_cutoff = 0.05, max_n_prebins = 200,
                                bin_separator = "%;%", prior_strength = 1,
                                credible_level = 0.95) {
  # Input preprocessing
  if (!is.character(feature)) {
    feature <- as.character(feature)
  }
  feature[is.na(feature)] <- "NA"
  target <- .ob_integer_target(target)

  .Call("_OptimalBinningWoE_optimal_binning_categorical_orbe",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.character(bin_separator),
    as.numeric(prior_strength),
    as.numeric(credible_level),
    PACKAGE = "OptimalBinningWoE"
  )
}
