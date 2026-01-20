#' @title Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation
#'
#' @description
#' Implements a \strong{hybrid binning algorithm} that initializes bins using
#' \strong{unsupervised statistical properties} (mean and standard deviation of
#' the feature) and refines them through \strong{supervised optimization} using
#' Weight of Evidence (WoE) and Information Value (IV).
#'
#' \strong{Important Clarification}: Despite "Unsupervised" in the name, this
#' method is \strong{predominantly supervised}. The unsupervised component is
#' limited to the initial bin creation step (~1\% of the algorithm). All subsequent
#' refinement (merge, monotonicity enforcement, bin count adjustment) uses the
#' target variable extensively.
#'
#' The statistical initialization via \eqn{\mu \pm k\sigma} provides a data-driven
#' starting point that may be advantageous for approximately normal distributions,
#' but offers no guarantees for skewed or multimodal data.
#'
#' @param feature Numeric vector of feature values. Missing values (NA) and infinite
#'   values are \strong{not permitted} and will trigger an error.
#' @param target Integer or numeric vector of binary target values (must contain
#'   only 0 and 1). Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be \eqn{\ge} \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Must be in (0, 1).
#' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
#'   Must be at least equal to \code{min_bins}.
#' @param convergence_threshold Convergence threshold for IV change (default: 1e-6).
#' @param max_iterations Maximum iterations for optimization (default: 1000).
#' @param laplace_smoothing Laplace smoothing parameter (default: 0.5). Must be
#'   non-negative.
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer bin identifiers (1-based).}
#'   \item{bin}{Character bin intervals \code{"[lower;upper)"}.}
#'   \item{woe}{Numeric WoE values (monotonic after enforcement).}
#'   \item{iv}{Numeric IV contributions per bin.}
#'   \item{count}{Integer total observations per bin.}
#'   \item{count_pos}{Integer positive class counts.}
#'   \item{count_neg}{Integer negative class counts.}
#'   \item{event_rate}{Numeric event rates per bin.}
#'   \item{cutpoints}{Numeric bin boundaries (excluding \eqn{\pm\infty}).}
#'   \item{total_iv}{Total Information Value.}
#'   \item{converged}{Logical convergence flag.}
#'   \item{iterations}{Integer iteration count.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' UBSD executes in six phases:
#'
#' \strong{Phase 1: Statistical Initialization (UNSUPERVISED)}
#'
#' Initial bin edges are created by combining two approaches:
#'
#' \enumerate{
#'   \item \strong{Standard deviation-based cutpoints}:
#'     \deqn{\{\mu - 2\sigma, \mu - \sigma, \mu, \mu + \sigma, \mu + 2\sigma\}}
#'     where \eqn{\mu} is the sample mean and \eqn{\sigma} is the sample standard
#'     deviation (with Bessel correction: \eqn{N-1} divisor).
#'
#'   \item \strong{Equal-width cutpoints}:
#'     \deqn{\left\{x_{\min} + i \times \frac{x_{\max} - x_{\min}}{\text{max\_n\_prebins}}\right\}_{i=1}^{\text{max\_n\_prebins}-1}}
#' }
#'
#' The union of these two sets is taken, sorted, and limited to \code{max_n_prebins}
#' edges (plus \eqn{-\infty} and \eqn{+\infty} boundaries).
#'
#' \strong{Rationale}: For approximately normal distributions, \eqn{\mu \pm k\sigma}
#' cutpoints align with natural quantiles:
#' \itemize{
#'   \item \eqn{\mu - 2\sigma} to \eqn{\mu + 2\sigma} captures ~95\% of data (68-95-99.7 rule)
#'   \item Equal-width ensures coverage of entire range
#' }
#'
#' \strong{Limitation}: For skewed distributions (e.g., log-normal), \eqn{\mu - 2\sigma}
#' may fall outside the data range, creating empty bins.
#'
#' \strong{Special Case}: If \eqn{\sigma < \epsilon} (feature is nearly constant),
#' fallback to pure equal-width binning.
#'
#' \strong{Phase 2: Observation Assignment}
#'
#' Each observation is assigned to a bin via linear search:
#' \deqn{\text{bin}(x_i) = \min\{j : x_i > \text{lower}_j \land x_i \le \text{upper}_j\}}
#'
#' Counts are accumulated: \code{count}, \code{count_pos}, \code{count_neg}.
#'
#' \strong{Phase 3: Rare Bin Merging (SUPERVISED)}
#'
#' Bins with \eqn{\text{count} < \text{bin\_cutoff} \times N} are merged with
#' adjacent bins. Merge direction is chosen to minimize IV loss:
#'
#' \deqn{\text{direction} = \arg\min_{d \in \{\text{left}, \text{right}\}} \left( \text{IV}_i + \text{IV}_{i+d} \right)}
#'
#' This is a \strong{supervised} step (uses IV computed from target).
#'
#' \strong{Phase 4: WoE/IV Calculation (SUPERVISED)}
#'
#' Weight of Evidence with Laplace smoothing:
#' \deqn{\text{WoE}_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} \bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)}
#'
#' Information Value:
#' \deqn{\text{IV}_i = \left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} - \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right) \times \text{WoE}_i}
#'
#' \strong{Phase 5: Monotonicity Enforcement (SUPERVISED)}
#'
#' Direction is auto-detected via majority vote:
#' \deqn{\text{increasing} = \begin{cases} \text{TRUE} & \text{if } \sum_i \mathbb{1}_{\{\text{WoE}_i > \text{WoE}_{i-1}\}} \ge \sum_i \mathbb{1}_{\{\text{WoE}_i < \text{WoE}_{i-1}\}} \\ \text{FALSE} & \text{otherwise} \end{cases}}
#'
#' Violations are resolved via PAVA (Pool Adjacent Violators Algorithm).
#'
#' \strong{Phase 6: Bin Count Adjustment (SUPERVISED)}
#'
#' If \eqn{k > \text{max\_bins}}, bins are merged to minimize IV loss:
#' \deqn{\text{merge\_idx} = \arg\min_{i=0}^{k-2} \left( \text{IV}_i + \text{IV}_{i+1} \right)}
#'
#' \strong{Convergence Criterion}:
#' \deqn{|\text{IV}_{\text{total}}^{(t)} - \text{IV}_{\text{total}}^{(t-1)}| < \text{convergence\_threshold}}
#'
#' \strong{Comparison with Related Methods}
#'
#' \tabular{llll}{
#'   \strong{Method} \tab \strong{Initialization} \tab \strong{Truly Unsupervised?} \tab \strong{Best For} \cr
#'   UBSD \tab \eqn{\mu \pm k\sigma} + equal-width \tab No (1 pct unsup) \tab Normal distributions \cr
#'   MOB/MRBLP \tab Equal-frequency \tab No (0 pct unsup) \tab General use \cr
#'   MDLP \tab Equal-frequency \tab No (0 pct unsup) \tab Information theory \cr
#'   Sketch \tab KLL Sketch quantiles \tab No (0 pct unsup) \tab Streaming data \cr
#' }
#'
#' \strong{When to Use UBSD}
#'
#' \itemize{
#'   \item \strong{Use UBSD}: If you have prior knowledge that the feature is
#'     approximately normally distributed and want bins aligned with standard
#'     deviations (e.g., for interpretability: "2 standard deviations below mean").
#'   \item \strong{Avoid UBSD}: For skewed distributions (use MDLP or MOB), for
#'     multimodal distributions (use LDB), or when you need provable optimality
#'     (use Sketch for quantile guarantees).
#'   \item \strong{Alternative}: For true unsupervised binning (no target), use
#'     \code{cut()} with \code{breaks = "Sturges"} or \code{"FD"} (Freedman-Diaconis).
#' }
#'
#' \strong{Computational Complexity}
#'
#' Identical to MOB/MRBLP: \eqn{O(N + k^2 \times \text{max\_iterations})}
#'
#' @references
#' \itemize{
#'   \item Sturges, H. A. (1926). "The Choice of a Class Interval". \emph{Journal
#'     of the American Statistical Association}, 21(153), 65-66.
#'   \item Scott, D. W. (1979). "On optimal and data-based histograms". \emph{Biometrika},
#'     66(3), 605-610.
#'   \item Freedman, D., & Diaconis, P. (1981). "On the histogram as a density estimator:
#'     L2 theory". \emph{Zeitschrift fuer Wahrscheinlichkeitstheorie}, 57(4), 453-476.
#'   \item Thomas, L. C. (2009). \emph{Consumer Credit Models: Pricing, Profit, and
#'     Portfolios}. Oxford University Press.
#'   \item Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm in
#'     Credit Scoring". \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards}. Wiley.
#' }
#'
#' @examples
#' \donttest{
#' # Simulate normally distributed credit scores
#' set.seed(123)
#' n <- 5000
#'
#' # Feature: Normally distributed FICO scores
#' feature <- rnorm(n, mean = 680, sd = 60)
#'
#' # Target: Logistic relationship with score
#' prob_default <- 1 / (1 + exp((feature - 680) / 30))
#' target <- rbinom(n, 1, prob_default)
#'
#' # Apply UBSD
#' result <- ob_numerical_ubsd(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Compare with MDLP (should be similar for normal data)
#' result_mdlp <- ob_numerical_mdlp(feature, target)
#'
#' data.frame(
#'   Method = c("UBSD", "MDLP"),
#'   N_Bins = c(length(result$woe), length(result_mdlp$woe)),
#'   Total_IV = c(result$total_iv, result_mdlp$total_iv)
#' )
#' }
#'
#' @author
#' Lopes, J. E.
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}} for information-theoretic binning,
#' \code{\link{ob_numerical_mob}} for pure supervised binning,
#' \code{\link{cut}} for true unsupervised binning.
#'
#' @export
ob_numerical_ubsd <- function(feature,
                              target,
                              min_bins = 3,
                              max_bins = 5,
                              bin_cutoff = 0.05,
                              max_n_prebins = 20,
                              convergence_threshold = 1e-6,
                              max_iterations = 1000,
                              laplace_smoothing = 0.5) {
  if (!is.numeric(feature)) {
    stop("Feature must be a numeric vector.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("Target must be an integer or numeric vector.")
  }

  if (length(feature) != length(target)) {
    stop("Feature and target must have the same length.")
  }

  feature <- as.numeric(feature)
  target <- as.integer(target)

  unique_target <- unique(target[!is.na(target)])
  if (!all(unique_target %in% c(0L, 1L)) || length(unique_target) != 2L) {
    stop("Target must contain exactly two classes: 0 and 1.")
  }

  if (min_bins < 2L) {
    stop("min_bins must be at least 2.")
  }

  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.")
  }

  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be in the range (0, 1).")
  }

  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be at least equal to min_bins.")
  }

  if (convergence_threshold <= 0) {
    stop("convergence_threshold must be positive.")
  }

  if (max_iterations < 1L) {
    stop("max_iterations must be at least 1.")
  }

  if (laplace_smoothing < 0) {
    stop("laplace_smoothing must be non-negative.")
  }

  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_ubsd",
    target,
    feature,
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    as.numeric(laplace_smoothing),
    PACKAGE = "OptimalBinningWoE"
  )

  class(result) <- c("OptimalBinningUBSD", "OptimalBinning", "list")

  return(result)
}
