#' @title Optimal Binning for Numerical Features using Monotonic Risk Binning with Likelihood Ratio Pre-binning
#'
#' @description
#' Implements a greedy binning algorithm with monotonicity enforcement and
#' \strong{majority-vote direction detection}. \emph{Important Note}: Despite the
#' "Likelihood Ratio Pre-binning" designation in the name, the current implementation
#' uses \strong{equal-frequency pre-binning} without likelihood ratio statistics.
#' The algorithm is functionally a variant of Monotonic Optimal Binning (MOB) with
#' minor differences in merge strategies.
#'
#' This method is suitable for credit scoring applications requiring monotonic WoE
#' patterns, but users should be aware that it does not employ the statistical rigor
#' implied by "Likelihood Ratio" in the name.
#'
#' @param feature Numeric vector of feature values to be binned. Missing values (NA)
#'   and infinite values are \strong{not permitted} and will trigger an error (unlike
#'   other binning methods that issue warnings).
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins to generate (default: 3). Must be at least 1.
#'   Acts as a hard constraint during monotonicity enforcement.
#' @param max_bins Maximum number of bins to generate (default: 5). Must be greater
#'   than or equal to \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations required in each bin
#'   (default: 0.05). Bins with frequency below this threshold are merged. Must be
#'   in the range (0, 1).
#' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
#'   Must be at least equal to \code{min_bins}.
#' @param convergence_threshold Convergence threshold (default: 1e-6). Currently used
#'   to check if WoE range is below threshold; primary stopping criterion is
#'   \code{max_iterations}.
#' @param max_iterations Maximum number of iterations for bin merging and monotonicity
#'   enforcement (default: 1000). Prevents infinite loops.
#' @param laplace_smoothing Laplace smoothing parameter for WoE calculation (default: 0.5).
#'   Must be non-negative.
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer vector of bin identifiers (1-based indexing).}
#'   \item{bin}{Character vector of bin intervals in the format \code{"[lower;upper)"}.}
#'   \item{woe}{Numeric vector of Weight of Evidence values. Guaranteed to be monotonic.}
#'   \item{iv}{Numeric vector of Information Value contributions per bin.}
#'   \item{count}{Integer vector of total observations per bin.}
#'   \item{count_pos}{Integer vector of positive class counts per bin.}
#'   \item{count_neg}{Integer vector of negative class counts per bin.}
#'   \item{event_rate}{Numeric vector of event rates per bin.}
#'   \item{cutpoints}{Numeric vector of bin boundaries (excluding -Inf and +Inf).}
#'   \item{total_iv}{Total Information Value (sum of bin IVs).}
#'   \item{converged}{Logical flag indicating convergence within \code{max_iterations}.}
#'   \item{iterations}{Integer count of iterations performed.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' The MRBLP algorithm executes in five phases:
#'
#' \strong{Phase 1: Equal-Frequency Pre-binning}
#'
#' Initial bins are created by dividing sorted data into approximately equal-sized
#' groups:
#'
#' \deqn{n_{\text{bin}} = \max\left(1, \left\lfloor \frac{N}{\text{max\_n\_prebins}} \right\rfloor\right)}
#'
#' \strong{Note}: Despite "Likelihood Ratio Pre-binning" in the name, no likelihood
#' ratio statistics are computed. A true likelihood ratio approach would compute:
#'
#' \deqn{\text{LR}(c) = \prod_{x \le c} \frac{P(x|y=1)}{P(x|y=0)} \times \prod_{x > c} \frac{P(x|y=1)}{P(x|y=0)}}
#'
#' and select cutpoints \eqn{c} that maximize \eqn{|\log \text{LR}(c)|}. This is
#' \strong{not implemented} in the current version.
#'
#' \strong{Phase 2: Rare Bin Merging}
#'
#' Bins with total count below \code{bin_cutoff} \eqn{\times N} are merged. The
#' merge direction (left or right) is chosen to minimize IV loss:
#'
#' \deqn{\text{direction} = \arg\min_{d \in \{\text{left}, \text{right}\}} \left( \text{IV}_i + \text{IV}_{i+d} - \text{IV}_{\text{merged}} \right)}
#'
#' \strong{Phase 3: Initial WoE/IV Calculation}
#'
#' Weight of Evidence for bin \eqn{i}:
#'
#' \deqn{\text{WoE}_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} \bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)}
#'
#' where \eqn{\alpha = \text{laplace\_smoothing}} and \eqn{k} is the number of bins.
#'
#' \strong{Phase 4: Monotonicity Enforcement}
#'
#' The algorithm determines the desired monotonicity direction via \strong{majority vote}:
#'
#' \deqn{\text{increasing} = \begin{cases} \text{TRUE} & \text{if } \#\{\text{WoE}_i > \text{WoE}_{i-1}\} \ge \#\{\text{WoE}_i < \text{WoE}_{i-1}\} \\ \text{FALSE} & \text{otherwise} \end{cases}}
#'
#' This differs from:
#' \itemize{
#'   \item \strong{MOB}: Uses first two bins only (\code{WoE[1] >= WoE[0]})
#'   \item \strong{MBLP}: Uses Pearson correlation between bin indices and WoE
#' }
#'
#' Violations are detected as:
#' \deqn{\text{violation} = \begin{cases} \text{WoE}_i < \text{WoE}_{i-1} & \text{if increasing} \\ \text{WoE}_i > \text{WoE}_{i-1} & \text{if decreasing} \end{cases}}
#'
#' Violating bins are merged iteratively until monotonicity is achieved or
#' \code{min_bins} is reached.
#'
#' \strong{Phase 5: Bin Count Reduction}
#'
#' If the number of bins exceeds \code{max_bins}, the algorithm merges bins with
#' the \strong{smallest absolute IV difference}:
#'
#' \deqn{\text{merge\_idx} = \arg\min_{i=0}^{k-2} |\text{IV}_i - \text{IV}_{i+1}|}
#'
#' \strong{Critique}: This criterion assumes bins with similar IVs are redundant,
#' which is not theoretically justified. A more rigorous approach (used in MBLP)
#' minimizes IV loss \strong{after merge}:
#'
#' \deqn{\Delta \text{IV} = \text{IV}_i + \text{IV}_{i+1} - \text{IV}_{\text{merged}}}
#'
#' \strong{Theoretical Foundations}
#'
#' \itemize{
#'   \item \strong{Monotonicity Enforcement}: Based on Zeng (2014), ensuring stability
#'     under data distribution shifts.
#'   \item \strong{Likelihood Ratio (Theoretical)}: Neyman-Pearson lemma establishes
#'     likelihood ratio as the optimal test statistic for hypothesis testing. For
#'     binning, cutpoints maximizing LR would theoretically yield optimal class
#'     separation. \strong{However, this is not implemented}.
#'   \item \strong{Practical Equivalence}: The algorithm is functionally equivalent to
#'     MOB with minor differences in direction detection and merge strategies.
#' }
#'
#' \strong{Comparison with Related Methods}
#'
#' \tabular{llll}{
#'   \strong{Method} \tab \strong{Pre-binning} \tab \strong{Direction Detection} \tab \strong{Merge Criterion} \cr
#'   MRBLP \tab Equal-frequency \tab Majority vote \tab Min IV difference \cr
#'   MOB \tab Equal-frequency \tab First two bins \tab Min IV loss \cr
#'   MBLP \tab Quantile-based \tab Pearson correlation \tab Min IV loss \cr
#'   MDLP \tab Equal-frequency \tab N/A (optional) \tab MDL cost \cr
#' }
#'
#' \strong{Computational Complexity}
#'
#' Identical to MOB: \eqn{O(N \log N + k^2 \times \text{max\_iterations})}
#'
#' \strong{When to Use MRBLP vs Alternatives}
#'
#' \itemize{
#'   \item \strong{Use MRBLP}: If you specifically need majority-vote direction detection
#'     and can tolerate the non-standard merge criterion.
#'   \item \strong{Use MOB}: For simplicity and slightly faster direction detection.
#'   \item \strong{Use MBLP}: For more robust direction detection via correlation.
#'   \item \strong{Use MDLP}: For information-theoretic optimality without mandatory
#'     monotonicity.
#' }
#'
#' @references
#' \itemize{
#'   \item Neyman, J., & Pearson, E. S. (1933). "On the Problem of the Most Efficient
#'     Tests of Statistical Hypotheses". \emph{Philosophical Transactions of the Royal
#'     Society A}, 231(694-706), 289-337. [Theoretical foundation for likelihood ratio,
#'     not implemented in code]
#'   \item Mironchyk, P., & Tchistiakov, V. (2017). "Monotone optimal binning algorithm
#'     for credit risk modeling". \emph{Frontiers in Applied Mathematics and Statistics}, 3, 2.
#'   \item Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm in
#'     Credit Scoring". \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards: Developing and Implementing
#'     Intelligent Credit Scoring}. Wiley.
#'   \item Anderson, R. (2007). \emph{The Credit Scoring Toolkit: Theory and Practice
#'     for Retail Credit Risk Management and Decision Automation}. Oxford University Press.
#'   \item Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). \emph{Applied
#'     Logistic Regression} (3rd ed.). Wiley.
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate credit scoring data
#' set.seed(2024)
#' n <- 10000
#' feature <- c(
#'   rnorm(4000, mean = 620, sd = 50),
#'   rnorm(4000, mean = 690, sd = 45),
#'   rnorm(2000, mean = 740, sd = 35)
#' )
#' target <- c(
#'   rbinom(4000, 1, 0.20),
#'   rbinom(4000, 1, 0.10),
#'   rbinom(2000, 1, 0.04)
#' )
#'
#' # Apply MRBLP
#' result <- ob_numerical_mrblp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Verify monotonicity
#' print(result$woe)
#' stopifnot(all(diff(result$woe) >= -1e-10))
#'
#' # Compare with MOB (should be very similar)
#' result_mob <- ob_numerical_mob(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' # Compare results
#' data.frame(
#'   Method = c("MRBLP", "MOB"),
#'   N_Bins = c(length(result$woe), length(result_mob$woe)),
#'   Total_IV = c(result$total_iv, result_mob$total_iv),
#'   Iterations = c(result$iterations, result_mob$iterations)
#' )
#' }
#'
#' @author
#' Lopes, J. E.
#'
#' @seealso
#' \code{\link{ob_numerical_mob}} for the base monotonic binning algorithm,
#' \code{\link{ob_numerical_mblp}} for correlation-based direction detection,
#' \code{\link{ob_numerical_mdlp}} for information-theoretic binning.
#'
#' @export
ob_numerical_mrblp <- function(feature,
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

  if (min_bins < 1L) {
    stop("min_bins must be at least 1.")
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
    "_OptimalBinningWoE_optimal_binning_numerical_mrblp",
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

  class(result) <- c("OptimalBinningMRBLP", "OptimalBinning", "list")

  return(result)
}
