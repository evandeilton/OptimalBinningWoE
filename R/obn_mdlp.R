#' @title Optimal Binning for Numerical Features using Minimum Description Length Principle
#'
#' @description
#' Implements the Minimum Description Length Principle (MDLP) for supervised
#' discretization of numerical features. MDLP balances model complexity (number
#' of bins) and data fit (information gain) through a rigorous information-theoretic
#' framework, automatically determining the optimal number of bins without arbitrary
#' thresholds.
#'
#' Unlike heuristic methods, MDLP provides a \strong{theoretically grounded stopping
#' criterion} based on the trade-off between encoding the binning structure and
#' encoding the data given that structure. This makes it particularly robust against
#' overfitting in noisy datasets.
#'
#' @param feature Numeric vector of feature values to be binned. Missing values (NA)
#'   are automatically removed during preprocessing. Infinite values trigger a warning
#'   but are handled internally.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins to generate (default: 3). Must be at least 1.
#'   If the number of unique feature values is less than \code{min_bins}, the algorithm
#'   adjusts automatically.
#' @param max_bins Maximum number of bins to generate (default: 5). Must be greater
#'   than or equal to \code{min_bins}. Acts as a hard constraint after MDLP optimization.
#' @param bin_cutoff Minimum fraction of total observations required in each bin
#'   (default: 0.05). Bins with frequency below this threshold are merged with adjacent
#'   bins to ensure statistical reliability. Must be in the range (0, 1).
#' @param max_n_prebins Maximum number of pre-bins before MDLP optimization (default: 20).
#'   Higher values allow finer granularity but increase computational cost. Must be
#'   at least 2.
#' @param convergence_threshold Convergence threshold for iterative optimization
#'   (default: 1e-6). Currently used internally for future extensions; MDLP convergence
#'   is primarily determined by the MDL cost function.
#' @param max_iterations Maximum number of iterations for bin merging operations
#'   (default: 1000). Prevents infinite loops in pathological cases. A warning is
#'   issued if this limit is reached.
#' @param laplace_smoothing Laplace smoothing parameter for WoE calculation (default: 0.5).
#'   Prevents division by zero and stabilizes WoE estimates in bins with zero counts
#'   for one class. Must be non-negative. Higher values increase regularization but
#'   may dilute signal in small bins.
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer vector of bin identifiers (1-based indexing).}
#'   \item{bin}{Character vector of bin intervals in the format \code{"[lower;upper)"}.
#'     The first bin starts with \code{-Inf} and the last bin ends with \code{+Inf}.}
#'   \item{woe}{Numeric vector of Weight of Evidence values for each bin, computed with
#'     Laplace smoothing.}
#'   \item{iv}{Numeric vector of Information Value contributions for each bin.}
#'   \item{count}{Integer vector of total observations in each bin.}
#'   \item{count_pos}{Integer vector of positive class (target = 1) counts per bin.}
#'   \item{count_neg}{Integer vector of negative class (target = 0) counts per bin.}
#'   \item{cutpoints}{Numeric vector of cutpoints defining bin boundaries (excluding
#'     -Inf and +Inf). These are the upper bounds of bins 1 to k-1.}
#'   \item{total_iv}{Numeric scalar representing the total Information Value (sum of
#'     all bin IVs).}
#'   \item{converged}{Logical flag indicating whether the algorithm converged. Set to
#'     \code{FALSE} if \code{max_iterations} was reached during any merging phase.}
#'   \item{iterations}{Integer count of iterations performed across all optimization
#'     phases (MDL merging, rare bin merging, monotonicity enforcement).}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' The MDLP algorithm executes in five sequential phases:
#'
#' \strong{Phase 1: Data Preparation and Validation}
#'
#' Input data is validated for:
#' \itemize{
#'   \item Binary target (only 0 and 1 values)
#'   \item Parameter consistency (\code{min_bins <= max_bins}, valid ranges)
#'   \item Missing value detection (NaN/Inf are filtered out with a warning)
#' }
#'
#' Feature-target pairs are sorted by feature value in ascending order, enabling
#' efficient bin assignment via linear scan.
#'
#' \strong{Phase 2: Equal-Frequency Pre-binning}
#'
#' Initial bins are created by dividing the sorted data into approximately equal-sized
#' groups:
#'
#' \deqn{n_{\text{records/bin}} = \max\left(1, \left\lfloor \frac{N}{\text{max\_n\_prebins}} \right\rfloor\right)}
#'
#' This ensures each pre-bin has sufficient observations for stable entropy estimation.
#' Bin boundaries are set to feature values at split points, with first and last
#' boundaries at \eqn{-\infty} and \eqn{+\infty}.
#'
#' For each bin \eqn{i}, Shannon entropy is computed:
#'
#' \deqn{H(S_i) = -p_i \log_2(p_i) - q_i \log_2(q_i)}
#'
#' where \eqn{p_i = n_i^{+} / n_i} (proportion of positives) and \eqn{q_i = 1 - p_i}.
#' Pure bins (\eqn{p_i = 0} or \eqn{p_i = 1}) have \eqn{H(S_i) = 0}.
#'
#' \strong{Performance Note}: Entropy calculation uses a precomputed lookup table
#' for bin counts 0-100, achieving 30-50\% speedup compared to runtime computation.
#'
#' \strong{Phase 3: MDL-Based Greedy Merging}
#'
#' The core optimization minimizes the Minimum Description Length, defined as:
#'
#' \deqn{\text{MDL}(k) = L_{\text{model}}(k) + L_{\text{data}}(k)}
#'
#' where:
#'
#' \itemize{
#'   \item \strong{Model Cost}: \eqn{L_{\text{model}}(k) = \log_2(k - 1)}
#'
#'     Encodes the number of bins. Increases logarithmically with bin count,
#'     penalizing complex models.
#'
#'   \item \strong{Data Cost}: \eqn{L_{\text{data}}(k) = N \cdot H(S_{\text{total}}) - \sum_{i=1}^{k} n_i \cdot H(S_i)}
#'
#'     Measures unexplained uncertainty after binning. Lower values indicate better
#'     class separation.
#' }
#'
#' The algorithm iteratively evaluates all \eqn{k-1} adjacent bin pairs, computing
#' \eqn{\text{MDL}(k-1)} for each potential merge. The pair minimizing MDL cost is
#' merged, continuing until:
#'
#' \enumerate{
#'   \item \eqn{k = \text{min\_bins}}, or
#'   \item No merge reduces MDL cost (local optimum), or
#'   \item \code{max_iterations} is reached
#' }
#'
#' \strong{Theoretical Guarantee} (Fayyad & Irani, 1993): The MDL criterion
#' provides a **consistent estimator** of the true discretization complexity under
#' mild regularity conditions, unlike ad-hoc stopping rules.
#'
#' \strong{Phase 4: Rare Bin Handling}
#'
#' Bins with frequency \eqn{n_i / N < \text{bin\_cutoff}} are merged with adjacent
#' bins. The merge direction (left or right) is chosen by minimizing post-merge entropy:
#'
#' \deqn{\text{direction} = \arg\min_{d \in \{\text{left}, \text{right}\}} H(S_i \cup S_{i+d})}
#'
#' This preserves class homogeneity while ensuring statistical reliability.
#'
#' \strong{Phase 5: Monotonicity Enforcement (Optional)}
#'
#' If WoE values violate monotonicity (\eqn{\text{WoE}_i < \text{WoE}_{i-1}}), bins
#' are iteratively merged until:
#'
#' \deqn{\text{WoE}_1 \le \text{WoE}_2 \le \cdots \le \text{WoE}_k}
#'
#' Merge decisions prioritize preserving Information Value:
#'
#' \deqn{\Delta \text{IV} = \text{IV}_i + \text{IV}_{i+1} - \text{IV}_{\text{merged}}}
#'
#' Merges proceed only if \eqn{\text{IV}_{\text{merged}} \ge 0.5 \times (\text{IV}_i + \text{IV}_{i+1})}.
#'
#' \strong{Weight of Evidence Computation}
#'
#' WoE for bin \eqn{i} includes Laplace smoothing to handle zero counts:
#'
#' \deqn{\text{WoE}_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} \bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)}
#'
#' where \eqn{\alpha = \text{laplace\_smoothing}} and \eqn{k} is the number of bins.
#'
#' \strong{Edge cases}:
#' \itemize{
#'   \item If \eqn{n_i^{+} + \alpha = n_i^{-} + \alpha = 0}: \eqn{\text{WoE}_i = 0}
#'   \item If \eqn{n_i^{+} + \alpha = 0}: \eqn{\text{WoE}_i = -20} (capped)
#'   \item If \eqn{n_i^{-} + \alpha = 0}: \eqn{\text{WoE}_i = +20} (capped)
#' }
#'
#' Information Value is computed as:
#'
#' \deqn{\text{IV}_i = \left(\frac{n_i^{+}}{n^{+}} - \frac{n_i^{-}}{n^{-}}\right) \times \text{WoE}_i}
#'
#' \strong{Comparison with Other Methods}
#'
#' \tabular{lll}{
#'   \strong{Method} \tab \strong{Stopping Criterion} \tab \strong{Optimality} \cr
#'   MDLP \tab Information-theoretic (MDL cost) \tab Local optimum with theoretical guarantees \cr
#'   LDB \tab Heuristic (density minima) \tab No formal optimality \cr
#'   MBLP \tab Heuristic (IV loss threshold) \tab Greedy approximation \cr
#'   ChiMerge \tab Statistical (\eqn{\chi^2} test) \tab Dependent on significance level \cr
#' }
#'
#' \strong{Computational Complexity}
#'
#' \itemize{
#'   \item Sorting: \eqn{O(N \log N)}
#'   \item Pre-binning: \eqn{O(N)}
#'   \item MDL optimization: \eqn{O(k^3 \times I)} where \eqn{I} is the number of
#'     merge iterations (typically \eqn{I \approx k})
#'   \item Total: \eqn{O(N \log N + k^3 \times I)}
#' }
#'
#' For typical credit scoring datasets (\eqn{N \sim 10^5}, \eqn{k \sim 5}), runtime
#' is dominated by sorting.
#'
#' @references
#' \itemize{
#'   \item Fayyad, U. M., & Irani, K. B. (1993). "Multi-Interval Discretization of
#'     Continuous-Valued Attributes for Classification Learning". \emph{Proceedings
#'     of the 13th International Joint Conference on Artificial Intelligence (IJCAI)},
#'     pp. 1022-1027.
#'   \item Rissanen, J. (1978). "Modeling by shortest data description". \emph{Automatica},
#'     14(5), 465-471.
#'   \item Shannon, C. E. (1948). "A Mathematical Theory of Communication". \emph{Bell
#'     System Technical Journal}, 27(3), 379-423.
#'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). "Supervised and Unsupervised
#'     Discretization of Continuous Features". \emph{Proceedings of the 12th International
#'     Conference on Machine Learning (ICML)}, pp. 194-202.
#'   \item Witten, I. H., Frank, E., & Hall, M. A. (2011). \emph{Data Mining: Practical
#'     Machine Learning Tools and Techniques} (3rd ed.). Morgan Kaufmann.
#'   \item Cerqueira, V., & Torgo, L. (2019). "Automatic Feature Engineering for
#'     Predictive Modeling of Multivariate Time Series". arXiv:1910.01344.
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate overdispersed credit scoring data with noise
#' set.seed(2024)
#' n <- 10000
#'
#' # Create feature with multiple regimes and noise
#' feature <- c(
#'   rnorm(3000, mean = 580, sd = 70), # High-risk cluster
#'   rnorm(4000, mean = 680, sd = 50), # Medium-risk cluster
#'   rnorm(2000, mean = 740, sd = 40), # Low-risk cluster
#'   runif(1000, min = 500, max = 800) # Noise (uniform distribution)
#' )
#'
#' target <- c(
#'   rbinom(3000, 1, 0.30), # 30% default rate
#'   rbinom(4000, 1, 0.12), # 12% default rate
#'   rbinom(2000, 1, 0.04), # 4% default rate
#'   rbinom(1000, 1, 0.15) # Noisy segment
#' )
#'
#' # Apply MDLP with default parameters
#' result <- ob_numerical_mdlp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   bin_cutoff = 0.05,
#'   max_n_prebins = 20
#' )
#'
#' # Inspect results
#' print(result$bin)
#' print(data.frame(
#'   Bin = result$bin,
#'   WoE = round(result$woe, 4),
#'   IV = round(result$iv, 4),
#'   Count = result$count
#' ))
#'
#' cat(sprintf("\nTotal IV: %.4f\n", result$total_iv))
#' cat(sprintf("Converged: %s\n", result$converged))
#' cat(sprintf("Iterations: %d\n", result$iterations))
#'
#' # Verify monotonicity
#' is_monotonic <- all(diff(result$woe) >= -1e-10)
#' cat(sprintf("WoE Monotonic: %s\n", is_monotonic))
#'
#' # Compare with different Laplace smoothing
#' result_nosmooth <- ob_numerical_mdlp(
#'   feature = feature,
#'   target = target,
#'   laplace_smoothing = 0.0 # No smoothing (risky for rare bins)
#' )
#'
#' result_highsmooth <- ob_numerical_mdlp(
#'   feature = feature,
#'   target = target,
#'   laplace_smoothing = 2.0 # Higher regularization
#' )
#'
#' # Compare WoE stability
#' data.frame(
#'   Bin = seq_along(result$woe),
#'   WoE_default = result$woe,
#'   WoE_no_smooth = result_nosmooth$woe,
#'   WoE_high_smooth = result_highsmooth$woe
#' )
#'
#' # Visualize binning structure
#' par(mfrow = c(1, 2))
#'
#' # WoE plot
#' plot(result$woe,
#'   type = "b", col = "blue", pch = 19,
#'   xlab = "Bin", ylab = "WoE",
#'   main = "Weight of Evidence by Bin"
#' )
#' grid()
#'
#' # IV contribution plot
#' barplot(result$iv,
#'   names.arg = seq_along(result$iv),
#'   col = "steelblue", border = "white",
#'   xlab = "Bin", ylab = "IV Contribution",
#'   main = sprintf("Total IV = %.4f", result$total_iv)
#' )
#' grid()
#' }
#'
#' @author
#' Lopes, J. E. (algorithm implementation based on Fayyad & Irani, 1993)
#'
#' @seealso
#' \code{\link{ob_numerical_ldb}} for density-based binning,
#' \code{\link{ob_numerical_mblp}} for monotonicity-constrained binning.
#'
#' @export
ob_numerical_mdlp <- function(feature,
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

  if (max_n_prebins < 2L) {
    stop("max_n_prebins must be at least 2.")
  }

  if (max_iterations < 1L) {
    stop("max_iterations must be at least 1.")
  }

  if (laplace_smoothing < 0) {
    stop("laplace_smoothing must be non-negative.")
  }

  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_mdlp",
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

  class(result) <- c("OptimalBinningMDLP", "OptimalBinning", "list")

  return(result)
}
