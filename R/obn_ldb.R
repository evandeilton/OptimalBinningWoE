#' @title Optimal Binning for Numerical Variables using Local Density Binning
#'
#' @description
#' Implements supervised discretization via Local Density Binning (LDB), a method
#' that leverages kernel density estimation to identify natural transition regions
#' in the feature space while optimizing the Weight of Evidence (WoE) monotonicity
#' and Information Value (IV) for binary classification tasks.
#'
#' @param feature Numeric vector of feature values to be binned. Missing values (NA)
#'   and infinite values are automatically filtered out during preprocessing.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins to generate (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins to generate (default: 5). Must be greater
#'   than or equal to \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
#'   Bins with frequency below this threshold are merged with adjacent bins. Must be
#'   in the range [0, 1].
#' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
#'   Controls granularity of initial density-based discretization.
#' @param enforce_monotonic Logical flag to enforce monotonicity in WoE values across
#'   bins (default: TRUE). When enabled, bins violating monotonicity are iteratively
#'   merged until global monotonicity is achieved.
#' @param convergence_threshold Convergence threshold for iterative optimization
#'   (default: 1e-6). Currently used for future extensions.
#' @param max_iterations Maximum number of iterations for merging operations
#'   (default: 1000). Prevents infinite loops in edge cases.
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer vector of bin identifiers (1-based indexing).}
#'   \item{bin}{Character vector of bin intervals in the format \code{"(lower;upper]"}.}
#'   \item{woe}{Numeric vector of Weight of Evidence values for each bin.}
#'   \item{iv}{Numeric vector of Information Value contributions for each bin.}
#'   \item{count}{Integer vector of total observations in each bin.}
#'   \item{count_pos}{Integer vector of positive class (target = 1) counts per bin.}
#'   \item{count_neg}{Integer vector of negative class (target = 0) counts per bin.}
#'   \item{event_rate}{Numeric vector of event rates (proportion of positives) per bin.}
#'   \item{cutpoints}{Numeric vector of cutpoints defining bin boundaries (excluding
#'     -Inf and +Inf).}
#'   \item{converged}{Logical flag indicating whether the algorithm converged within
#'     \code{max_iterations}.}
#'   \item{iterations}{Integer count of iterations performed during optimization.}
#'   \item{total_iv}{Numeric scalar representing the total Information Value
#'     (sum of all bin IVs).}
#'   \item{monotonicity}{Character string indicating monotonicity status: \code{"increasing"},
#'     \code{"decreasing"}, or \code{"none"}.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' The Local Density Binning (LDB) algorithm operates in four sequential phases:
#'
#' \strong{Phase 1: Density-Based Pre-binning}
#'
#' The algorithm employs kernel density estimation (KDE) with a Gaussian kernel
#' to identify the local density structure of the feature:
#'
#' \deqn{\hat{f}(x) = \frac{1}{nh\sqrt{2\pi}} \sum_{i=1}^{n} \exp\left[-\frac{(x - x_i)^2}{2h^2}\right]}
#'
#' where \eqn{h} is the bandwidth computed via Silverman's rule of thumb:
#'
#' \deqn{h = 0.9 \times \min(\hat{\sigma}, \text{IQR}/1.34) \times n^{-1/5}}
#'
#' Bin boundaries are placed at local minima of \eqn{\hat{f}(x)}, which correspond
#' to natural transition regions where density is lowest (analogous to valleys in
#' the density landscape). This strategy ensures bins capture homogeneous subpopulations.
#'
#' \strong{Phase 2: Weight of Evidence Computation}
#'
#' For each bin \eqn{i}, the WoE quantifies the log-ratio of positive to negative
#' class distributions, adjusted with Laplace smoothing (\eqn{\alpha = 0.5}) to
#' prevent division by zero:
#'
#' \deqn{\text{WoE}_i = \ln\left(\frac{\text{DistGood}_i}{\text{DistBad}_i}\right)}
#'
#' where:
#'
#' \deqn{\text{DistGood}_i = \frac{n_{i}^{+} + \alpha}{n^{+} + K\alpha}, \quad \text{DistBad}_i = \frac{n_{i}^{-} + \alpha}{n^{-} + K\alpha}}
#'
#' and \eqn{K} is the total number of bins. The Information Value for bin \eqn{i} is:
#'
#' \deqn{\text{IV}_i = (\text{DistGood}_i - \text{DistBad}_i) \times \text{WoE}_i}
#'
#' Total IV aggregates discriminatory power: \eqn{\text{IV}_{\text{total}} = \sum_{i=1}^{K} \text{IV}_i}.
#'
#' \strong{Phase 3: Monotonicity Enforcement}
#'
#' When \code{enforce_monotonic = TRUE}, the algorithm ensures WoE values are
#' monotonic with respect to bin order. The direction (increasing/decreasing) is
#' determined via Pearson correlation between bin indices and WoE values. Bins
#' violating monotonicity are iteratively merged using the merge strategy described
#' in Phase 4, continuing until global monotonicity is achieved or \code{min_bins}
#' is reached.
#'
#' This approach is rooted in isotonic regression principles (Robertson et al., 1988),
#' ensuring the scorecard maintains a consistent logical relationship between
#' feature values and credit risk.
#'
#' \strong{Phase 4: Adaptive Bin Merging}
#'
#' Two merging criteria are applied sequentially:
#'
#' \enumerate{
#'   \item \strong{Frequency-based merging}: Bins with total count below
#'     \code{bin_cutoff} \eqn{\times n} are merged with the adjacent bin having
#'     the most similar event rate (minimizing heterogeneity). If event rates are
#'     equivalent, the merge that preserves higher IV is preferred.
#'   \item \strong{Cardinality reduction}: If the number of bins exceeds \code{max_bins},
#'     the pair of adjacent bins minimizing IV loss when merged is identified via:
#'     \deqn{\Delta \text{IV}_{i,i+1} = \text{IV}_i + \text{IV}_{i+1} - \text{IV}_{\text{merged}}}
#'     This greedy optimization continues until \eqn{K \le} \code{max_bins}.
#' }
#'
#' \strong{Theoretical Foundations}
#'
#' \itemize{
#'   \item \strong{Kernel Density Estimation}: The bandwidth selection follows
#'     Silverman (1986, Chapter 3), balancing bias-variance tradeoff for univariate
#'     density estimation.
#'   \item \strong{Weight of Evidence}: Siddiqi (2006) formalizes WoE/IV as measures
#'     of predictive strength in credit scoring, with IV thresholds: \eqn{< 0.02}
#'     (unpredictive), 0.02-0.1 (weak), 0.1-0.3 (medium), 0.3-0.5 (strong), \eqn{> 0.5}
#'     (suspect overfitting).
#'   \item \strong{Supervised Discretization}: García et al. (2013) categorize LDB
#'     within "static" supervised methods that do not require iterative feedback
#'     from the model, unlike dynamic methods (e.g., ChiMerge).
#' }
#'
#' \strong{Computational Complexity}
#'
#' \itemize{
#'   \item KDE computation: \eqn{O(n^2)} for naive implementation (each of \eqn{n}
#'     points evaluates \eqn{n} kernel terms).
#'   \item Binary search for bin assignment: \eqn{O(n \log K)} where \eqn{K} is
#'     the number of bins.
#'   \item Merge iterations: \eqn{O(K^2 \times \text{max\_iterations})} in worst case.
#' }
#'
#' For large datasets (\eqn{n > 10^5}), the KDE phase dominates runtime.
#'
#' @references
#' \itemize{
#'   \item Silverman, B. W. (1986). \emph{Density Estimation for Statistics and
#'     Data Analysis}. Chapman and Hall/CRC.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards: Developing and
#'     Implementing Intelligent Credit Scoring}. Wiley.
#'   \item Dougherty, J., Kohavi, R., & Sahami, M. (1995). "Supervised and
#'     Unsupervised Discretization of Continuous Features". \emph{Proceedings of
#'     the 12th International Conference on Machine Learning}, pp. 194-202.
#'   \item Robertson, T., Wright, F. T., & Dykstra, R. L. (1988). \emph{Order
#'     Restricted Statistical Inference}. Wiley.
#'   \item García, S., Luengo, J., Sáez, J. A., López, V., & Herrera, F. (2013).
#'     "A Survey of Discretization Techniques: Taxonomy and Empirical Analysis in
#'     Supervised Learning". \emph{IEEE Transactions on Knowledge and Data Engineering},
#'     25(4), 734-750.
#' }
#'
#' @examples
#' \donttest{
#' # Simulate credit scoring data
#' set.seed(42)
#' n <- 10000
#' feature <- c(
#'   rnorm(3000, mean = 600, sd = 50), # Low-risk segment
#'   rnorm(4000, mean = 700, sd = 40), # Medium-risk segment
#'   rnorm(3000, mean = 750, sd = 30) # High-risk segment
#' )
#' target <- c(
#'   rbinom(3000, 1, 0.15), # 15% default rate
#'   rbinom(4000, 1, 0.08), # 8% default rate
#'   rbinom(3000, 1, 0.03) # 3% default rate
#' )
#'
#' # Apply LDB with monotonicity enforcement
#' result <- ob_numerical_ldb(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   bin_cutoff = 0.05,
#'   max_n_prebins = 20,
#'   enforce_monotonic = TRUE
#' )
#'
#' # Inspect binning quality
#' print(result$total_iv) # Should be > 0.1 for predictive features
#' print(result$monotonicity) # Should indicate direction
#'
#' # Visualize WoE pattern
#' plot(result$woe,
#'   type = "b", xlab = "Bin", ylab = "WoE",
#'   main = "Monotonic WoE Trend"
#' )
#'
#' # Generate scorecard transformation
#' bin_mapping <- data.frame(
#'   bin = result$bin,
#'   woe = result$woe,
#'   iv = result$iv
#' )
#' print(bin_mapping)
#' }
#'
#' @author
#' Lopes, J. E. (implemented algorithm)
#'
#' @seealso
#' \code{\link{ob_numerical_mdlp}} for Minimum Description Length Principle binning,
#' \code{\link{ob_numerical_mob}} for monotonic binning with similar constraints.
#'
#' @export
ob_numerical_ldb <- function(feature,
                             target,
                             min_bins = 3,
                             max_bins = 5,
                             bin_cutoff = 0.05,
                             max_n_prebins = 20,
                             enforce_monotonic = TRUE,
                             convergence_threshold = 1e-6,
                             max_iterations = 1000) {
  # Input validation (R-level defensive programming)
  if (!is.numeric(feature)) {
    stop("Feature must be a numeric vector.")
  }

  if (!is.vector(target) || !(is.integer(target) || is.numeric(target))) {
    stop("Target must be an integer or numeric vector.")
  }

  if (length(feature) != length(target)) {
    stop("Feature and target must have the same length.")
  }

  # Type coercion for C++ safety
  feature <- as.numeric(feature)
  target <- as.integer(target)

  # Validate binary target before expensive C++ call
  unique_target <- unique(target[!is.na(target)])
  if (!all(unique_target %in% c(0L, 1L)) || length(unique_target) != 2L) {
    stop("Target must contain exactly two classes: 0 and 1.")
  }

  # Validate parameters
  if (min_bins < 2L) {
    stop("min_bins must be at least 2.")
  }

  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.")
  }

  if (bin_cutoff < 0 || bin_cutoff > 1) {
    stop("bin_cutoff must be in the range [0, 1].")
  }

  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be at least equal to min_bins.")
  }

  if (max_iterations < 1L) {
    stop("max_iterations must be at least 1.")
  }

  # Call C++ implementation
  # NOTE: C++ signature is (target, feature, ...), matching the .Call order below
  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_ldb",
    target, # IntegerVector (position 1 in C++)
    feature, # NumericVector (position 2 in C++)
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.logical(enforce_monotonic),
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )

  # Add class attribute for S3 method dispatch
  class(result) <- c("OptimalBinningLDB", "OptimalBinning", "list")

  return(result)
}
