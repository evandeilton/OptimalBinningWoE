#' @title Optimal Binning for Numerical Features Using Monotonic Binning via Linear Programming
#'
#' @description
#' Implements a greedy optimization algorithm for supervised discretization of
#' numerical features with **guaranteed monotonicity** in Weight of Evidence (WoE).
#' Despite the "Linear Programming" designation, this method employs an iterative
#' heuristic based on quantile pre-binning, Information Value (IV) optimization,
#' and monotonicity enforcement through adaptive bin merging.
#'
#' \strong{Important Note}: This algorithm does not use formal Linear Programming
#' solvers (e.g., simplex method). The name reflects the conceptual formulation
#' of binning as a constrained optimization problem, but the implementation uses
#' a deterministic greedy heuristic for computational efficiency.
#'
#' @param feature Numeric vector of feature values to be binned. Missing values (NA)
#'   and infinite values are automatically removed during preprocessing.
#' @param target Integer vector of binary target values (must contain only 0 and 1).
#'   Must have the same length as \code{feature}.
#' @param min_bins Minimum number of bins to generate (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins to generate (default: 5). Must be greater
#'   than or equal to \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
#'   Bins with frequency below this threshold are merged with adjacent bins. Must be
#'   in the range (0, 1).
#' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
#'   Controls granularity of initial quantile-based discretization.
#' @param force_monotonic_direction Integer flag to force a specific monotonicity
#'   direction (default: 0). Valid values:
#'   \itemize{
#'     \item \code{0}: Automatically determine direction via correlation between
#'       bin indices and WoE values.
#'     \item \code{1}: Force increasing monotonicity (WoE increases with feature value).
#'     \item \code{-1}: Force decreasing monotonicity (WoE decreases with feature value).
#'   }
#' @param convergence_threshold Convergence threshold for iterative optimization
#'   (default: 1e-6). Iteration stops when the absolute change in total IV between
#'   consecutive iterations falls below this value.
#' @param max_iterations Maximum number of iterations for the optimization loop
#'   (default: 1000). Prevents infinite loops in pathological cases.
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
#' The Monotonic Binning via Linear Programming (MBLP) algorithm operates in four
#' sequential phases designed to balance predictive power (IV maximization) and
#' interpretability (monotonic WoE):
#'
#' \strong{Phase 1: Quantile-Based Pre-binning}
#'
#' Initial bin boundaries are determined using empirical quantiles of the feature
#' distribution. For \eqn{k} pre-bins, cutpoints are computed as:
#'
#' \deqn{q_i = x_{(\lceil p_i \times (N - 1) \rceil)}, \quad p_i = \frac{i}{k}, \quad i = 1, 2, \ldots, k-1}
#'
#' where \eqn{x_{(j)}} denotes the \eqn{j}-th order statistic. This approach ensures
#' equal-frequency bins under the assumption of continuous data, though ties may
#' cause deviations in practice. The first and last boundaries are set to
#' \eqn{-\infty} and \eqn{+\infty}, respectively.
#'
#' \strong{Phase 2: Frequency-Based Bin Merging}
#'
#' Bins with total count below \code{bin_cutoff} \eqn{\times N} are iteratively
#' merged with adjacent bins to ensure statistical reliability. The merge strategy
#' selects the neighbor with the smallest count (greedy heuristic), continuing
#' until all bins meet the frequency threshold or \code{min_bins} is reached.
#'
#' \strong{Phase 3: Monotonicity Direction Determination}
#'
#' If \code{force_monotonic_direction = 0}, the algorithm computes the Pearson
#' correlation between bin indices and WoE values:
#'
#' \deqn{\rho = \frac{\sum_{i=1}^{k} (i - \bar{i})(\text{WoE}_i - \overline{\text{WoE}})}{\sqrt{\sum_{i=1}^{k} (i - \bar{i})^2 \sum_{i=1}^{k} (\text{WoE}_i - \overline{\text{WoE}})^2}}}
#'
#' The monotonicity direction is set as:
#' \deqn{\text{direction} = \begin{cases} 1 & \text{if } \rho \ge 0 \text{ (increasing)} \\ -1 & \text{if } \rho < 0 \text{ (decreasing)} \end{cases}}
#'
#' If \code{force_monotonic_direction} is explicitly set to 1 or -1, that value
#' overrides the correlation-based determination.
#'
#' \strong{Phase 4: Iterative Optimization Loop}
#'
#' The core optimization alternates between two enforcement steps until convergence:
#'
#' \enumerate{
#'   \item \strong{Cardinality Constraint}: If the number of bins \eqn{k} exceeds
#'     \code{max_bins}, the algorithm identifies the pair of adjacent bins
#'     \eqn{(i, i+1)} that minimizes the IV loss when merged:
#'     \deqn{\Delta \text{IV}_{i,i+1} = \text{IV}_i + \text{IV}_{i+1} - \text{IV}_{\text{merged}}}
#'     where \eqn{\text{IV}_{\text{merged}}} is recalculated using combined counts.
#'     The merge is performed only if it preserves monotonicity (checked via WoE
#'     comparison with neighboring bins).
#'
#'   \item \strong{Monotonicity Enforcement}: For each pair of consecutive bins,
#'     violations are detected as:
#'     \itemize{
#'       \item \strong{Increasing}: \eqn{\text{WoE}_i < \text{WoE}_{i-1} - \epsilon}
#'       \item \strong{Decreasing}: \eqn{\text{WoE}_i > \text{WoE}_{i-1} + \epsilon}
#'     }
#'     where \eqn{\epsilon = 10^{-10}} (numerical tolerance). Violating bins are
#'     immediately merged.
#'
#'   \item \strong{Convergence Test}: After each iteration, the total IV is compared
#'     to the previous iteration. If \eqn{|\text{IV}^{(t)} - \text{IV}^{(t-1)}| < \text{convergence\_threshold}}
#'     or monotonicity is achieved, the loop terminates.
#' }
#'
#' \strong{Weight of Evidence Computation}
#'
#' WoE for bin \eqn{i} uses Laplace smoothing (\eqn{\alpha = 0.5}) to handle zero counts:
#'
#' \deqn{\text{WoE}_i = \ln\left(\frac{\text{DistGood}_i}{\text{DistBad}_i}\right)}
#'
#' where:
#' \deqn{\text{DistGood}_i = \frac{n_i^{+} + \alpha}{n^{+} + k\alpha}, \quad \text{DistBad}_i = \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}}
#'
#' and \eqn{k} is the current number of bins. The Information Value contribution is:
#'
#' \deqn{\text{IV}_i = (\text{DistGood}_i - \text{DistBad}_i) \times \text{WoE}_i}
#'
#' \strong{Theoretical Foundations}
#'
#' \itemize{
#'   \item \strong{Monotonicity Requirement}: Zeng (2014) proves that monotonic WoE
#'     is a necessary condition for stable scorecards under data drift. Non-monotonic
#'     patterns often indicate overfitting to noise.
#'   \item \strong{Greedy Optimization}: Unlike global optimizers (MILP), greedy
#'     heuristics provide no optimality guarantees but achieve O(k²) complexity
#'     per iteration versus exponential for exact methods.
#'   \item \strong{Quantile Binning}: Ensures initial bins have approximately equal
#'     sample sizes, reducing variance in WoE estimates (especially critical for
#'     minority classes).
#' }
#'
#' \strong{Comparison with True Linear Programming}
#'
#' Formal LP formulations for binning (Belotti et al., 2016) express the problem as:
#'
#' \deqn{\max_{\mathbf{z}, \mathbf{b}} \sum_{i=1}^{k} \text{IV}_i(\mathbf{b})}
#'
#' subject to:
#' \deqn{\text{WoE}_i \le \text{WoE}_{i+1} \quad \forall i \quad \text{(monotonicity)}}
#' \deqn{\sum_{j=1}^{N} z_{ij} = 1 \quad \forall j \quad \text{(assignment)}}
#' \deqn{z_{ij} \in \{0, 1\}, \quad b_i \in \mathbb{R}}
#'
#' where \eqn{z_{ij}} indicates if observation \eqn{j} is in bin \eqn{i}, and
#' \eqn{b_i} are bin boundaries. Such formulations require MILP solvers (CPLEX,
#' Gurobi) and scale poorly beyond \eqn{N > 10^4}. MBLP sacrifices global optimality
#' for \strong{scalability} and \strong{determinism}.
#'
#' \strong{Computational Complexity}
#'
#' \itemize{
#'   \item Initial sorting: \eqn{O(N \log N)}
#'   \item Quantile computation: \eqn{O(k)}
#'   \item Per-iteration operations: \eqn{O(k^2)} (pairwise comparisons for merging)
#'   \item Total: \eqn{O(N \log N + k^2 \times \text{max\_iterations})}
#' }
#'
#' For typical credit scoring datasets (\eqn{N \sim 10^5}, \eqn{k \sim 5}),
#' runtime is dominated by sorting. Pathological cases (highly non-monotonic data)
#' may require many iterations to enforce monotonicity.
#'
#' @references
#' \itemize{
#'   \item Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm in
#'     Credit Scoring". \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'   \item Mironchyk, P., & Tchistiakov, V. (2017). "Monotone optimal binning
#'     algorithm for credit risk modeling". \emph{Frontiers in Applied Mathematics
#'     and Statistics}, 3, 2.
#'   \item Belotti, P., Bonami, P., Fischetti, M., Lodi, A., Monaci, M.,
#'     Nogales-Gómez, A., & Salvagnin, D. (2016). "On handling indicator constraints
#'     in mixed integer programming". \emph{Computational Optimization and Applications},
#'     65(3), 545-566.
#'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). \emph{Credit Scoring
#'     and Its Applications}. SIAM.
#'   \item Louzada, F., Ara, A., & Fernandes, G. B. (2016). "Classification methods
#'     applied to credit scoring: Systematic review and overall comparison". \emph{Surveys
#'     in Operations Research and Management Science}, 21(2), 117-134.
#'   \item Naeem, B., Huda, N., & Aziz, A. (2013). "Developing Scorecards with
#'     Constrained Logistic Regression". \emph{Proceedings of the International Workshop
#'     on Data Mining Applications}.
#' }
#'
#' @examples
#' \dontrun{
#' # Simulate non-monotonic credit scoring data
#' set.seed(123)
#' n <- 8000
#' feature <- c(
#'   rnorm(2000, mean = 550, sd = 60), # High-risk segment (low scores)
#'   rnorm(3000, mean = 680, sd = 50), # Medium-risk segment
#'   rnorm(2000, mean = 720, sd = 40), # Low-risk segment
#'   rnorm(1000, mean = 620, sd = 55) # Mixed segment (creates non-monotonicity)
#' )
#' target <- c(
#'   rbinom(2000, 1, 0.25), # 25% default rate
#'   rbinom(3000, 1, 0.10), # 10% default rate
#'   rbinom(2000, 1, 0.03), # 3% default rate
#'   rbinom(1000, 1, 0.15) # 15% default rate (violates monotonicity)
#' )
#'
#' # Apply MBLP with automatic monotonicity detection
#' result_auto <- ob_numerical_mblp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5,
#'   bin_cutoff = 0.05,
#'   max_n_prebins = 20,
#'   force_monotonic_direction = 0 # Auto-detect
#' )
#'
#' print(result_auto$monotonicity) # Check detected direction
#' print(result_auto$total_iv) # Should be > 0.1 for predictive features
#'
#' # Force decreasing monotonicity (higher score = lower WoE = lower risk)
#' result_forced <- ob_numerical_mblp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 4,
#'   max_bins = 6,
#'   force_monotonic_direction = -1 # Force decreasing
#' )
#'
#' # Verify monotonicity enforcement
#' stopifnot(all(diff(result_forced$woe) <= 1e-9)) # Should be non-increasing
#'
#' # Compare convergence
#' cat(sprintf(
#'   "Auto mode: %d iterations, IV = %.4f\n",
#'   result_auto$iterations, result_auto$total_iv
#' ))
#' cat(sprintf(
#'   "Forced mode: %d iterations, IV = %.4f\n",
#'   result_forced$iterations, result_forced$total_iv
#' ))
#'
#' # Visualize binning quality
#' par(mfrow = c(1, 2))
#' plot(result_auto$woe,
#'   type = "b", col = "blue", pch = 19,
#'   xlab = "Bin", ylab = "WoE", main = "Auto-Detected Monotonicity"
#' )
#' plot(result_forced$woe,
#'   type = "b", col = "red", pch = 19,
#'   xlab = "Bin", ylab = "WoE", main = "Forced Decreasing"
#' )
#' }
#'
#' @author
#' Lopes, J. E. (implemented algorithm based on Mironchyk & Tchistiakov, 2017)
#'
#' @seealso
#' \code{\link{ob_numerical_ldb}} for density-based binning,
#' \code{\link{ob_numerical_mdlp}} for entropy-based discretization with MDLP criterion.
#'
#' @export
ob_numerical_mblp <- function(feature,
                              target,
                              min_bins = 3,
                              max_bins = 5,
                              bin_cutoff = 0.05,
                              max_n_prebins = 20,
                              force_monotonic_direction = 0,
                              convergence_threshold = 1e-6,
                              max_iterations = 1000) {
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

  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("bin_cutoff must be in the range (0, 1).")
  }

  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be at least equal to min_bins.")
  }

  if (max_iterations < 1L) {
    stop("max_iterations must be at least 1.")
  }

  # Validate force_monotonic_direction
  if (!force_monotonic_direction %in% c(-1L, 0L, 1L)) {
    stop("force_monotonic_direction must be -1 (decreasing), 0 (auto), or 1 (increasing).")
  }

  result <- .Call(
    "_OptimalBinningWoE_optimal_binning_numerical_mblp",
    target, # IntegerVector (position 1 in C++)
    feature, # NumericVector (position 2 in C++)
    as.integer(min_bins),
    as.integer(max_bins),
    as.numeric(bin_cutoff),
    as.integer(max_n_prebins),
    as.integer(force_monotonic_direction), # Explicit coercion critical
    as.numeric(convergence_threshold),
    as.integer(max_iterations),
    PACKAGE = "OptimalBinningWoE"
  )

  return(result)
}
