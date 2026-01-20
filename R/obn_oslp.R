#' @title Optimal Binning for Numerical Variables using Optimal Supervised Learning Partitioning
#'
#' @description
#' Implements a greedy binning algorithm with quantile-based pre-binning and
#' monotonicity enforcement. \strong{Important Note}: Despite "Optimal Supervised
#' Learning Partitioning" and "LP" in the name, the algorithm uses \strong{greedy
#' heuristics} without formal Linear Programming or convex optimization. The method
#' is functionally equivalent to \code{\link{ob_numerical_mrblp}} with minor
#' differences in pre-binning strategy and bin reduction criteria.
#'
#' Users seeking true optimization-based binning should consider Mixed-Integer
#' Programming (MIP) implementations (e.g., via \code{ompr} or \code{lpSolve}
#' packages), though these scale poorly beyond N > 10,000 observations.
#'
#' @param feature Numeric vector of feature values. Missing values (NA) and infinite
#'   values are \strong{not permitted} and will trigger an error.
#' @param target Integer or numeric vector of binary target values (must contain only
#'   0 and 1). Must have the same length as \code{feature}. Unlike other binning
#'   methods, OSLP internally uses \code{double} for target, allowing implicit
#'   conversion from integer.
#' @param min_bins Minimum number of bins (default: 3). Must be at least 2.
#' @param max_bins Maximum number of bins (default: 5). Must be greater than or
#'   equal to \code{min_bins}.
#' @param bin_cutoff Minimum fraction of total observations per bin (default: 0.05).
#'   Must be in (0, 1).
#' @param max_n_prebins Maximum number of pre-bins (default: 20). Must be at least
#'   equal to \code{min_bins}.
#' @param convergence_threshold Convergence threshold for IV change (default: 1e-6).
#' @param max_iterations Maximum iterations (default: 1000).
#' @param laplace_smoothing Laplace smoothing parameter (default: 0.5). Must be
#'   non-negative.
#'
#' @return A list containing:
#' \describe{
#'   \item{id}{Integer bin identifiers (1-based).}
#'   \item{bin}{Character bin intervals \code{"[lower;upper)"}.}
#'   \item{woe}{Numeric WoE values (guaranteed monotonic).}
#'   \item{iv}{Numeric IV contributions per bin.}
#'   \item{count}{Integer total observations per bin.}
#'   \item{count_pos}{Integer positive class counts.}
#'   \item{count_neg}{Integer negative class counts.}
#'   \item{event_rate}{Numeric event rates.}
#'   \item{cutpoints}{Numeric bin boundaries (excluding ±Inf).}
#'   \item{total_iv}{Total Information Value.}
#'   \item{converged}{Logical convergence flag.}
#'   \item{iterations}{Integer iteration count.}
#' }
#'
#' @details
#' \strong{Algorithm Overview}
#'
#' OSLP executes in five phases:
#'
#' \strong{Phase 1: Quantile-Based Pre-binning}
#'
#' Unlike equal-frequency methods that ensure balanced bin sizes, OSLP places
#' cutpoints at quantiles of \strong{unique feature values}:
#'
#' \deqn{\text{edge}_i = \text{unique\_vals}\left[\left\lfloor p_i \times (n_{\text{unique}} - 1) \right\rfloor\right]}
#'
#' where \eqn{p_i = i / \text{max\_n\_prebins}}.
#'
#' \strong{Critique}: If unique values are clustered (e.g., many observations at
#' specific values), bins may have vastly different sizes, violating the equal-frequency
#' principle that ensures statistical stability.
#'
#' \strong{Phase 2: Rare Bin Merging}
#'
#' Bins with \eqn{n_i / N < \text{bin\_cutoff}} are merged. The merge direction
#' minimizes IV loss:
#'
#' \deqn{\Delta \text{IV} = \text{IV}_i + \text{IV}_{i+d} - \text{IV}_{\text{merged}}}
#'
#' where \eqn{d \in \{-1, +1\}} (left or right neighbor).
#'
#' \strong{Phase 3: Initial WoE/IV Calculation}
#'
#' Standard WoE with Laplace smoothing:
#'
#' \deqn{\text{WoE}_i = \ln\left(\frac{n_i^{+} + \alpha}{n^{+} + k\alpha} \bigg/ \frac{n_i^{-} + \alpha}{n^{-} + k\alpha}\right)}
#'
#' \strong{Phase 4: Monotonicity Enforcement}
#'
#' Direction determined via majority vote (identical to MRBLP):
#'
#' \deqn{\text{increasing} = \begin{cases} \text{TRUE} & \text{if } \sum_i \mathbb{1}_{\{\text{WoE}_i > \text{WoE}_{i-1}\}} \ge \sum_i \mathbb{1}_{\{\text{WoE}_i < \text{WoE}_{i-1}\}} \\ \text{FALSE} & \text{otherwise} \end{cases}}
#'
#' Violations are merged iteratively.
#'
#' \strong{Phase 5: Bin Count Reduction}
#'
#' If \eqn{k > \text{max\_bins}}, merge bins with the \strong{smallest combined IV}:
#'
#' \deqn{\text{merge\_idx} = \arg\min_{i=0}^{k-2} \left( \text{IV}_i + \text{IV}_{i+1} \right)}
#'
#' \strong{Rationale}: Assumes bins with low total IV contribute least to predictive
#' power. However, this ignores the interaction between bins; a low-IV bin may be
#' essential for monotonicity or preventing gaps.
#'
#' \strong{Theoretical Foundations}
#'
#' Despite the name "Optimal Supervised Learning Partitioning", the algorithm lacks:
#' \itemize{
#'   \item \strong{Global optimality guarantees}: Greedy merging is myopic
#'   \item \strong{Formal loss function}: No explicit objective being minimized
#'   \item \strong{LP formulation}: No constraint matrix, simplex solver, or dual variables
#' }
#'
#' A true optimal partitioning approach would formulate the problem as:
#'
#' \deqn{\min_{\mathbf{z}, \mathbf{b}} \left\{ -\sum_{i=1}^{k} \text{IV}_i(\mathbf{b}) + \lambda k \right\}}
#'
#' subject to:
#' \deqn{\sum_{j=1}^{k} z_{ij} = 1 \quad \forall i \in \{1, \ldots, N\}}
#' \deqn{\text{WoE}_j \le \text{WoE}_{j+1} \quad \forall j}
#' \deqn{z_{ij} \in \{0, 1\}, \quad b_j \in \mathbb{R}}
#'
#' where \eqn{z_{ij}} indicates observation \eqn{i} assigned to bin \eqn{j}, and
#' \eqn{\lambda} is a complexity penalty. This requires MILP solvers (CPLEX, Gurobi)
#' and is intractable for \eqn{N > 10^4}.
#'
#' \strong{Comparison with Related Methods}
#'
#' \tabular{lllll}{
#'   \strong{Method} \tab \strong{Pre-binning} \tab \strong{Direction} \tab \strong{Merge (max_bins)} \tab \strong{Target Type} \cr
#'   OSLP \tab Quantile (unique vals) \tab Majority vote \tab Min (IV(i) + IV(i+1)) \tab double \cr
#'   MRBLP \tab Equal-frequency \tab Majority vote \tab Min |IV(i) - IV(i+1)| \tab int \cr
#'   MOB \tab Equal-frequency \tab First two bins \tab Min IV loss \tab int \cr
#'   MBLP \tab Quantile (data) \tab Correlation \tab Min IV loss \tab int \cr
#' }
#'
#' \strong{When to Use OSLP}
#'
#' \itemize{
#'   \item \strong{Use OSLP}: Never. Use MBLP or MOB instead for better pre-binning
#'     and merge strategies.
#'   \item \strong{Use MBLP}: For robust direction detection via correlation.
#'   \item \strong{Use MDLP}: For information-theoretic stopping criteria.
#'   \item \strong{Use True LP}: For small datasets (N < 1000) where global optimality
#'     is critical and computational cost is acceptable.
#' }
#'
#' @references
#' \itemize{
#'   \item Mironchyk, P., & Tchistiakov, V. (2017). "Monotone optimal binning algorithm
#'     for credit risk modeling". \emph{Frontiers in Applied Mathematics and Statistics}, 3, 2.
#'   \item Zeng, G. (2014). "A Necessary Condition for a Good Binning Algorithm in
#'     Credit Scoring". \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'   \item Fayyad, U. M., & Irani, K. B. (1993). "Multi-Interval Discretization of
#'     Continuous-Valued Attributes". \emph{IJCAI}, pp. 1022-1027.
#'   \item Good, I. J. (1952). "Rational Decisions". \emph{Journal of the Royal
#'     Statistical Society B}, 14(1), 107-114.
#'   \item Siddiqi, N. (2006). \emph{Credit Risk Scorecards}. Wiley.
#' }
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' n <- 5000
#' feature <- c(
#'   rnorm(2000, 600, 50),
#'   rnorm(2000, 680, 40),
#'   rnorm(1000, 740, 30)
#' )
#' target <- c(
#'   rbinom(2000, 1, 0.25),
#'   rbinom(2000, 1, 0.10),
#'   rbinom(1000, 1, 0.03)
#' )
#'
#' result <- ob_numerical_oslp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' print(result$woe)
#' print(result$total_iv)
#'
#' # Compare with MRBLP (should be nearly identical)
#' result_mrblp <- ob_numerical_mrblp(
#'   feature = feature,
#'   target = target,
#'   min_bins = 3,
#'   max_bins = 5
#' )
#'
#' data.frame(
#'   Method = c("OSLP", "MRBLP"),
#'   Total_IV = c(result$total_iv, result_mrblp$total_iv),
#'   N_Bins = c(length(result$woe), length(result_mrblp$woe))
#' )
#' }
#'
#' @author
#' Lopes, J. E.
#'
#' @seealso
#' \code{\link{ob_numerical_mrblp}} for nearly identical algorithm with better pre-binning,
#' \code{\link{ob_numerical_mblp}} for correlation-based direction detection,
#' \code{\link{ob_numerical_mdlp}} for information-theoretic optimality.
#'
#' @export
ob_numerical_oslp <- function(feature,
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
    "_OptimalBinningWoE_optimal_binning_numerical_oslp",
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

  class(result) <- c("OptimalBinningOSLP", "OptimalBinning", "list")

  return(result)
}
