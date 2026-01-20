#' @title Unified Optimal Binning and Weight of Evidence Transformation
#'
#' @description
#' Master interface for optimal discretization and Weight of Evidence (WoE)
#' computation across numerical and categorical predictors. This function serves
#' as the primary entry point for the \strong{OptimalBinningWoE} package, providing
#' automatic feature type detection, intelligent algorithm selection, and unified
#' output structures for seamless integration into credit scoring and predictive
#' modeling workflows.
#'
#' @param data A \code{data.frame} containing the predictor variables (features)
#'   and the response variable (target). All features to be binned must be present
#'   in this data frame. The data frame should not contain list-columns.
#' @param target Character string specifying the column name of the response
#'   variable. Must be a binary outcome encoded as integers \code{0} (non-event)
#'   and \code{1} (event), or a multinomial outcome encoded as integers
#'   \code{0, 1, 2, ..., K}. Missing values in the target are not permitted.
#' @param feature Optional character vector specifying which columns to process.
#'   If \code{NULL} (default), all columns except \code{target} are processed.
#'   Features containing only missing values are automatically skipped with a warning.
#' @param min_bins Integer specifying the minimum number of bins. Must satisfy
#'   \eqn{2 \le} \code{min_bins} \eqn{\le} \code{max_bins}. Algorithms may
#'   produce fewer bins if the data has insufficient unique values. Default is 2.
#' @param max_bins Integer specifying the maximum number of bins. Controls the
#'   granularity of discretization. Higher values capture more detail but risk
#'   overfitting. Typical values range from 5 to 10 for credit scoring applications.
#'   Default is 7.
#' @param algorithm Character string specifying the binning algorithm.
#'   Use \code{"auto"} (default) for automatic selection based on target type:
#'   \code{"jedi"} for binary targets, \code{"jedi_mwoe"} for multinomial.
#'   See Details for the complete algorithm taxonomy.
#' @param control A list of algorithm-specific control parameters created by
#'   \code{\link{control.obwoe}}. Provides fine-grained control over convergence
#'   thresholds, bin cutoffs, and other optimization parameters.
#'
#' @return An S3 object of class \code{"obwoe"} containing:
#' \describe{
#'   \item{\code{results}}{Named list where each element contains the binning
#'     result for a single feature, including:
#'     \describe{
#'       \item{\code{bin}}{Character vector of bin labels/intervals}
#'       \item{\code{woe}}{Numeric vector of Weight of Evidence per bin}
#'       \item{\code{iv}}{Numeric vector of Information Value contribution per bin}
#'       \item{\code{count}}{Integer vector of observation counts per bin}
#'       \item{\code{count_pos}}{Integer vector of positive (event) counts per bin}
#'       \item{\code{count_neg}}{Integer vector of negative (non-event) counts per bin}
#'       \item{\code{cutpoints}}{Numeric vector of bin boundaries (numerical only)}
#'       \item{\code{converged}}{Logical indicating algorithm convergence}
#'       \item{\code{iterations}}{Integer count of optimization iterations}
#'     }
#'   }
#'   \item{\code{summary}}{Data frame with one row per feature containing:
#'     \code{feature} (name), \code{type} (numerical/categorical),
#'     \code{algorithm} (used), \code{n_bins} (count), \code{total_iv} (sum),
#'     \code{error} (logical flag)}
#'   \item{\code{target}}{Name of the target column}
#'   \item{\code{target_type}}{Detected type: \code{"binary"} or \code{"multinomial"}}
#'   \item{\code{n_features}}{Number of features processed}
#'   \item{\code{call}}{The matched function call for reproducibility}
#' }
#'
#' @details
#' \subsection{Theoretical Foundation}{
#'
#' Weight of Evidence (WoE) transformation is a staple of credit scoring methodology,
#' originating from information theory and the concept of evidential support
#' (Good, 1950; Kullback, 1959). For a bin \eqn{i}, the WoE is defined as:
#'
#' \deqn{WoE_i = \ln\left(\frac{p_i}{n_i}\right) = \ln\left(\frac{N_{i,1}/N_1}{N_{i,0}/N_0}\right)}
#'
#' where:
#' \itemize{
#'   \item \eqn{N_{i,1}} = number of events (target=1) in bin \eqn{i}
#'   \item \eqn{N_{i,0}} = number of non-events (target=0) in bin \eqn{i}
#'   \item \eqn{N_1}, \eqn{N_0} = total events and non-events, respectively
#'   \item \eqn{p_i = N_{i,1}/N_1} = proportion of events in bin \eqn{i}
#'   \item \eqn{n_i = N_{i,0}/N_0} = proportion of non-events in bin \eqn{i}
#' }
#'
#' The Information Value (IV) quantifies the total predictive power of a binning:
#'
#' \deqn{IV = \sum_{i=1}^{k} (p_i - n_i) \times WoE_i = \sum_{i=1}^{k} (p_i - n_i) \times \ln\left(\frac{p_i}{n_i}\right)}
#'
#' where \eqn{k} is the number of bins. IV is equivalent to the Kullback-Leibler
#' divergence between the event and non-event distributions.
#' }
#'
#' \subsection{Algorithm Taxonomy}{
#'
#' The package provides 28 algorithms organized by supported feature types:
#'
#' \strong{Universal Algorithms} (both numerical and categorical):
#' \tabular{lll}{
#'   \strong{ID} \tab \strong{Full Name} \tab \strong{Method} \cr
#'   \code{jedi} \tab Joint Entropy-Driven Information \tab Heuristic + IV optimization \cr
#'   \code{jedi_mwoe} \tab JEDI Multinomial WoE \tab Extension for K>2 classes \cr
#'   \code{cm} \tab ChiMerge \tab Bottom-up chi-squared merging \cr
#'   \code{dp} \tab Dynamic Programming \tab Exact optimal IV partitioning \cr
#'   \code{dmiv} \tab Decision Tree MIV \tab Recursive partitioning \cr
#'   \code{fetb} \tab Fisher's Exact Test \tab Statistical significance-based \cr
#'   \code{mob} \tab Monotonic Optimal Binning \tab IV-optimal with monotonicity \cr
#'   \code{sketch} \tab Sketching \tab Probabilistic data structures \cr
#'   \code{udt} \tab Unsupervised Decision Tree \tab Entropy-based without target
#' }
#'
#' \strong{Numerical-Only Algorithms}:
#' \tabular{ll}{
#'   \strong{ID} \tab \strong{Description} \cr
#'   \code{bb} \tab Branch and Bound (exact search) \cr
#'   \code{ewb} \tab Equal Width Binning (unsupervised) \cr
#'   \code{fast_mdlp} \tab Fast MDLP with pruning \cr
#'   \code{ir} \tab Isotonic Regression \cr
#'   \code{kmb} \tab K-Means Binning \cr
#'   \code{ldb} \tab Local Density Binning \cr
#'   \code{lpdb} \tab Local Polynomial Density \cr
#'   \code{mblp} \tab Monotonic Binning LP \cr
#'   \code{mdlp} \tab Minimum Description Length \cr
#'   \code{mrblp} \tab Monotonic Regression LP \cr
#'   \code{oslp} \tab Optimal Supervised LP \cr
#'   \code{ubsd} \tab Unsupervised Std-Dev Based
#' }
#'
#' \strong{Categorical-Only Algorithms}:
#' \tabular{ll}{
#'   \strong{ID} \tab \strong{Description} \cr
#'   \code{gmb} \tab Greedy Monotonic Binning \cr
#'   \code{ivb} \tab Information Value DP (exact) \cr
#'   \code{mba} \tab Modified Binning Algorithm \cr
#'   \code{milp} \tab Mixed Integer LP \cr
#'   \code{sab} \tab Simulated Annealing \cr
#'   \code{sblp} \tab Similarity-Based LP \cr
#'   \code{swb} \tab Sliding Window Binning
#' }
#' }
#'
#' \subsection{Automatic Type Detection}{
#'
#' Feature types are detected as follows:
#' \itemize{
#'   \item \strong{Numerical}: \code{numeric} or \code{integer} vectors not of class \code{factor}
#'   \item \strong{Categorical}: \code{character}, \code{factor}, or \code{logical} vectors
#' }
#'
#' When \code{algorithm = "auto"}, the function selects:
#' \itemize{
#'   \item \code{"jedi"} for binary targets (recommended for most use cases)
#'   \item \code{"jedi_mwoe"} for multinomial targets (K > 2 classes)
#' }
#' }
#'
#' \subsection{IV Interpretation Guidelines}{
#'
#' Siddiqi (2006) provides the following IV thresholds for variable selection:
#' \tabular{ll}{
#'   \strong{IV Range} \tab \strong{Predictive Power} \cr
#'   < 0.02 \tab Unpredictive \cr
#'   0.02 - 0.10 \tab Weak \cr
#'   0.10 - 0.30 \tab Medium \cr
#'   0.30 - 0.50 \tab Strong \cr
#'   > 0.50 \tab Suspicious (likely overfitting)
#' }
#' }
#'
#' \subsection{Computational Considerations}{
#'
#' Time complexity varies by algorithm:
#' \itemize{
#'   \item \strong{JEDI, ChiMerge, MOB}: \eqn{O(n \log n + k^2 m)} where \eqn{n} = observations, \eqn{k} = bins, \eqn{m} = iterations
#'   \item \strong{Dynamic Programming}: \eqn{O(n \cdot k^2)} for exact solution
#'   \item \strong{Equal Width}: \eqn{O(n)} (fastest, but unsupervised)
#'   \item \strong{MILP, SBLP}: Potentially exponential (NP-hard problems)
#' }
#'
#' For large datasets (\eqn{n > 10^6}), consider:
#' \enumerate{
#'   \item Using \code{algorithm = "sketch"} for approximate streaming
#'   \item Reducing \code{max_n_prebins} via \code{control.obwoe()}
#'   \item Sampling the data before binning
#' }
#' }
#'
#' @references
#' Good, I. J. (1950). Probability and the Weighing of Evidence.
#' \emph{Griffin, London}.
#'
#' Kullback, S. (1959). Information Theory and Statistics.
#' \emph{Wiley, New York}.
#'
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring
#' and Its Applications. \emph{SIAM Monographs on Mathematical Modeling
#' and Computation}. \doi{10.1137/1.9780898718317}
#'
#' Navas-Palencia, G. (2020). Optimal Binning: Mathematical Programming
#' Formulation and Solution Approach. \emph{Expert Systems with Applications},
#' 158, 113508. \doi{10.1016/j.eswa.2020.113508}
#'
#' Zeng, G. (2014). A Necessary Condition for a Good Binning Algorithm
#' in Credit Scoring. \emph{Applied Mathematical Sciences}, 8(65), 3229-3242.
#'
#' @seealso
#' \code{\link{control.obwoe}} for algorithm-specific parameters,
#' \code{\link{obwoe_algorithms}} to list all available algorithms with capabilities,
#' \code{\link{print.obwoe}} for display methods,
#' \code{\link{ob_apply_woe_num}} and \code{\link{ob_apply_woe_cat}} to apply
#' WoE transformations to new data.
#'
#' For individual algorithms with full parameter control:
#' \code{\link{ob_numerical_jedi}}, \code{\link{ob_categorical_jedi}},
#' \code{\link{ob_numerical_mdlp}}, \code{\link{ob_categorical_ivb}}.
#'
#' @examples
#' \donttest{
#' # =============================================================================
#' # Example 1: Basic Usage with Mixed Feature Types
#' # =============================================================================
#' set.seed(42)
#' n <- 2000
#'
#' # Simulate credit scoring data
#' df <- data.frame(
#'   # Numerical features
#'   age = pmax(18, pmin(80, rnorm(n, 45, 15))),
#'   income = exp(rnorm(n, 10, 0.8)),
#'   debt_ratio = rbeta(n, 2, 5),
#'   credit_history_months = rpois(n, 60),
#'
#'   # Categorical features
#'   education = sample(c("High School", "Bachelor", "Master", "PhD"),
#'     n,
#'     replace = TRUE, prob = c(0.35, 0.40, 0.20, 0.05)
#'   ),
#'   employment = sample(c("Employed", "Self-Employed", "Unemployed", "Retired"),
#'     n,
#'     replace = TRUE, prob = c(0.60, 0.20, 0.10, 0.10)
#'   ),
#'
#'   # Binary target (default probability varies by features)
#'   target = rbinom(n, 1, 0.15)
#' )
#'
#' # Process all features with automatic algorithm selection
#' result <- obwoe(df, target = "target")
#' print(result)
#'
#' # View detailed summary
#' print(result$summary)
#'
#' # Access results for a specific feature
#' age_bins <- result$results$age
#' print(data.frame(
#'   bin = age_bins$bin,
#'   woe = round(age_bins$woe, 3),
#'   iv = round(age_bins$iv, 4),
#'   count = age_bins$count
#' ))
#'
#' # =============================================================================
#' # Example 2: Using a Specific Algorithm
#' # =============================================================================
#'
#' # Use MDLP for numerical features (entropy-based)
#' result_mdlp <- obwoe(df,
#'   target = "target",
#'   feature = c("age", "income"),
#'   algorithm = "mdlp",
#'   min_bins = 3,
#'   max_bins = 6
#' )
#'
#' cat("\nMDLP Results:\n")
#' print(result_mdlp$summary)
#'
#' # =============================================================================
#' # Example 3: Custom Control Parameters
#' # =============================================================================
#'
#' # Fine-tune algorithm behavior
#' ctrl <- control.obwoe(
#'   bin_cutoff = 0.02, # Minimum 2% per bin
#'   max_n_prebins = 30, # Allow more initial bins
#'   convergence_threshold = 1e-8
#' )
#'
#' result_custom <- obwoe(df,
#'   target = "target",
#'   feature = "debt_ratio",
#'   algorithm = "jedi",
#'   control = ctrl
#' )
#'
#' cat("\nCustom JEDI Result:\n")
#' print(result_custom$results$debt_ratio$bin)
#'
#' # =============================================================================
#' # Example 4: Comparing Multiple Algorithms
#' # =============================================================================
#'
#' algorithms <- c("jedi", "mdlp", "ewb", "mob")
#' iv_comparison <- sapply(algorithms, function(algo) {
#'   tryCatch(
#'     {
#'       res <- obwoe(df, target = "target", feature = "income", algorithm = algo)
#'       res$summary$total_iv
#'     },
#'     error = function(e) NA_real_
#'   )
#' })
#'
#' cat("\nAlgorithm Comparison (IV for 'income'):\n")
#' print(sort(iv_comparison, decreasing = TRUE))
#'
#' # =============================================================================
#' # Example 5: Feature Selection Based on IV
#' # =============================================================================
#'
#' # Process all features and select those with IV > 0.02
#' result_all <- obwoe(df, target = "target")
#'
#' strong_features <- result_all$summary[
#'   result_all$summary$total_iv >= 0.02 & !result_all$summary$error,
#'   c("feature", "total_iv", "n_bins")
#' ]
#' strong_features <- strong_features[order(-strong_features$total_iv), ]
#'
#' cat("\nFeatures with IV >= 0.02 (predictive):\n")
#' print(strong_features)
#'
#' # =============================================================================
#' # Example 6: Handling Algorithm Compatibility
#' # =============================================================================
#'
#' # MDLP only works for numerical - will fail for categorical
#' result_mixed <- obwoe(df,
#'   target = "target",
#'   algorithm = "mdlp"
#' )
#'
#' # Check for errors
#' cat("\nCompatibility check:\n")
#' print(result_mixed$summary[, c("feature", "type", "error")])
#' }
#'
#' @importFrom graphics abline axis barplot legend lines mtext par plot.new segments text title
#' @importFrom stats median quantile sd
#' @export
obwoe <- function(data,
                  target,
                  feature = NULL,
                  min_bins = 2,
                  max_bins = 7,
                  algorithm = "auto",
                  control = control.obwoe()) {
  call <- match.call()

  # Input validation
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.")
  }

  if (!is.character(target) || length(target) != 1) {
    stop("'target' must be a single character string.")
  }

  if (!target %in% names(data)) {
    stop(sprintf("Target column '%s' not found in data.", target))
  }

  # Validate feature argument
  if (!is.null(feature)) {
    if (!is.character(feature)) {
      stop("'feature' must be NULL or a character vector.")
    }
    missing_cols <- setdiff(feature, names(data))
    if (length(missing_cols) > 0) {
      stop(sprintf(
        "Feature column(s) not found: %s",
        paste(missing_cols, collapse = ", ")
      ))
    }
    feature <- setdiff(feature, target)
  } else {
    feature <- setdiff(names(data), target)
  }

  if (length(feature) == 0) {
    stop("No feature columns to process.")
  }

  # Validate bins
  min_bins <- as.integer(min_bins)
  max_bins <- as.integer(max_bins)

  if (min_bins < 2) stop("'min_bins' must be at least 2.")
  if (max_bins < min_bins) stop("'max_bins' must be >= 'min_bins'.")

  # Validate algorithm
  algorithm <- tolower(as.character(algorithm))

  # Validate control
  if (!inherits(control, "obwoe_control")) {
    if (is.list(control)) {
      control <- do.call(control.obwoe, control)
    } else {
      stop("'control' must be created by control.obwoe().")
    }
  }

  # Ensure target is integer-encoded (0, 1, ...)
  # If it's a factor (standard for tidymodels classification), convert to 0-based integers
  raw_target <- data[[target]]
  if (is.factor(raw_target)) {
    target_vec <- as.integer(raw_target) - 1L
  } else {
    target_vec <- as.integer(raw_target)
  }

  # Detect target type
  unique_targets <- sort(unique(target_vec[!is.na(target_vec)]))
  if (length(unique_targets) == 2 && all(unique_targets %in% c(0L, 1L))) {
    target_type <- "binary"
  } else if (length(unique_targets) > 1 && all(unique_targets >= 0L)) {
    # Could be binary with values other than 0/1, or true multinomial
    # We treat any non-0/1 multi-level as multinomial to be safe
    target_type <- "multinomial"
  } else {
    stop("Target must be binary (0/1) or multinomial (0,1,2,...). Factors are automatically converted.")
  }

  # Select algorithm
  if (algorithm == "auto") {
    algorithm <- if (target_type == "multinomial") "jedi_mwoe" else "jedi"
  }

  # Process each feature
  results <- list()
  summary_data <- data.frame(
    id = integer(),
    feature = character(),
    type = character(),
    algorithm = character(),
    n_bins = integer(),
    total_iv = numeric(),
    converged = logical(),
    iterations = integer(),
    error = logical(),
    stringsAsFactors = FALSE
  )

  for (col in feature) {
    feat_vec <- data[[col]]

    # Detect feature type
    if (is.numeric(feat_vec) && !is.factor(feat_vec)) {
      feat_type <- "numerical"
    } else {
      feat_type <- "categorical"
    }

    # Route to appropriate function
    result <- tryCatch(
      {
        .dispatch_algorithm(
          feat_type = feat_type,
          algorithm = algorithm,
          target_vec = target_vec,
          feat_vec = feat_vec,
          min_bins = min_bins,
          max_bins = max_bins,
          control = control
        )
      },
      error = function(e) {
        list(error = e$message)
      }
    )

    # Add metadata
    result$feature <- col
    result$type <- feat_type
    result$algorithm <- algorithm

    results[[col]] <- result

    # Build summary row
    has_error <- !is.null(result$error)

    # Calculate total_iv: use total_iv if present, otherwise sum iv vector
    total_iv_val <- NA_real_
    if (!has_error) {
      if (!is.null(result$total_iv) && length(result$total_iv) == 1) {
        total_iv_val <- result$total_iv
      } else if (!is.null(result$iv) && is.numeric(result$iv)) {
        total_iv_val <- sum(result$iv, na.rm = TRUE)
      }
    }

    summary_data <- rbind(summary_data, data.frame(
      feature = col,
      type = feat_type,
      algorithm = algorithm,
      n_bins = if (!has_error && !is.null(result$bin)) length(result$bin) else NA_integer_,
      total_iv = total_iv_val,
      converged = if (!has_error && !is.null(result$converged)) result$converged else NA,
      iterations = if (!has_error && !is.null(result$iterations)) result$iterations else NA_integer_,
      error = has_error,
      stringsAsFactors = FALSE
    ))
  }

  # Build output
  out <- list(
    results = results,
    summary = summary_data,
    target = target,
    target_type = target_type,
    n_features = length(feature),
    call = call
  )

  class(out) <- "obwoe"
  return(out)
}


#' @title Internal Algorithm Dispatcher
#' @keywords internal
.dispatch_algorithm <- function(feat_type, algorithm, target_vec, feat_vec,
                                min_bins, max_bins, control) {
  algo <- algorithm

  # Get algorithm registry
  registry <- .get_algorithm_registry()

  if (!algo %in% names(registry)) {
    stop(sprintf("Unknown algorithm: '%s'", algo))
  }

  info <- registry[[algo]]

  # Check compatibility
  if (feat_type == "numerical" && !info$numerical) {
    stop(sprintf("Algorithm '%s' does not support numerical features.", algo))
  }
  if (feat_type == "categorical" && !info$categorical) {
    stop(sprintf("Algorithm '%s' does not support categorical features.", algo))
  }

  # Prepare arguments
  args <- list(
    target = target_vec,
    feature = if (feat_type == "categorical") as.character(feat_vec) else as.numeric(feat_vec),
    min_bins = min_bins,
    max_bins = max_bins,
    bin_cutoff = control$bin_cutoff,
    max_n_prebins = control$max_n_prebins,
    convergence_threshold = control$convergence_threshold,
    max_iterations = control$max_iterations
  )

  # Add categorical-specific args
  if (feat_type == "categorical") {
    args$bin_separator <- control$bin_separator
  }

  # Get function name
  if (feat_type == "numerical") {
    fn_name <- paste0("ob_numerical_", algo)
  } else {
    fn_name <- paste0("ob_categorical_", algo)
  }

  # Special cases for function naming
  if (algo == "fast_mdlp") {
    fn_name <- "ob_numerical_fast_mdlp"
  }

  # Check if function exists
  fn <- tryCatch(get(fn_name, envir = asNamespace("OptimalBinningWoE")), error = function(e) NULL)

  if (is.null(fn)) {
    # Try alternative names
    fn_name_alt <- switch(algo,
      "jedi_mwoe" = if (feat_type == "numerical") "obn_jedi_mwoe" else "obc_jedi_mwoe",
      "fast_mdlp" = "obn_fast_mdlp",
      NULL
    )
    if (!is.null(fn_name_alt)) {
      fn <- tryCatch(get(fn_name_alt, envir = asNamespace("OptimalBinningWoE")), error = function(e) NULL)
    }
  }

  if (is.null(fn)) {
    stop(sprintf("Function '%s' not found for algorithm '%s'.", fn_name, algo))
  }

  # Filter args to match function signature
  fn_args <- names(formals(fn))
  args <- args[names(args) %in% fn_args]

  # Call function
  do.call(fn, args)
}


#' @title Get Algorithm Registry
#' @keywords internal
.get_algorithm_registry <- function() {
  list(
    # Both numerical and categorical
    cm = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    dmiv = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    dp = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    fetb = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    jedi = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    jedi_mwoe = list(numerical = TRUE, categorical = TRUE, multinomial = TRUE),
    mob = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    sketch = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),
    udt = list(numerical = TRUE, categorical = TRUE, multinomial = FALSE),

    # Categorical only
    gmb = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    ivb = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    mba = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    milp = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    sab = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    sblp = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),
    swb = list(numerical = FALSE, categorical = TRUE, multinomial = FALSE),

    # Numerical only
    bb = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    ewb = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    fast_mdlp = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    ir = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    kmb = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    ldb = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    lpdb = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    mblp = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    mdlp = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    mrblp = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    oslp = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE),
    ubsd = list(numerical = TRUE, categorical = FALSE, multinomial = FALSE)
  )
}


#' @title Control Parameters for Optimal Binning Algorithms
#'
#' @description
#' Constructs a validated list of control parameters for the \code{\link{obwoe}}
#' master interface. These parameters govern the behavior of all supported
#' binning algorithms, including convergence criteria, minimum bin sizes,
#' and optimization limits.
#'
#' @param bin_cutoff Numeric value in \eqn{(0, 1)} specifying the minimum
#'   proportion of total observations that a bin must contain. Bins with
#'   fewer observations are merged with adjacent bins. Serves as a regularization
#'   mechanism to prevent overfitting and ensure statistical stability of
#'   WoE estimates. Recommended range: 0.02 to 0.10. Default is 0.05 (5\%).
#'
#' @param max_n_prebins Integer specifying the maximum number of initial bins
#'   created before optimization. For high-cardinality categorical features,
#'   categories with similar event rates are pre-merged until this limit is
#'   reached. Higher values preserve more granularity but increase computational
#'   cost. Typical range: 10 to 50. Default is 20.
#'
#' @param convergence_threshold Numeric value specifying the tolerance for
#'   algorithm convergence. Iteration stops when the absolute change in
#'   Information Value between successive iterations falls below this threshold:
#'   \eqn{|IV_{t} - IV_{t-1}| < \epsilon}. Smaller values yield more precise
#'   solutions at higher computational cost. Typical range: \eqn{10^{-4}} to
#'   \eqn{10^{-8}}. Default is \eqn{10^{-6}}.
#'
#' @param max_iterations Integer specifying the maximum number of optimization
#'   iterations. Prevents infinite loops in degenerate cases. If the algorithm
#'   does not converge within this limit, it returns the best solution found.
#'   Typical range: 100 to 10000. Default is 1000.
#'
#' @param bin_separator Character string used to concatenate category names
#'   when multiple categories are merged into a single bin. Should be a string
#'   unlikely to appear in actual category names. Default is \code{"\%;\%"}.
#'
#' @param verbose Logical indicating whether to print progress messages during
#'   feature processing. Useful for debugging or monitoring long-running jobs.
#'   Default is \code{FALSE}.
#'
#' @param ... Additional named parameters reserved for algorithm-specific
#'   extensions. Currently unused but included for forward compatibility.
#'
#' @return An S3 object of class \code{"obwoe_control"} containing all specified
#'   parameters. This object is validated and can be passed directly to
#'   \code{\link{obwoe}}.
#'
#' @details
#' \subsection{Parameter Impact on Results}{
#'
#' \strong{bin_cutoff}: Lower values allow smaller bins, which may capture
#' subtle patterns but risk unstable WoE estimates. The variance of WoE
#' estimates increases as \eqn{1/n_i} where \eqn{n_i} is the bin size.
#' For bins with fewer than ~30 observations, consider using Laplace or
#' Bayesian smoothing (applied automatically by most algorithms).
#'
#' \strong{max_n_prebins}: Critical for categorical features with many levels.
#' If a feature has 100 categories, setting \code{max_n_prebins = 20} will
#' pre-merge similar categories into 20 groups before optimization.
#'
#' \strong{convergence_threshold}: Trade-off between precision and speed.
#' For exploratory analysis, \eqn{10^{-4}} is sufficient. For production
#' models requiring reproducibility, use \eqn{10^{-8}} or smaller.
#' }
#'
#' @seealso \code{\link{obwoe}} for the main binning interface.
#'
#' @examples
#' # Default control parameters
#' ctrl_default <- control.obwoe()
#' print(ctrl_default)
#'
#' # Conservative settings for production
#' ctrl_production <- control.obwoe(
#'   bin_cutoff = 0.03,
#'   max_n_prebins = 30,
#'   convergence_threshold = 1e-8,
#'   max_iterations = 5000
#' )
#'
#' # Aggressive settings for exploration
#' ctrl_explore <- control.obwoe(
#'   bin_cutoff = 0.01,
#'   max_n_prebins = 50,
#'   convergence_threshold = 1e-4,
#'   max_iterations = 500
#' )
#'
#' @export
control.obwoe <- function(bin_cutoff = 0.05,
                          max_n_prebins = 20,
                          convergence_threshold = 1e-6,
                          max_iterations = 1000,
                          bin_separator = "%;%",
                          verbose = FALSE,
                          ...) {
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    stop("'bin_cutoff' must be between 0 and 1.")
  }

  ctrl <- list(
    bin_cutoff = bin_cutoff,
    max_n_prebins = as.integer(max_n_prebins),
    convergence_threshold = convergence_threshold,
    max_iterations = as.integer(max_iterations),
    bin_separator = bin_separator,
    verbose = verbose,
    ...
  )

  class(ctrl) <- "obwoe_control"
  return(ctrl)
}


#' @title List Available Algorithms
#'
#' @description
#' Returns a data frame with all available binning algorithms.
#'
#' @return A data frame with algorithm information.
#'
#' @examples
#' obwoe_algorithms()
#'
#' @export
obwoe_algorithms <- function() {
  reg <- .get_algorithm_registry()

  data.frame(
    algorithm = names(reg),
    numerical = sapply(reg, `[[`, "numerical"),
    categorical = sapply(reg, `[[`, "categorical"),
    multinomial = sapply(reg, `[[`, "multinomial"),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
}


#' @title Print Method for obwoe Objects
#'
#' @description
#' Displays a concise summary of optimal binning results, including
#' the number of successfully processed features and top predictors
#' ranked by Information Value.
#'
#' @param x An object of class \code{"obwoe"}.
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns \code{x}.
#'
#' @seealso \code{\link{summary.obwoe}} for detailed statistics,
#'   \code{\link{plot.obwoe}} for visualization.
#'
#' @method print obwoe
#' @export
print.obwoe <- function(x, ...) {
  cat("Optimal Binning Weight of Evidence\n")
  cat("===================================\n\n")

  cat("Target:", x$target, "(", x$target_type, ")\n")
  cat("Features processed:", x$n_features, "\n\n")

  n_success <- sum(!x$summary$error)
  n_error <- sum(x$summary$error)

  cat("Results: ", n_success, " successful")
  if (n_error > 0) cat(", ", n_error, " errors")
  cat("\n\n")

  if (n_success > 0) {
    successful <- x$summary[!x$summary$error, ]
    ordered <- successful[order(-successful$total_iv), ]
    top_n <- min(5, nrow(ordered))

    cat("Top features by IV:\n")
    for (i in seq_len(top_n)) {
      cat(sprintf(
        "  %s: IV = %.4f (%d bins, %s)\n",
        ordered$feature[i],
        ordered$total_iv[i],
        ordered$n_bins[i],
        ordered$algorithm[i]
      ))
    }
    if (nrow(ordered) > 5) {
      cat(sprintf("  ... and %d more\n", nrow(ordered) - 5))
    }
  }

  invisible(x)
}


#' @title Summary Method for obwoe Objects
#'
#' @description
#' Generates comprehensive summary statistics for optimal binning results,
#' including predictive power classification based on established IV thresholds
#' (Siddiqi, 2006), aggregate metrics, and feature-level diagnostics.
#'
#' @param object An object of class \code{"obwoe"}.
#' @param sort_by Character string specifying the column to sort by.
#'   Options: \code{"iv"} (default), \code{"n_bins"}, \code{"feature"}.
#' @param decreasing Logical. Sort in decreasing order? Default is \code{TRUE}
#'   for IV, \code{FALSE} for feature names.
#' @param ... Additional arguments (currently ignored).
#'
#' @return An S3 object of class \code{"summary.obwoe"} containing:
#' \describe{
#'   \item{\code{feature_summary}}{Data frame with per-feature statistics including
#'     IV classification (Unpredictive/Weak/Medium/Strong/Suspicious)}
#'   \item{\code{aggregate}}{Named list of aggregate statistics:
#'     \describe{
#'       \item{\code{n_features}}{Total features processed}
#'       \item{\code{n_successful}}{Features without errors}
#'       \item{\code{n_errors}}{Features with errors}
#'       \item{\code{total_iv_sum}}{Sum of all feature IVs}
#'       \item{\code{mean_iv}}{Mean IV across features}
#'       \item{\code{median_iv}}{Median IV across features}
#'       \item{\code{mean_bins}}{Mean number of bins}
#'       \item{\code{iv_range}}{Min and max IV values}
#'     }
#'   }
#'   \item{\code{iv_distribution}}{Table of IV classification counts}
#'   \item{\code{target}}{Target column name}
#'   \item{\code{target_type}}{Target type (binary/multinomial)}
#' }
#'
#' @details
#' \subsection{IV Classification Thresholds}{
#'
#' Following Siddiqi (2006), features are classified by predictive power:
#'
#' \tabular{ll}{
#'   \strong{Classification} \tab \strong{IV Range} \cr
#'   Unpredictive \tab < 0.02 \cr
#'   Weak \tab 0.02 - 0.10 \cr
#'   Medium \tab 0.10 - 0.30 \cr
#'   Strong \tab 0.30 - 0.50 \cr
#'   Suspicious \tab > 0.50
#' }
#'
#' Features with IV > 0.50 should be examined for data leakage or
#' overfitting, as such high values are rarely observed in practice.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' @seealso \code{\link{obwoe}} for the main binning function,
#'   \code{\link{print.obwoe}}, \code{\link{plot.obwoe}}.
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' df <- data.frame(
#'   x1 = rnorm(500), x2 = rnorm(500), x3 = rnorm(500),
#'   target = rbinom(500, 1, 0.2)
#' )
#' result <- obwoe(df, target = "target")
#' summary(result)
#' }
#'
#' @method summary obwoe
#' @export
summary.obwoe <- function(object, sort_by = "iv", decreasing = TRUE, ...) {
  # Extract successful results
  summ <- object$summary
  successful <- summ[!summ$error, ]

  # Classify IV using Siddiqi thresholds
  classify_iv <- function(iv) {
    if (is.na(iv)) {
      return("Error")
    }
    if (iv < 0.02) {
      return("Unpredictive")
    }
    if (iv < 0.10) {
      return("Weak")
    }
    if (iv < 0.30) {
      return("Medium")
    }
    if (iv < 0.50) {
      return("Strong")
    }
    return("Suspicious")
  }

  summ$iv_class <- sapply(summ$total_iv, classify_iv)
  summ$iv_class <- factor(summ$iv_class,
    levels = c(
      "Unpredictive", "Weak", "Medium",
      "Strong", "Suspicious", "Error"
    )
  )

  # Sort
  sort_col <- switch(sort_by,
    "iv" = "total_iv",
    "n_bins" = "n_bins",
    "feature" = "feature",
    "total_iv"
  )

  if (sort_by == "feature") {
    decreasing <- FALSE
  }

  ord <- order(summ[[sort_col]], decreasing = decreasing, na.last = TRUE)
  summ <- summ[ord, ]

  # Aggregate statistics
  if (nrow(successful) > 0) {
    ivs <- successful$total_iv
    bins <- successful$n_bins

    aggregate <- list(
      n_features = nrow(summ),
      n_successful = nrow(successful),
      n_errors = sum(summ$error),
      total_iv_sum = sum(ivs, na.rm = TRUE),
      mean_iv = mean(ivs, na.rm = TRUE),
      median_iv = median(ivs, na.rm = TRUE),
      sd_iv = sd(ivs, na.rm = TRUE),
      mean_bins = mean(bins, na.rm = TRUE),
      iv_range = c(min = min(ivs, na.rm = TRUE), max = max(ivs, na.rm = TRUE))
    )
  } else {
    aggregate <- list(
      n_features = nrow(summ),
      n_successful = 0,
      n_errors = sum(summ$error),
      total_iv_sum = NA_real_,
      mean_iv = NA_real_,
      median_iv = NA_real_,
      sd_iv = NA_real_,
      mean_bins = NA_real_,
      iv_range = c(min = NA_real_, max = NA_real_)
    )
  }

  # IV distribution
  iv_dist <- table(summ$iv_class)

  result <- list(
    feature_summary = summ,
    aggregate = aggregate,
    iv_distribution = iv_dist,
    target = object$target,
    target_type = object$target_type
  )

  class(result) <- "summary.obwoe"
  return(result)
}


#' @method print summary.obwoe
#' @export
print.summary.obwoe <- function(x, ...) {
  cat("Summary: Optimal Binning Weight of Evidence\n")
  cat("============================================\n\n")

  cat("Target:", x$target, "(", x$target_type, ")\n\n")

  # Aggregate statistics
  cat("Aggregate Statistics:\n")
  cat(sprintf(
    "  Features: %d total, %d successful, %d errors\n",
    x$aggregate$n_features,
    x$aggregate$n_successful,
    x$aggregate$n_errors
  ))

  if (x$aggregate$n_successful > 0) {
    cat(sprintf("  Total IV: %.4f\n", x$aggregate$total_iv_sum))
    cat(sprintf(
      "  Mean IV: %.4f (SD: %.4f)\n",
      x$aggregate$mean_iv, x$aggregate$sd_iv
    ))
    cat(sprintf("  Median IV: %.4f\n", x$aggregate$median_iv))
    cat(sprintf(
      "  IV Range: [%.4f, %.4f]\n",
      x$aggregate$iv_range["min"], x$aggregate$iv_range["max"]
    ))
    cat(sprintf("  Mean Bins: %.1f\n", x$aggregate$mean_bins))
  }

  # IV distribution
  cat("\nIV Classification (Siddiqi, 2006):\n")
  for (lev in levels(x$feature_summary$iv_class)) {
    count <- x$iv_distribution[lev]
    if (!is.na(count) && count > 0) {
      cat(sprintf("  %-12s: %d features\n", lev, count))
    }
  }

  # Feature table
  cat("\nFeature Details:\n")
  print(x$feature_summary[, c("feature", "type", "n_bins", "total_iv", "iv_class")],
    row.names = FALSE
  )

  invisible(x)
}


#' @title Plot Method for obwoe Objects
#'
#' @description
#' Creates publication-quality visualizations of optimal binning results.
#' Supports multiple plot types including IV ranking charts, WoE profiles,
#' and bin distribution plots. All plots follow credit scoring visualization
#' conventions.
#'
#' @param x An object of class \code{"obwoe"}.
#' @param type Character string specifying the plot type:
#'   \describe{
#'     \item{\code{"iv"}}{Information Value ranking bar chart (default)}
#'     \item{\code{"woe"}}{Weight of Evidence profile for selected features}
#'     \item{\code{"bins"}}{Bin distribution (count and event rate)}
#'   }
#' @param feature Character vector of feature names to plot (for \code{"woe"}
#'   and \code{"bins"} types). If \code{NULL}, uses top 6 features by IV.
#' @param top_n Integer. For \code{"iv"} type, number of top features to display.
#'   Default is 15. Set to \code{NULL} to display all.
#' @param show_threshold Logical. For \code{"iv"} type, draw horizontal lines
#'   at IV thresholds (0.02, 0.10, 0.30)? Default is \code{TRUE}.
#' @param ... Additional arguments passed to base plotting functions.
#'
#' @return Invisibly returns \code{NULL}. Called for side effect (plotting).
#'
#' @details
#' \subsection{Plot Types}{
#'
#' \strong{IV Ranking (\code{type = "iv"})}:
#' Horizontal bar chart showing features ranked by Information Value.
#' Colors indicate predictive power classification:
#' \itemize{
#'   \item Gray: IV < 0.02 (Unpredictive)
#'   \item Yellow: 0.02 <= IV < 0.10 (Weak)
#'   \item Orange: 0.10 <= IV < 0.30 (Medium)
#'   \item Green: 0.30 <= IV < 0.50 (Strong)
#'   \item Red: IV >= 0.50 (Suspicious)
#' }
#'
#' \strong{WoE Profile (\code{type = "woe"})}:
#' Bar chart showing Weight of Evidence values for each bin.
#' Positive WoE indicates higher-than-average event rate;
#' negative WoE indicates lower-than-average event rate.
#' Monotonic WoE patterns are generally preferred for interpretability.
#'
#' \strong{Bin Distribution (\code{type = "bins"})}:
#' Dual-axis plot showing observation counts (bars) and event rates (line).
#' Useful for diagnosing bin quality and class imbalance.
#' }
#'
#' @references
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring
#' and Its Applications. \emph{SIAM Monographs on Mathematical Modeling
#' and Computation}. \doi{10.1137/1.9780898718317}
#'
#' @seealso \code{\link{obwoe}}, \code{\link{summary.obwoe}}.
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' df <- data.frame(
#'   age = rnorm(500, 40, 15),
#'   income = rgamma(500, 2, 0.0001),
#'   score = rnorm(500, 600, 100),
#'   target = rbinom(500, 1, 0.2)
#' )
#' result <- obwoe(df, target = "target")
#'
#' # IV ranking chart
#' plot(result, type = "iv")
#'
#' # WoE profile for specific feature
#' plot(result, type = "woe", feature = "age")
#'
#' # Bin distribution
#' plot(result, type = "bins", feature = "income")
#' }
#'
#' @method plot obwoe
#' @export
plot.obwoe <- function(x, type = c("iv", "woe", "bins"),
                       feature = NULL, top_n = 15,
                       show_threshold = TRUE, ...) {
  type <- match.arg(type)

  # Filter successful results
  summ <- x$summary[!x$summary$error, ]

  if (nrow(summ) == 0) {
    message("No successful binning results to plot.")
    return(invisible(NULL))
  }

  if (type == "iv") {
    .plot_iv_ranking(summ, top_n, show_threshold, ...)
  } else if (type == "woe") {
    .plot_woe_profile(x, feature, ...)
  } else if (type == "bins") {
    .plot_bin_distribution(x, feature, ...)
  }

  invisible(NULL)
}


#' @keywords internal
.plot_iv_ranking <- function(summ, top_n, show_threshold, ...) {
  # Sort by IV
  summ <- summ[order(summ$total_iv, decreasing = TRUE), ]

  # Limit to top_n
  if (!is.null(top_n) && nrow(summ) > top_n) {
    summ <- summ[1:top_n, ]
  }

  # Reverse for horizontal barplot (bottom to top)
  summ <- summ[nrow(summ):1, ]

  # Color by IV classification
  get_iv_color <- function(iv) {
    if (is.na(iv)) {
      return("gray80")
    }
    if (iv < 0.02) {
      return("gray70")
    }
    if (iv < 0.10) {
      return("#FFC107")
    } # Yellow
    if (iv < 0.30) {
      return("#FF9800")
    } # Orange
    if (iv < 0.50) {
      return("#4CAF50")
    } # Green
    return("#F44336") # Red (suspicious)
  }
  colors <- sapply(summ$total_iv, get_iv_color)

  # Plot
  old_par <- par(mar = c(4, 10, 3, 2))
  on.exit(par(old_par))

  bp <- barplot(summ$total_iv,
    horiz = TRUE,
    names.arg = summ$feature,
    las = 1,
    col = colors,
    border = NA,
    xlab = "Information Value (IV)",
    main = "Feature Importance by Information Value",
    xlim = c(0, max(summ$total_iv, na.rm = TRUE) * 1.1),
    ...
  )

  # Add threshold lines
  if (show_threshold) {
    max_iv <- max(summ$total_iv, na.rm = TRUE)
    if (max_iv > 0.02) abline(v = 0.02, lty = 2, col = "gray50")
    if (max_iv > 0.10) abline(v = 0.10, lty = 2, col = "gray50")
    if (max_iv > 0.30) abline(v = 0.30, lty = 2, col = "gray50")
  }

  # Add IV values
  text(summ$total_iv + max(summ$total_iv, na.rm = TRUE) * 0.02,
    bp,
    labels = sprintf("%.3f", summ$total_iv),
    cex = 0.8, adj = 0
  )
}


#' @keywords internal
.plot_woe_profile <- function(x, feature, ...) {
  if (is.null(feature)) {
    # Select top features by IV
    summ <- x$summary[!x$summary$error, ]
    summ <- summ[order(-summ$total_iv), ]
    feature <- summ$feature[1:min(6, nrow(summ))]
  }

  n_feat <- length(feature)

  if (n_feat > 6) {
    message("Showing first 6 features only.")
    feature <- feature[1:6]
    n_feat <- 6
  }

  # Setup multi-panel
  if (n_feat > 1) {
    old_par <- par(mfrow = c(ceiling(n_feat / 2), 2), mar = c(4, 4, 3, 1))
    on.exit(par(old_par))
  } else {
    old_par <- par(mar = c(4, 4, 3, 1))
    on.exit(par(old_par))
  }

  for (feat in feature) {
    res <- x$results[[feat]]

    if (is.null(res) || !is.null(res$error)) {
      plot.new()
      title(main = paste(feat, "(Error)"))
      next
    }

    woe <- res$woe
    bins <- res$bin

    # Truncate long bin names
    bins <- substr(bins, 1, 15)

    # Colors based on WoE sign
    colors <- ifelse(woe >= 0, "#4CAF50", "#F44336")

    bp <- barplot(woe,
      names.arg = bins,
      las = 2,
      col = colors,
      border = NA,
      ylab = "Weight of Evidence",
      main = feat,
      cex.names = 0.7,
      ...
    )

    abline(h = 0, lty = 1, col = "gray30")
  }
}


#' @keywords internal
.plot_bin_distribution <- function(x, feature, ...) {
  if (is.null(feature)) {
    summ <- x$summary[!x$summary$error, ]
    summ <- summ[order(-summ$total_iv), ]
    feature <- summ$feature[1]
  }

  if (length(feature) > 1) {
    message("Showing first feature only for bin distribution.")
    feature <- feature[1]
  }

  res <- x$results[[feature]]

  if (is.null(res) || !is.null(res$error)) {
    message("Feature '", feature, "' has no valid results.")
    return(invisible(NULL))
  }

  bins <- res$bin
  counts <- res$count
  count_pos <- res$count_pos
  event_rate <- count_pos / counts

  # Truncate bin names
  bins <- substr(bins, 1, 12)

  old_par <- par(mar = c(6, 4, 4, 4))
  on.exit(par(old_par))

  # Bar plot for counts
  bp <- barplot(counts,
    names.arg = bins,
    las = 2,
    col = "#2196F3",
    border = NA,
    ylab = "Count",
    main = paste("Bin Distribution:", feature),
    cex.names = 0.8,
    ...
  )

  # Overlay event rate
  par(new = TRUE)
  plot(bp, event_rate,
    type = "b",
    pch = 19,
    col = "#F44336",
    axes = FALSE,
    xlab = "",
    ylab = "",
    ylim = c(0, max(event_rate, na.rm = TRUE) * 1.2)
  )

  axis(4, col = "#F44336", col.axis = "#F44336")
  mtext("Event Rate", side = 4, line = 2.5, col = "#F44336")

  # Legend
  legend("topright",
    legend = c("Count", "Event Rate"),
    fill = c("#2196F3", NA),
    border = NA,
    pch = c(NA, 19),
    col = c(NA, "#F44336"),
    bty = "n",
    cex = 0.8
  )
}


#' @title Apply Weight of Evidence Transformations to New Data
#'
#' @description
#' Applies the binning and Weight of Evidence (WoE) transformations learned by
#' \code{\link{obwoe}} to new data. This is the scoring function for deploying
#' WoE-based models in production. For each feature, the function assigns
#' observations to bins and maps them to their corresponding WoE values.
#'
#' @param data A \code{data.frame} containing the features to transform.
#'   Must include all features present in the \code{obj} results. The target
#'   column is optional; if present, it will be included in the output.
#' @param obj An object of class \code{"obwoe"} returned by \code{\link{obwoe}}.
#' @param suffix_bin Character string suffix for bin columns.
#'   Default is \code{"_bin"}.
#' @param suffix_woe Character string suffix for WoE columns.
#'   Default is \code{"_woe"}.
#' @param keep_original Logical. If \code{TRUE} (default), include the original
#'   feature columns in the output. If \code{FALSE}, only bin and WoE columns
#'   are returned.
#' @param na_woe Numeric value to assign when an observation cannot be mapped
#'   to a bin (e.g., new categories not seen during training). Default is 0.
#'
#' @return A \code{data.frame} containing:
#' \describe{
#'   \item{\code{target}}{The target column (if present in \code{data})}
#'   \item{\code{<feature>}}{Original feature values (if \code{keep_original = TRUE})}
#'   \item{\code{<feature>_bin}}{Assigned bin label for each observation}
#'   \item{\code{<feature>_woe}}{Weight of Evidence value for the assigned bin}
#' }
#'
#' @details
#' \subsection{Bin Assignment Logic}{
#'
#' \strong{Numerical Features}:
#' Observations are assigned to bins based on cutpoints stored in the
#' \code{obwoe} object. The \code{cut()} function is used with intervals
#' \eqn{(a_i, a_{i+1}]} where \eqn{a_0 = -\infty} and \eqn{a_k = +\infty}.
#'
#' \strong{Categorical Features}:
#' Categories are matched directly to bin labels. Categories not seen
#' during training are assigned \code{NA} for bin and \code{na_woe} for WoE.
#' }
#'
#' \subsection{Production Deployment}{
#'
#' For production scoring, it is recommended to:
#' \enumerate{
#'   \item Train the binning model using \code{obwoe()} on the training set
#'   \item Save the fitted object with \code{saveRDS()}
#'   \item Load and apply using \code{obwoe_apply()} on new data
#' }
#'
#' The WoE-transformed features can be used directly as inputs to logistic
#' regression or other linear models, enabling interpretable credit scorecards.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' @seealso \code{\link{obwoe}} for fitting the binning model,
#'   \code{\link{summary.obwoe}} for model diagnostics.
#'
#' @examples
#' \donttest{
#' # =============================================================================
#' # Example 1: Basic Usage - Train and Apply
#' # =============================================================================
#' set.seed(42)
#' n <- 1000
#'
#' # Training data
#' train_df <- data.frame(
#'   age = rnorm(n, 40, 15),
#'   income = exp(rnorm(n, 10, 0.8)),
#'   education = sample(c("HS", "BA", "MA", "PhD"), n, replace = TRUE),
#'   target = rbinom(n, 1, 0.15)
#' )
#'
#' # Fit binning model
#' model <- obwoe(train_df, target = "target")
#'
#' # New data for scoring (could be validation/test set)
#' new_df <- data.frame(
#'   age = c(25, 45, 65),
#'   income = c(20000, 50000, 80000),
#'   education = c("HS", "MA", "PhD")
#' )
#'
#' # Apply transformations
#' scored <- obwoe_apply(new_df, model)
#' print(scored)
#'
#' # Use WoE features for downstream modeling
#' woe_cols <- grep("_woe$", names(scored), value = TRUE)
#' print(woe_cols)
#'
#' # =============================================================================
#' # Example 2: Without Original Features
#' # =============================================================================
#'
#' scored_compact <- obwoe_apply(new_df, model, keep_original = FALSE)
#' print(scored_compact)
#' }
#'
#' @export
obwoe_apply <- function(data,
                        obj,
                        suffix_bin = "_bin",
                        suffix_woe = "_woe",
                        keep_original = TRUE,
                        na_woe = 0) {
  # Validate inputs
  if (!is.data.frame(data)) {
    stop("'data' must be a data.frame.")
  }

  if (!inherits(obj, "obwoe")) {
    stop("'obj' must be an object of class 'obwoe' from obwoe().")
  }

  # Get successful features from the model
  successful <- obj$summary[!obj$summary$error, "feature"]

  if (length(successful) == 0) {
    stop("No successful binning results in 'obj'.")
  }

  # Check which features are available in data
  available <- intersect(successful, names(data))

  if (length(available) == 0) {
    stop("None of the binned features are present in 'data'.")
  }

  missing <- setdiff(successful, names(data))
  if (length(missing) > 0) {
    warning(sprintf(
      "Features not in data (skipped): %s",
      paste(missing, collapse = ", ")
    ))
  }

  # Initialize output data.frame
  n <- nrow(data)
  result <- data.frame(row.names = seq_len(n))

  # Include target if present
  target_name <- obj$target
  if (target_name %in% names(data)) {
    result[[target_name]] <- data[[target_name]]
  }

  # Process each feature
  for (feat in available) {
    res <- obj$results[[feat]]
    feat_type <- res$type

    feat_vec <- data[[feat]]

    # Add original if requested
    if (keep_original) {
      result[[feat]] <- feat_vec
    }

    # Get binning info
    bins <- res$bin
    woe <- res$woe
    names(woe) <- bins

    bin_col <- paste0(feat, suffix_bin)
    woe_col <- paste0(feat, suffix_woe)

    if (feat_type == "numerical") {
      # Numerical: use cutpoints
      cutpoints <- res$cutpoints

      if (is.null(cutpoints) || length(cutpoints) == 0) {
        # No cutpoints - single bin
        result[[bin_col]] <- bins[1]
        result[[woe_col]] <- woe[1]
      } else {
        # Create breaks including -Inf and Inf
        breaks <- c(-Inf, cutpoints, Inf)

        # Cut into bins
        bin_idx <- cut(as.numeric(feat_vec),
          breaks = breaks,
          labels = FALSE,
          include.lowest = TRUE,
          right = TRUE
        )

        result[[bin_col]] <- bins[bin_idx]
        result[[woe_col]] <- woe[bin_idx]

        # Handle NAs
        result[[woe_col]][is.na(result[[woe_col]])] <- na_woe
      }
    } else {
      # Categorical: direct mapping
      feat_char <- as.character(feat_vec)

      # Build category-to-bin mapping
      # Parse bins which may contain merged categories (e.g., "A%;%B")
      cat_to_bin <- list()
      cat_to_woe <- list()

      for (i in seq_along(bins)) {
        bin_label <- bins[i]
        # Split by separator (%;%)
        cats <- strsplit(bin_label, "%;%", fixed = TRUE)[[1]]
        for (cat in cats) {
          cat <- trimws(cat)
          cat_to_bin[[cat]] <- bin_label
          cat_to_woe[[cat]] <- woe[i]
        }
      }

      # Map each observation
      mapped_bin <- sapply(feat_char, function(x) {
        if (is.na(x)) {
          return(NA_character_)
        }
        if (x %in% names(cat_to_bin)) cat_to_bin[[x]] else NA_character_
      }, USE.NAMES = FALSE)

      mapped_woe <- sapply(feat_char, function(x) {
        if (is.na(x)) {
          return(na_woe)
        }
        if (x %in% names(cat_to_woe)) cat_to_woe[[x]] else na_woe
      }, USE.NAMES = FALSE)

      result[[bin_col]] <- mapped_bin
      result[[woe_col]] <- as.numeric(mapped_woe)
    }
  }

  return(result)
}


#' @title Gains Table Statistics for Credit Risk Scorecard Evaluation
#'
#' @description
#' Computes a comprehensive gains table (also known as a lift table or
#' decile analysis) for evaluating the discriminatory power of credit scoring
#' models and optimal binning transformations. The gains table is a fundamental
#' tool in credit risk management for model validation, cutoff selection,
#' and regulatory reporting (Basel II/III, IFRS 9).
#'
#' This function accepts three types of input:
#' \enumerate{
#'   \item An \code{"obwoe"} object from \code{\link{obwoe}} (uses stored binning)
#'   \item A \code{data.frame} from \code{\link{obwoe_apply}} (uses bin/WoE columns)
#'   \item Any \code{data.frame} with a grouping variable (e.g., score deciles)
#' }
#'
#' @param obj Input object: an \code{"obwoe"} object, a \code{data.frame} from
#'   \code{\link{obwoe_apply}}, or any \code{data.frame} containing a grouping
#'   variable and target values.
#' @param target Integer vector of binary target values (0/1) or the name of
#'   the target column in \code{obj}. Required for \code{data.frame} inputs.
#'   For \code{"obwoe"} objects, the target is extracted automatically.
#' @param feature Character string specifying the feature/variable to analyze.
#'   For \code{"obwoe"} objects: defaults to the feature with highest IV.
#'   For \code{data.frame} objects: can be any column name representing groups
#'   (e.g., \code{"age_bin"}, \code{"age_woe"}, \code{"score_decile"}).
#' @param use_column Character string specifying which column type to use when
#'   \code{obj} is a \code{data.frame} from \code{\link{obwoe_apply}}:
#'   \describe{
#'     \item{\code{"bin"}}{Use the \code{<feature>_bin} column (default)}
#'     \item{\code{"woe"}}{Use the \code{<feature>_woe} column (groups by WoE values)}
#'     \item{\code{"auto"}}{Automatically detect: use \code{_bin} if available}
#'     \item{\code{"direct"}}{Use the \code{feature} column name directly (for any variable)}
#'   }
#' @param sort_by Character string specifying sort order for bins:
#'   \describe{
#'     \item{\code{"woe"}}{Descending WoE (highest risk first) - default}
#'     \item{\code{"event_rate"}}{Descending event rate}
#'     \item{\code{"bin"}}{Alphabetical/natural order}
#'   }
#' @param n_groups Integer. For continuous variables (e.g., scores), the number
#'   of groups (deciles) to create. Default is \code{NULL} (use existing groups).
#'   Set to 10 for standard decile analysis.
#'
#' @return An S3 object of class \code{"obwoe_gains"} containing:
#' \describe{
#'   \item{\code{table}}{Data frame with 18 statistics per bin (see Details)}
#'   \item{\code{metrics}}{Named list of global performance metrics:
#'     \describe{
#'       \item{\code{ks}}{Kolmogorov-Smirnov statistic (\%)}
#'       \item{\code{gini}}{Gini coefficient (\%)}
#'       \item{\code{auc}}{Area Under ROC Curve}
#'       \item{\code{total_iv}}{Total Information Value}
#'       \item{\code{ks_bin}}{Bin where maximum KS occurs}
#'     }
#'   }
#'   \item{\code{feature}}{Feature/variable name analyzed}
#'   \item{\code{n_bins}}{Number of bins/groups}
#'   \item{\code{n_obs}}{Total observations}
#'   \item{\code{event_rate}}{Overall event rate}
#' }
#'
#' @details
#' \subsection{Gains Table Construction}{
#'
#' The gains table is constructed by:
#' \enumerate{
#'   \item Sorting observations by risk score or WoE (highest risk first)
#'   \item Grouping into bins (pre-defined or created via quantiles)
#'   \item Computing bin-level and cumulative statistics
#' }
#'
#' The table enables assessment of model rank-ordering ability: a well-calibrated
#' model should show monotonically increasing event rates as risk score increases.
#' }
#'
#' \subsection{Bin-Level Statistics (18 metrics)}{
#'
#' \tabular{lll}{
#'   \strong{Column} \tab \strong{Formula} \tab \strong{Description} \cr
#'   \code{bin} \tab - \tab Bin label or interval \cr
#'   \code{count} \tab \eqn{n_i} \tab Total observations in bin \cr
#'   \code{count_pct} \tab \eqn{n_i / N} \tab Proportion of total population \cr
#'   \code{pos_count} \tab \eqn{n_{i,1}} \tab Event count (Bad, target=1) \cr
#'   \code{neg_count} \tab \eqn{n_{i,0}} \tab Non-event count (Good, target=0) \cr
#'   \code{pos_rate} \tab \eqn{n_{i,1} / n_i} \tab Event rate (Bad rate) in bin \cr
#'   \code{neg_rate} \tab \eqn{n_{i,0} / n_i} \tab Non-event rate (Good rate) \cr
#'   \code{pos_pct} \tab \eqn{n_{i,1} / N_1} \tab Distribution of events \cr
#'   \code{neg_pct} \tab \eqn{n_{i,0} / N_0} \tab Distribution of non-events \cr
#'   \code{odds} \tab \eqn{n_{i,1} / n_{i,0}} \tab Odds of event \cr
#'   \code{log_odds} \tab \eqn{\ln(\text{odds})} \tab Log-odds (logit) \cr
#'   \code{woe} \tab \eqn{\ln(p_i / q_i)} \tab Weight of Evidence \cr
#'   \code{iv} \tab \eqn{(p_i - q_i) \cdot WoE_i} \tab Information Value contribution \cr
#'   \code{cum_pos_pct} \tab \eqn{\sum_{j \le i} p_j} \tab Cumulative events captured \cr
#'   \code{cum_neg_pct} \tab \eqn{\sum_{j \le i} q_j} \tab Cumulative non-events \cr
#'   \code{ks} \tab \eqn{|F_1(i) - F_0(i)|} \tab KS statistic at bin \cr
#'   \code{lift} \tab \eqn{\text{pos\_rate} / \bar{p}} \tab Lift over random \cr
#'   \code{capture_rate} \tab \eqn{cum\_pos\_pct} \tab Cumulative capture rate
#' }
#' }
#'
#' \subsection{Global Performance Metrics}{
#'
#' \strong{Kolmogorov-Smirnov (KS) Statistic}:
#' Maximum absolute difference between cumulative distributions of events
#' and non-events. Measures the model's ability to separate populations.
#'
#' \deqn{KS = \max_i |F_1(i) - F_0(i)|}
#'
#' \tabular{ll}{
#'   \strong{KS Range} \tab \strong{Interpretation} \cr
#'   < 20\% \tab Poor discrimination \cr
#'   20-40\% \tab Acceptable \cr
#'   40-60\% \tab Good \cr
#'   60-75\% \tab Very good \cr
#'   > 75\% \tab Excellent (verify for data leakage)
#' }
#'
#' \strong{Gini Coefficient}:
#' Measure of inequality between event and non-event distributions.
#' Equivalent to 2*AUC - 1, representing the area between the Lorenz
#' curve and the line of equality.
#'
#' \deqn{Gini = 2 \times AUC - 1}
#'
#' \strong{Area Under ROC Curve (AUC)}:
#' Probability that a randomly chosen event is ranked higher than a
#' randomly chosen non-event. Computed via the trapezoidal rule.
#'
#' \strong{Total Information Value (IV)}:
#' Sum of IV contributions across all bins. See \code{\link{obwoe}} for
#' interpretation guidelines.
#' }
#'
#' \subsection{Use Cases}{
#'
#' \strong{Model Validation}:
#' Verify rank-ordering (monotonic event rates) and acceptable KS/Gini.
#'
#' \strong{Cutoff Selection}:
#' Identify the bin where the model provides optimal separation for
#' business rules (e.g., auto-approve above score X).
#'
#' \strong{Population Stability}:
#' Compare gains tables over time to detect model drift.
#'
#' \strong{Regulatory Reporting}:
#' Generate metrics required by Basel II/III and IFRS 9 frameworks.
#' }
#'
#' @references
#' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' \doi{10.1002/9781119201731}
#'
#' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit Scoring
#' and Its Applications. \emph{SIAM Monographs on Mathematical Modeling
#' and Computation}. \doi{10.1137/1.9780898718317}
#'
#' Anderson, R. (2007). The Credit Scoring Toolkit: Theory and Practice
#' for Retail Credit Risk Management. \emph{Oxford University Press}.
#'
#' Hand, D. J., & Henley, W. E. (1997). Statistical Classification Methods
#' in Consumer Credit Scoring: A Review. \emph{Journal of the Royal
#' Statistical Society: Series A}, 160(3), 523-541.
#' \doi{10.1111/j.1467-985X.1997.00078.x}
#'
#' @seealso
#' \code{\link{obwoe}} for optimal binning,
#' \code{\link{obwoe_apply}} for scoring new data,
#' \code{\link{plot.obwoe_gains}} for visualization (cumulative gains, KS, lift).
#'
#' @examples
#' \donttest{
#' # =============================================================================
#' # Example 1: From obwoe Object (Standard Usage)
#' # =============================================================================
#' set.seed(42)
#' n <- 1000
#' df <- data.frame(
#'   age = rnorm(n, 40, 15),
#'   income = exp(rnorm(n, 10, 0.8)),
#'   score = rnorm(n, 600, 100),
#'   target = rbinom(n, 1, 0.15)
#' )
#'
#' model <- obwoe(df, target = "target")
#' gains <- obwoe_gains(model, feature = "age")
#' print(gains)
#'
#' # Access metrics
#' cat("KS:", gains$metrics$ks, "%\n")
#' cat("Gini:", gains$metrics$gini, "%\n")
#'
#' # =============================================================================
#' # Example 2: From obwoe_apply Output - Using Bin Column
#' # =============================================================================
#' scored <- obwoe_apply(df, model)
#'
#' # Default: uses age_bin column
#' gains_bin <- obwoe_gains(scored,
#'   target = df$target, feature = "age",
#'   use_column = "bin"
#' )
#'
#' # =============================================================================
#' # Example 3: From obwoe_apply Output - Using WoE Column
#' # =============================================================================
#' # Group by WoE values (continuous analysis)
#' gains_woe <- obwoe_gains(scored,
#'   target = df$target, feature = "age",
#'   use_column = "woe", n_groups = 5
#' )
#'
#' # =============================================================================
#' # Example 4: Any Variable - Score Decile Analysis
#' # =============================================================================
#' # Create score deciles manually
#' df$score_decile <- cut(df$score,
#'   breaks = quantile(df$score, probs = seq(0, 1, 0.1)),
#'   include.lowest = TRUE, labels = 1:10
#' )
#'
#' # Analyze score deciles directly
#' gains_score <- obwoe_gains(df,
#'   target = "target", feature = "score_decile",
#'   use_column = "direct"
#' )
#' print(gains_score)
#'
#' # =============================================================================
#' # Example 5: Automatic Decile Creation
#' # =============================================================================
#' # Use n_groups to automatically create quantile groups
#' gains_auto <- obwoe_gains(df,
#'   target = "target", feature = "score",
#'   use_column = "direct", n_groups = 10
#' )
#' }
#' @importFrom utils head tail
#' @export
obwoe_gains <- function(obj,
                        target = NULL,
                        feature = NULL,
                        use_column = c("auto", "bin", "woe", "direct"),
                        sort_by = c("id", "woe", "event_rate", "bin"),
                        n_groups = NULL) {
  use_column <- match.arg(use_column)
  sort_by <- match.arg(sort_by)

  # ------------------------------------------------------------------------#
  # 1. Input Validation and Data Extraction
  # ------------------------------------------------------------------------#

  if (inherits(obj, "obwoe")) {
    # Case A: Input is an 'obwoe' object (training result)
    # ---------------------------------------------------#
    if (is.null(feature)) {
      # Default: Select the feature with the highest Total IV
      summ <- obj$summary[!obj$summary$error, ]
      feature <- summ$feature[which.max(summ$total_iv)]
    }

    res <- obj$results[[feature]]

    if (is.null(res) || !is.null(res$error)) {
      stop(sprintf("Feature '%s' has no valid binning results in the provided object.", feature))
    }

    # Extract pre-calculated bin vectors
    id <- res$id
    bins <- res$bin
    woe <- res$woe
    counts <- res$count
    pos_counts <- res$count_pos
    neg_counts <- res$count_neg

    # Pass to builder (sorting happens inside)
    gt <- .build_gains_table(bins, counts, pos_counts, neg_counts, woe, sort_by, id)
  } else if (is.data.frame(obj)) {
    # Case B: Input is a data.frame (e.g., from obwoe_apply or bake)
    # -------------------------------------------------------------#
    if (is.null(target)) {
      stop("The 'target' argument is required when 'obj' is a data.frame.")
    }

    # Resolve target vector
    if (is.character(target) && length(target) == 1 && target %in% names(obj)) {
      target_vec <- as.integer(obj[[target]])
    } else {
      target_vec <- as.integer(target)
    }

    if (length(target_vec) != nrow(obj)) {
      stop("The 'target' vector length must match the number of rows in 'obj'.")
    }

    # Resolve feature name automatically if NULL
    if (is.null(feature)) {
      # Look for columns ending in _bin or _woe
      bin_cols <- grep("_bin$", names(obj), value = TRUE)
      woe_cols <- grep("_woe$", names(obj), value = TRUE)

      if (length(bin_cols) > 0) {
        feature <- sub("_bin$", "", bin_cols[1])
      } else if (length(woe_cols) > 0) {
        feature <- sub("_woe$", "", woe_cols[1])
      } else {
        stop("No '_bin' or '_woe' columns found. Please specify 'feature' and 'use_column'.")
      }
    }

    # Determine grouping column and WoE source
    group_col <- NULL
    woe_source <- NULL

    if (use_column == "direct") {
      group_col <- feature
      woe_source <- NULL # Will calculate empirical WoE
    } else if (use_column == "woe") {
      group_col <- paste0(feature, "_woe")
      woe_source <- "woe" # Use the grouping column itself as WoE
    } else if (use_column == "bin") {
      group_col <- paste0(feature, "_bin")
      # Try to find corresponding WoE column
      check_woe <- paste0(feature, "_woe")
      woe_source <- if (check_woe %in% names(obj)) obj[[check_woe]] else NULL
    } else {
      # "auto": Try _bin, then _woe, then direct
      c_bin <- paste0(feature, "_bin")
      c_woe <- paste0(feature, "_woe")

      if (c_bin %in% names(obj)) {
        group_col <- c_bin
        woe_source <- if (c_woe %in% names(obj)) obj[[c_woe]] else NULL
      } else if (c_woe %in% names(obj)) {
        group_col <- c_woe
        woe_source <- "woe"
      } else if (feature %in% names(obj)) {
        group_col <- feature
        woe_source <- NULL
      } else {
        stop(sprintf("Could not automatically locate columns for feature '%s'.", feature))
      }
    }

    # Validate column existence
    if (!group_col %in% names(obj)) {
      if (feature %in% names(obj)) {
        group_col <- feature
      } else {
        stop(sprintf("Grouping column '%s' not found in data.", group_col))
      }
    }

    group_vec <- obj[[group_col]]

    # Quantile Grouping (Deciles/Percentiles) if requested
    if (!is.null(n_groups) && is.numeric(group_vec)) {
      probs <- seq(0, 1, length.out = n_groups + 1)
      # Use unique() to handle ties (e.g., zero-inflated data)
      breaks <- unique(stats::quantile(group_vec, probs = probs, na.rm = TRUE))

      if (length(breaks) < 2) {
        warning("Insufficient unique values to create groups. Treating as discrete.")
        group_vec <- as.character(group_vec)
      } else {
        # Create ordered factor for quantiles
        group_vec <- cut(group_vec,
          breaks = breaks,
          include.lowest = TRUE,
          labels = paste0("G", formatC(1:(length(breaks) - 1), width = 2, flag = "0"))
        )
      }
    }

    # -------------------------------------------------------------#
    # 2. Aggregation (Observation Level -> Bin Level)
    # -------------------------------------------------------------#

    # Ensure factor levels are respected if present, otherwise sort unique values
    if (is.factor(group_vec)) {
      bins <- levels(group_vec)
      # Filter out unused levels to avoid rows with 0 counts
      bins <- bins[bins %in% unique(as.character(group_vec[!is.na(group_vec)]))]
    } else {
      bins <- sort(unique(as.character(group_vec[!is.na(group_vec)])))
    }

    # Convert to factor for robust tapply aggregation
    f_bins <- factor(as.character(group_vec), levels = bins)

    # Vectorized aggregation
    counts <- as.vector(tapply(rep(1, length(f_bins)), f_bins, sum, default = 0))
    pos_counts <- as.vector(tapply(target_vec, f_bins, function(x) sum(x == 1, na.rm = TRUE), default = 0))
    neg_counts <- counts - pos_counts

    # Resolve WoE per bin
    if (identical(woe_source, "woe")) {
      # The grouping variable itself is the WoE (numeric)
      woe <- as.numeric(bins)
    } else if (!is.null(woe_source) && is.numeric(woe_source)) {
      # Average the auxiliary WoE column
      woe <- as.vector(tapply(woe_source, f_bins, mean, na.rm = TRUE, default = 0))
    } else {
      # Calculate empirical WoE from counts
      total_pos <- max(sum(pos_counts), 1)
      total_neg <- max(sum(neg_counts), 1)

      # Simple protection against log(0)
      # (Ideally, bins should not be empty of one class, but we must handle it)
      p_pos <- ifelse(pos_counts == 0, 0.5, pos_counts) / total_pos
      p_neg <- ifelse(neg_counts == 0, 0.5, neg_counts) / total_neg

      woe <- log(p_pos / p_neg)
    }

    # Create id vector for data.frame case (sequential ordering)
    id <- seq_along(bins)

    # Pass aggregated vectors to builder
    gt <- .build_gains_table(bins, counts, pos_counts, neg_counts, woe, sort_by, id)
  } else {
    stop("Argument 'obj' must be an 'obwoe' object or a 'data.frame'.")
  }

  # ------------------------------------------------------------------------#
  # 3. Global Metric Calculation (Post-Construction)
  # ------------------------------------------------------------------------#

  # Note: KS and cum_pos_pct are already correctly calculated in the ordered table 'gt'

  ks <- max(gt$ks, na.rm = TRUE)
  total_iv <- sum(gt$iv, na.rm = TRUE)

  # Calculate Gini Coefficient (Area Between Lorenz Curve and Equality)
  # Gini = 2*AUC - 1
  cum_neg <- c(0, gt$cum_neg_pct)
  cum_pos <- c(0, gt$cum_pos_pct)

  # Trapezoidal rule for AUC
  auc <- sum(diff(cum_neg) * (head(cum_pos, -1) + tail(cum_pos, -1))) / 2
  gini <- abs(2 * auc - 1)

  # Identify bin with max KS
  ks_bin <- gt$bin[which.max(gt$ks)]

  result <- list(
    table = gt,
    metrics = list(
      ks = ks * 100, # Percentage
      gini = gini * 100, # Percentage
      auc = auc,
      total_iv = total_iv,
      ks_bin = ks_bin
    ),
    feature = feature,
    n_bins = nrow(gt),
    n_obs = sum(gt$count)
  )

  class(result) <- "obwoe_gains"
  return(result)
}


#' @keywords internal
.build_gains_table <- function(bins, counts, pos_counts, neg_counts, woe, sort_by, id) {
  # 1. Create Temporary Data Frame for Safe Sorting
  # ---------------------------------------------#
  df <- data.frame(
    id = id,
    bin = bins,
    count = counts,
    pos_count = pos_counts,
    neg_count = neg_counts,
    woe = woe,
    stringsAsFactors = FALSE # Avoid converting bins to factors prematurely
  )

  # Pre-calculate event rate for sorting purposes
  df$pos_rate <- ifelse(df$count > 0, df$pos_count / df$count, 0)

  # 2. APPLY SORTING (CRITICAL STEP)
  # --------------------------------#
  # Sorting must happen BEFORE calculating cumulative metrics.
  # Otherwise, KS and Lift curves will be mathematically incorrect.

  if (sort_by == "woe") {
    # Descending WoE (Typically: High Score/Good -> Low Score/Bad)
    # Note: Check your specific WoE sign convention.
    # If WoE = ln(Bad/Good), High WoE = High Risk.
    # If WoE = ln(Good/Bad), High WoE = Low Risk.
    # Standard gains tables usually go from High Risk to Low Risk.
    ord <- order(df$woe, decreasing = TRUE)
  } else if (sort_by == "event_rate") {
    # Descending Event Rate (High Risk -> Low Risk)
    ord <- order(df$pos_rate, decreasing = TRUE)
  } else if (sort_by == "id") {
    # Original Algorithm Order (Respecting internal logic)
    ord <- order(df$id)
  } else {
    # "bin": Alphabetical or Factor Order
    if (is.factor(bins)) {
      ord <- order(bins) # Respects levels
    } else {
      ord <- order(df$bin) # Alphabetical
    }
  }

  # Reorder the dataframe
  df <- df[ord, ]

  # 3. Calculate Global Totals (Invariant)
  # --------------------------------------#
  total_pos <- sum(df$pos_count)
  total_neg <- sum(df$neg_count)
  total_n <- sum(df$count)

  # Division by zero protection
  if (total_pos == 0) total_pos <- 1
  if (total_neg == 0) total_neg <- 1
  if (total_n == 0) total_n <- 1

  overall_rate <- total_pos / (total_pos + total_neg)

  # 4. Calculate Vectorized Metrics (Bin Level)
  # -------------------------------------------#
  df$count_pct <- df$count / total_n
  df$neg_rate <- df$neg_count / df$count

  # Distributions (Share of Total)
  df$pos_pct <- df$pos_count / total_pos # % of Total Events (Recall per bin)
  df$neg_pct <- df$neg_count / total_neg # % of Total Non-Events

  # Odds and Log-Odds
  df$odds <- ifelse(df$neg_count > 0, df$pos_count / df$neg_count, NA)
  df$log_odds <- ifelse(!is.na(df$odds) & df$odds > 0, log(df$odds), 0)

  # Information Value (IV)
  # Formula: (Dist_Event - Dist_NonEvent) * WoE
  term <- (df$pos_pct - df$neg_pct) * df$woe
  df$iv <- ifelse(is.finite(term), term, 0)

  # 5. Calculate Cumulative Metrics (Correctly Ordered)
  # ---------------------------------------------------#
  df$cum_pos_pct <- cumsum(df$pos_pct)
  df$cum_neg_pct <- cumsum(df$neg_pct)

  # KS Statistic: Max absolute difference between cumulative distributions
  df$ks <- abs(df$cum_pos_pct - df$cum_neg_pct)

  # Lift: Segment Event Rate / Overall Event Rate
  df$lift <- ifelse(overall_rate > 0, df$pos_rate / overall_rate, 0)

  # Capture Rate (Cumulative % of Events)
  df$capture_rate <- df$cum_pos_pct

  # Cleanup
  rownames(df) <- NULL
  return(df)
}


#' @method print obwoe_gains
#' @export
print.obwoe_gains <- function(x, digits = 4, ...) {
  cat("Gains Table:", x$feature, "\n")
  cat(strrep("=", 50), "\n\n")

  cat(sprintf("Observations: %d  |  Bins: %d\n", x$n_obs, x$n_bins))
  cat(sprintf("Total IV: %.4f\n\n", x$metrics$total_iv))

  cat("Performance Metrics:\n")
  cat(sprintf("  KS Statistic: %.2f%%\n", x$metrics$ks))
  cat(sprintf("  Gini Coefficient: %.2f%%\n", x$metrics$gini))
  cat(sprintf("  AUC: %.4f\n\n", x$metrics$auc))

  # Print table with key columns
  display_cols <- c("bin", "count", "pos_rate", "woe", "iv", "cum_pos_pct", "ks", "lift")
  tbl <- x$table[, display_cols]

  tbl$pos_rate <- sprintf("%.2f%%", tbl$pos_rate * 100)
  tbl$woe <- round(tbl$woe, digits)
  tbl$iv <- round(tbl$iv, digits)
  tbl$cum_pos_pct <- sprintf("%.1f%%", tbl$cum_pos_pct * 100)
  tbl$ks <- sprintf("%.1f%%", tbl$ks * 100)
  tbl$lift <- round(tbl$lift, 2)

  print(tbl, row.names = FALSE)

  invisible(x)
}


#' @title Plot Gains Table
#'
#' @description
#' Visualizes gains table metrics including cumulative capture curves,
#' KS plot, and lift chart.
#'
#' @param x An object of class \code{"obwoe_gains"}.
#' @param type Character string: \code{"cumulative"} (default), \code{"ks"},
#'   \code{"lift"}, or \code{"woe_iv"}.
#' @param ... Additional arguments passed to plotting functions.
#'
#' @return Invisibly returns \code{NULL}.
#'
#' @method plot obwoe_gains
#' @export
plot.obwoe_gains <- function(x, type = c("cumulative", "ks", "lift", "woe_iv"), ...) {
  type <- match.arg(type)
  gt <- x$table

  old_par <- par(mar = c(5, 4, 4, 4))
  on.exit(par(old_par))

  if (type == "cumulative") {
    # Cumulative capture curve
    n <- nrow(gt)
    cum_pct <- seq_len(n) / n

    plot(cum_pct * 100, gt$cum_pos_pct * 100,
      type = "b", pch = 19, col = "#F44336",
      xlab = "% Population (sorted by risk)",
      ylab = "% Events Captured",
      main = paste("Cumulative Gains:", x$feature),
      xlim = c(0, 100), ylim = c(0, 100), ...
    )

    # Add random line
    abline(0, 1, lty = 2, col = "gray50")

    # Add perfect model line
    event_pct <- sum(gt$pos_count) / sum(gt$count)
    segments(0, 0, event_pct * 100, 100, lty = 3, col = "gray30")
    segments(event_pct * 100, 100, 100, 100, lty = 3, col = "gray30")

    legend("bottomright",
      legend = c("Model", "Random", "Perfect"),
      lty = c(1, 2, 3),
      col = c("#F44336", "gray50", "gray30"),
      pch = c(19, NA, NA), bty = "n"
    )
  } else if (type == "ks") {
    # KS plot
    n <- nrow(gt)
    x_axis <- seq_len(n)

    plot(x_axis, gt$cum_pos_pct * 100,
      type = "b", pch = 19, col = "#F44336",
      xlab = "Bin (sorted by risk)",
      ylab = "Cumulative %",
      main = paste("KS Plot:", x$feature, "| KS =", round(x$metrics$ks, 1), "%"),
      ylim = c(0, 100), ...
    )

    lines(x_axis, gt$cum_neg_pct * 100, type = "b", pch = 17, col = "#2196F3")

    # Mark maximum KS
    ks_idx <- which.max(gt$ks)
    segments(ks_idx, gt$cum_neg_pct[ks_idx] * 100,
      ks_idx, gt$cum_pos_pct[ks_idx] * 100,
      col = "#4CAF50", lwd = 2
    )

    legend("bottomright",
      legend = c("Events (Bad)", "Non-Events (Good)", "Max KS"),
      lty = 1, pch = c(19, 17, NA),
      col = c("#F44336", "#2196F3", "#4CAF50"),
      lwd = c(1, 1, 2), bty = "n"
    )
  } else if (type == "lift") {
    # Lift chart
    barplot(gt$lift,
      names.arg = substr(gt$bin, 1, 10),
      col = ifelse(gt$lift > 1, "#4CAF50", "#F44336"),
      border = NA,
      las = 2,
      ylab = "Lift",
      main = paste("Lift Chart:", x$feature), ...
    )

    abline(h = 1, lty = 2, col = "gray50")
  } else if (type == "woe_iv") {
    # WoE and IV bars
    par(mar = c(6, 4, 4, 4))

    bp <- barplot(gt$woe,
      names.arg = substr(gt$bin, 1, 10),
      col = ifelse(gt$woe >= 0, "#4CAF50", "#F44336"),
      border = NA,
      las = 2,
      ylab = "Weight of Evidence",
      main = paste("WoE & IV:", x$feature), ...
    )

    abline(h = 0, lty = 1, col = "gray30")

    # Overlay IV as points
    par(new = TRUE)
    plot(bp, gt$iv,
      type = "b", pch = 17, col = "#FF9800",
      axes = FALSE, xlab = "", ylab = "",
      ylim = c(0, max(gt$iv, na.rm = TRUE) * 1.2)
    )

    axis(4, col = "#FF9800", col.axis = "#FF9800")
    mtext("IV Contribution", side = 4, line = 2.5, col = "#FF9800")
  }

  invisible(NULL)
}
