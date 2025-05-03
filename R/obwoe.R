#' @title
#' Optimal Binning and Weight of Evidence Calculation
#'
#' @description
#' This function implements a comprehensive suite of state-of-the-art algorithms
#' for optimal binning and Weight of Evidence (WoE) calculation for both numerical
#' and categorical variables. It maximizes predictive power while preserving
#' interpretability through monotonic constraints, information-theoretic optimization,
#' and statistical validation. Primarily designed for credit risk modeling,
#' classification problems, and predictive analytics applications.
#'
#' @details
#' ## Algorithm Classification
#'
#' ### Categorical Variables
#'
#' | Algorithm | Abbreviation | Theoretical Foundation | Key Features |
#' |-----------|--------------|------------------------|--------------|
#' | ChiMerge | CM | Statistical Tests | Uses chi-square tests to merge adjacent bins |
#' | Dynamic Programming with Local Constraints | DPLC | Mathematical Programming | Maximizes IV with global optimization |
#' | Fisher's Exact Test Binning | FETB | Statistical Tests | Uses Fisher's exact test for optimal merging |
#' | Greedy Merge Binning | GMB | Iterative Optimization | Iteratively merges bins to maximize IV |
#' | Information Value Binning | IVB | Information Theory | Dynamic programming for IV maximization |
#' | Joint Entropy-Driven Information | JEDI | Information Theory | Adaptive merging with entropy optimization |
#' | Monotonic Binning Algorithm | MBA | Information Theory | Combines WoE/IV with monotonicity constraints |
#' | Mixed Integer Linear Programming | MILP | Mathematical Programming | Mathematical optimization with constraints |
#' | Monotonic Optimal Binning | MOB | Iterative Optimization | Specialized for monotonicity preservation |
#' | Simulated Annealing Binning | SAB | Metaheuristic Optimization | Simulated annealing for global optimization |
#' | Similarity-Based Logistic Partitioning | SBLP | Distance-Based Methods | Similarity measures for optimal partitioning |
#' | Sliding Window Binning | SWB | Iterative Optimization | Sliding window approach with adaptive merging |
#' | User-Defined Technique | UDT | Hybrid Methods | Flexible hybrid approach for binning |
#' | JEDI Multinomial WoE | JEDI_MWOE | Information Theory | Extension of JEDI for multinomial response |
#'
#' ### Numerical Variables
#'
#' | Algorithm | Abbreviation | Theoretical Foundation | Key Features |
#' |-----------|--------------|------------------------|--------------|
#' | Branch and Bound | BB | Mathematical Programming | Efficient search in solution space |
#' | ChiMerge | CM | Statistical Tests | Chi-square-based merging strategy |
#' | Dynamic Programming with Local Constraints | DPLC | Mathematical Programming | Constrained optimization with DP |
#' | Equal-Width Binning | EWB | Simple Discretization | Equal-width intervals with adaptive refinement |
#' | Fisher's Exact Test Binning | FETB | Statistical Tests | Fisher's test for statistical significance |
#' | Joint Entropy-Driven Interval | JEDI | Information Theory | Entropy optimization with adaptive merging |
#' | K-means Binning | KMB | Clustering | K-means inspired clustering approach |
#' | Local Density Binning | LDB | Density Estimation | Adapts to local density structure |
#' | Local Polynomial Density Binning | LPDB | Density Estimation | Polynomial density estimation approach |
#' | Monotonic Binning with Linear Programming | MBLP | Mathematical Programming | Linear programming with monotonicity |
#' | Minimum Description Length Principle | MDLP | Information Theory | MDL criterion with monotonicity |
#' | Monotonic Optimal Binning | MOB | Iterative Optimization | Specialized monotonicity preservation |
#' | Monotonic Risk Binning with LR Pre-binning | MRBLP | Hybrid Methods | Likelihood ratio pre-binning approach |
#' | Optimal Supervised Learning Partitioning | OSLP | Supervised Learning | Specialized supervised approach |
#' | Unsupervised Binning with Standard Deviation | UBSD | Statistical Methods | Standard deviation-based approach |
#' | Unsupervised Decision Tree | UDT | Decision Trees | Decision tree inspired binning |
#' | Isotonic Regression | IR | Statistical Methods | Pool Adjacent Violators algorithm |
#' | Fast MDLP with Monotonicity | FAST_MDLPM | Information Theory | Optimized MDL implementation |
#' | JEDI Multinomial WoE | JEDI_MWOE | Information Theory | Multinomial extension of JEDI |
#' | Sketch-based Binning | SKETCH | Approximate Computing | KLL sketch for efficient quantile approximation |
#'
#' ## Mathematical Framework
#'
#' ### Weight of Evidence (WoE)
#' The Weight of Evidence measures the predictive power of a bin and is defined as:
#'
#' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
#'
#' Where \eqn{P(X_i|Y=1)} is the proportion of positive events in bin i relative to all positive events,
#' and \eqn{P(X_i|Y=0)} is the proportion of negative events in bin i relative to all negative events.
#'
#' With Bayesian smoothing applied (used in many implementations):
#'
#' \deqn{WoE_i = \ln\left(\frac{n_{1i} + \alpha\pi}{n_1 + m\alpha} \cdot \frac{n_0 + m\alpha}{n_{0i} + \alpha(1-\pi)}\right)}
#'
#' Where:
#' \itemize{
#'   \item \eqn{n_{1i}} is the count of positive cases in bin i
#'   \item \eqn{n_{0i}} is the count of negative cases in bin i
#'   \item \eqn{n_1} is the total count of positive cases
#'   \item \eqn{n_0} is the total count of negative cases
#'   \item \eqn{\pi} is the overall positive rate
#'   \item \eqn{\alpha} is the smoothing parameter (typically 0.5)
#'   \item \eqn{m} is the number of bins
#' }
#'
#' ### Information Value (IV)
#' The Information Value quantifies the predictive power of a variable:
#'
#' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
#'
#' The total Information Value is the sum across all bins:
#'
#' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
#'
#' IV can be interpreted as follows:
#' \itemize{
#'   \item IV < 0.02: Not predictive
#'   \item 0.02 <= IV < 0.1: Weak predictive power
#'   \item 0.1 <= IV < 0.3: Medium predictive power
#'   \item 0.3 <= IV < 0.5: Strong predictive power
#'   \item IV >= 0.5: Suspicious (possible overfitting)
#' }
#'
#' ### Monotonicity Constraint
#' Many algorithms enforce monotonicity of WoE values across bins, which means:
#'
#' \deqn{WoE_1 \leq WoE_2 \leq \ldots \leq WoE_n} (increasing)
#'
#' or
#'
#' \deqn{WoE_1 \geq WoE_2 \geq \ldots \geq WoE_n} (decreasing)
#'
#' ## Method Selection
#' When method = "auto", the function tests multiple algorithms and selects the one
#' that produces the highest total Information Value while respecting the specified constraints.
#' The selection process considers:
#' \itemize{
#'   \item Total Information Value (IV)
#'   \item Monotonicity of WoE values
#'   \item Number of bins created
#'   \item Bin frequency distribution
#'   \item Statistical stability
#' }
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable column (must be binary: 0/1).
#' @param features Vector of feature names to process. If NULL, all features except the target will be processed.
#' @param min_bins Minimum number of bins (default: 3).
#' @param max_bins Maximum number of bins (default: 4).
#' @param method The binning method to use. Can be "auto" or one of the methods listed in the details section tables. Default is 'jedi'.
#' @param positive Character string specifying which category should be considered as positive. Must be either "bad|1" or "good|1".
#' @param preprocess Logical. Whether to preprocess the data before binning (default: TRUE).
#' @param progress Logical. Whether to display a progress bar. Default is TRUE.
#' @param trace Logical. Whether to generate error logs when testing existing methods.
#' @param outputall Logical. If TRUE, returns only the optimal binning gains table. If FALSE, returns a list with data, gains table, and reports (default: TRUE).
#' @param control A list of additional control parameters:
#'   \itemize{
#'     \item cat_cutoff: Minimum frequency for a category (default: 0.05)
#'     \item bin_cutoff: Minimum frequency for a bin (default: 0.05)
#'     \item min_bads: Minimum proportion of bad cases in a bin (default: 0.05)
#'     \item pvalue_threshold: P-value threshold for statistical tests (default: 0.05)
#'     \item max_n_prebins: Maximum number of pre-bins (default: 20)
#'     \item monotonicity_direction: Direction of monotonicity for some algorithms ("increase" or "decrease")
#'     \item lambda: Regularization parameter for some algorithms (default: 0.1)
#'     \item min_bin_size: Minimum bin size as a proportion of total observations (default: 0.05)
#'     \item min_iv_gain: Minimum IV gain for bin splitting for some algorithms (default: 0.01)
#'     \item max_depth: Maximum depth for tree-based algorithms (default: 10)
#'     \item num_miss_value: Value to replace missing numeric values (default: -999.0)
#'     \item char_miss_value: Value to replace missing categorical values (default: "N/A")
#'     \item outlier_method: Method for outlier detection ("iqr", "zscore", or "grubbs")
#'     \item outlier_process: Whether to process outliers (default: FALSE)
#'     \item iqr_k: IQR multiplier for outlier detection (default: 1.5)
#'     \item zscore_threshold: Z-score threshold for outlier detection (default: 3)
#'     \item grubbs_alpha: Significance level for Grubbs' test (default: 0.05)
#'     \item n_threads: Number of threads for parallel processing (default: 1)
#'     \item is_monotonic: Whether to enforce monotonicity in binning (default: TRUE)
#'     \item population_size: Population size for genetic algorithm (default: 50)
#'     \item max_generations: Maximum number of generations for genetic algorithm (default: 100)
#'     \item mutation_rate: Mutation rate for genetic algorithm (default: 0.1)
#'     \item initial_temperature: Initial temperature for simulated annealing (default: 1)
#'     \item cooling_rate: Cooling rate for simulated annealing (default: 0.995)
#'     \item max_iterations: Maximum number of iterations for iterative algorithms (default: 1000)
#'     \item include_upper_bound: Include upper bound for numeric bins (default is TRUE)
#'     \item bin_separator: Bin separator for optimal bins categorical variables (default = "%;%")
#'     \item laplace_smoothing: Smoothing parameter for WoE calculation (default: 0.5)
#'     \item sketch_k: Parameter controlling the accuracy of sketch-based algorithms (default: 200)
#'     \item sketch_width: Width parameter for sketch-based algorithms (default: 2000)
#'     \item sketch_depth: Depth parameter for sketch-based algorithms (default: 5)
#'     \item polynomial_degree: Degree of polynomial for LPDB algorithm (default: 3)
#'     \item auto_monotonicity: Auto-detect monotonicity direction (default: TRUE)
#'     \item monotonic_trend: Monotonicity direction for DP algorithm (default: "auto")
#'     \item use_chi2_algorithm: Whether to use enhanced Chi2 algorithm (default: FALSE)
#'     \item chi_merge_threshold: Threshold for chi-merge algorithm (default: 0.05)
#'     \item force_monotonic_direction: Force direction in MBLP (0=auto, 1=increasing, -1=decreasing)
#'     \item monotonicity_direction: Monotonicity for UDT ("none", "increasing", "decreasing", "auto")
#'     \item divergence_method: Divergence measure for DMIV ("he", "kl", "tr", "klj", "sc", "js", "l1", "l2", "ln")
#'     \item bin_method: Method for WoE calculation in DMIV ("woe", "woe1")
#'     \item adaptive_cooling: Whether to use adaptive cooling in SAB (default: TRUE)
#'     \item enforce_monotonic: Whether to enforce monotonicity in various algorithms (default: TRUE)
#'   }
#'
#' @return Depending on the value of outputall:
#' If outputall = FALSE:
#'   A data.table containing the optimal binning gains table (woebin).
#' If outputall = TRUE:
#'   A list containing:
#'   \item{data}{The original dataset with added WoE columns}
#'   \item{woebin}{Information about the bins created, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item bin: Bin label or range
#'       \item count: Number of observations in the bin
#'       \item count_distr: Proportion of observations in the bin
#'       \item good: Number of good cases (target = 0) in the bin
#'       \item bad: Number of bad cases (target = 1) in the bin
#'       \item good_rate: Proportion of good cases in the bin
#'       \item bad_rate: Proportion of bad cases in the bin
#'       \item woe: Weight of Evidence for the bin
#'       \item iv: Information Value contribution of the bin
#'     }
#'   }
#'   \item{report_best_model}{Report on the best tested models, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item method: Best method selected for the feature
#'       \item iv_total: Total Information Value achieved
#'       \item n_bins: Number of bins created
#'       \item runtime: Execution time for binning the feature
#'     }
#'   }
#'   \item{report_preprocess}{Preprocessing report for each feature, including:
#'     \itemize{
#'       \item feature: Name of the feature
#'       \item type: Data type of the feature
#'       \item missing_count: Number of missing values
#'       \item outlier_count: Number of outliers detected
#'       \item unique_count: Number of unique values
#'       \item mean_before: Mean value before preprocessing
#'       \item mean_after: Mean value after preprocessing
#'       \item sd_before: Standard deviation before preprocessing
#'       \item sd_after: Standard deviation after preprocessing
#'     }
#'   }
#'
#' @references
#' \itemize{
#'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
#'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
#'   \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and Its Applications. SIAM.
#'   \item Zeng, G. (2013). Metric Divergence Measures and Information Value in Credit Scoring. Journal of Mathematics, 2013, Article ID 848271, 10 pages.
#'   \item Zeng, Y. (2014). Univariate feature selection and binner. arXiv preprint arXiv:1410.5420.
#'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone Optimal Binning Algorithm for Credit Risk Modeling. Working Paper.
#'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In AAAI'92.
#'   \item Liu, H. & Setiono, R. (1995). Chi2: Feature Selection and Discretization of Numeric Attributes. In TAI'95.
#'   \item Fayyad, U., & Irani, K. (1993). Multi-interval discretization of continuous-valued attributes for classification learning. Proceedings of the 13th International Joint Conference on Artificial Intelligence, 1022-1027.
#'   \item Barlow, R. E., & Brunk, H. D. (1972). The isotonic regression problem and its dual. Journal of the American Statistical Association, 67(337), 140-147.
#'   \item Fisher, R. A. (1922). On the interpretation of X^2 from contingency tables, and the calculation of P. Journal of the Royal Statistical Society, 85, 87-94.
#'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
#'   \item Bertsimas, D., & Tsitsiklis, J. N. (1997). Introduction to Linear Optimization. Athena Scientific.
#'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
#'   \item Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. Science, 220(4598), 671-680.
#'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
#' }
#'
#' @examples
#' \dontrun{
#' # Example 1: Using the German Credit Data
#' library(OptimalBinningWoE)
#' library(data.table)
#' library(scorecard)
#' data(germancredit, package = "scorecard")
#' dt <- as.data.table(germancredit)
#'
#' # Process all features with MBLP method
#' result <- obwoe(dt,
#'   target = "creditability", method = "mblp",
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View WoE binning information
#' print(result)
#'
#' # Process only numeric features with MBLP method and get detailed output
#' numeric_features <- names(dt)[sapply(dt, is.numeric)]
#' numeric_features <- setdiff(numeric_features, "creditability")
#'
#' result_detailed <- obwoe(dt,
#'   target = "creditability", features = numeric_features,
#'   method = "mblp", preprocess = TRUE, outputall = FALSE,
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View WoE-transformed data
#' head(result_detailed$data)
#'
#' # View preprocessing report
#' print(result_detailed$report_preprocess)
#'
#' # View best model report
#' print(result_detailed$report_best_model)
#'
#' # Process only categoric features with UDT method
#' categoric_features <- names(dt)[sapply(dt, function(i) !is.numeric(i))]
#' categoric_features <- setdiff(categoric_features, "creditability")
#' result_cat <- obwoe(dt,
#'   target = "creditability", features = categoric_features,
#'   method = "udt", preprocess = TRUE,
#'   min_bins = 3, max_bins = 4, positive = "bad|1"
#' )
#'
#' # View binning information for categorical features
#' print(result_cat)
#'
#' # Example 2: Automatic method selection
#' result_auto <- obwoe(dt,
#'   target = "creditability",
#'   method = "auto", # Tries multiple methods and selects the best
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#'
#' # View which methods were selected for each feature
#' print(result_auto$report_best_model)
#'
#' # Example 3: Using specialized algorithms
#' # For numerical features with complex distributions
#' result_lpdb <- obwoe(dt,
#'   target = "creditability",
#'   features = numeric_features[1:3],
#'   method = "lpdb", # Local Polynomial Density Binning
#'   min_bins = 3, max_bins = 5, positive = "bad|1",
#'   control = list(polynomial_degree = 3)
#' )
#'
#' # For categorical features with many levels
#' result_jedi <- obwoe(dt,
#'   target = "creditability",
#'   features = categoric_features[1:3],
#'   method = "jedi", # Joint Entropy-Driven Information
#'   min_bins = 3, max_bins = 5, positive = "bad|1"
#' )
#' }
#' @import data.table
#' @importFrom stats rnorm pchisq median sd quantile
#' @importFrom utils modifyList setTxtProgressBar txtProgressBar
#'
#' @export
obwoe <- function(dt, target, features = NULL, min_bins = 3, max_bins = 4, method = "jedi",
                  positive = "bad|1", preprocess = TRUE, progress = TRUE, trace = FALSE,
                  outputall = TRUE, control = list()) {
  # Step 1: Treatment of arguments and exceptions
  # Set default control parameters
  default_control <- list(
    cat_cutoff = 0.05, bin_cutoff = 0.05, min_bads = 0.05,
    pvalue_threshold = 0.05, max_n_prebins = 20,
    monotonicity_direction = "increase", lambda = 0.1,
    min_bin_size = 0.05, min_iv_gain = 0.01, max_depth = 10,
    num_miss_value = -999.0, char_miss_value = "N/A",
    outlier_method = "iqr", outlier_process = FALSE, iqr_k = 1.5,
    zscore_threshold = 3, grubbs_alpha = 0.05, n_threads = 1L,
    is_monotonic = TRUE, population_size = 50L, max_generations = 100L,
    mutation_rate = 0.1, initial_temperature = 1, cooling_rate = 0.995,
    max_iterations = 1000L,
    include_upper_bound = TRUE,
    bin_separator = "%;%"
  )
  # Update default control with user-provided control parameters
  control <- modifyList(default_control, control)

  # Validate input arguments and handle exceptions
  OBValidateInputs(dt, target, features, method, preprocess, min_bins, max_bins, control, positive)

  # Step 2: Check input data 'dt', and if not data.table, transform
  if (!data.table::is.data.table(dt)) {
    dt <- data.table::as.data.table(dt)
  } else {
    dt <- data.table::copy(dt)
  }

  # Step 3: Capture the features to be optimized in the model
  if (is.null(features)) {
    features <- setdiff(colnames(dt), c(target, "target"))
  }

  # Step 4: Map the target variable and preprocess data
  # Map the target variable based on 'positive' argument
  data <- OBMapTargetVariable(dt, target, positive)

  if (preprocess) {
    # Preprocess data (handling outliers, missing values, etc.)
    preprocessed_data <- OBPreprocessData(data, target, features, control, preprocess = "both")
    # Extract preprocessing reports
    preprocessed_report <- data.table::rbindlist(lapply(preprocessed_data, function(x) data.table::setDT(x$report)), idcol = "feature")
    # Prepare preprocessed data
    for (feat in features) {
      data[[feat]] <- preprocessed_data[[feat]]$preprocess$feature_preprocessed
    }
  } else {
    # Generate preprocessing reports without modifying data
    preprocessed_data <- OBPreprocessData(data, target, features, control, preprocess = "report")
    preprocessed_report <- data.table::rbindlist(lapply(preprocessed_data, function(x) data.table::setDT(x$report)), idcol = "feature")
  }

  # Step 5: Validate the target variable
  # Check that the target variable is binary [0,1]
  if (!all(data[[target]] %in% c(0, 1))) {
    stop("Target variable must be binary and coded as 0 and 1 after mapping.")
  }

  # Step 6: Further treatments of variable types and other exceptions
  # Initialize lists to keep track of features
  nonprocessed_features <- character()
  singleclass_target_features <- character()

  # Check for unsupported variable types and single-class target in categorical variables
  for (feat in features) {
    featdim <- OBCheckDistinctsLength(data[[feat]], data[[target]])

    if (featdim[1] <= 2) {
      data[[feat]] <- as.factor(data[[feat]])
    }

    if (!is.numeric(data[[feat]]) && !is.factor(data[[feat]]) && !is.character(data[[feat]])) {
      # Unsupported variable type
      nonprocessed_features <- c(nonprocessed_features, feat)
    } else if (is.factor(data[[feat]]) || is.character(data[[feat]])) {
      # For categorical features, check if target variable has only one class
      target_values <- data[[target]][!is.na(data[[feat]])]
      if (length(unique(target_values)) == 1) {
        singleclass_target_features <- c(singleclass_target_features, feat)
      }
    }
  }

  # Remove nonprocessed and singleclass target features from the features list
  features <- setdiff(features, c(nonprocessed_features, singleclass_target_features))

  if (length(features) == 0) {
    stop("No features to process after removing unsupported or single-class target features.")
  }

  # Step 7: Apply algorithms
  # If there are features to process, apply the optimal binning algorithms
  results <- OBSelectBestModel(data, target, features, method, min_bins, max_bins, control, progress, trace)

  # Step 8: Prepare outputs conditional on user choice
  # Prepare WoE binning gains table
  woebin <- data.table::rbindlist(lapply(results, function(x) data.table::setDT(x$woebin)), idcol = "feature")

  if (!outputall) {
    # If outputall is FALSE, return the woebin gains table
    return(woebin)
  } else {
    # Prepare data with WoE features
    data_woe <- data.table::data.table()[, (target) := dt[[target]]]
    # Add WoE features
    data_woe <- cbind(data_woe, do.call(cbind, lapply(results, function(x) data.table::setDT(x$woefeature))))
    # Prepare best model selection report
    report_best_model <- data.table::rbindlist(lapply(results, function(x) data.table::setDT(x$report)), idcol = "feature")
    # Return the results
    return(list(data = data_woe, woebin = woebin, report_best_model = report_best_model, report_preprocess = preprocessed_report))
  }
}

#' Select Optimal Features Based on Weight of Evidence
#'
#' @description
#' This function selects optimal features from the result of an Optimal Binning and
#' Weight of Evidence (WoE) analysis. It filters features based on their Information
#' Value (IV), allowing for fine-tuned feature selection for predictive modeling.
#'
#' @param obresult A list containing the result of the Optimal Binning and WoE analysis.
#'   Must include elements 'woedt' (a data.table with WoE transformed data) and
#'   'bestsreport' (a data.table with feature performance metrics).
#' @param target Character. The name of the target variable in the dataset.
#' @param iv_threshold Numeric. The minimum Information Value threshold for feature selection.
#'   Features with IV below this threshold will be excluded. Default is 0.02.
#' @param min_features Integer. The minimum number of features to select, regardless of
#'   their IV. If fewer features meet the IV threshold, this ensures a minimum
#'   set is still selected. Default is 5.
#' @param max_features Integer or NULL. The maximum number of features to select.
#'   If NULL (default), no maximum limit is applied.
#'
#' @return A list containing:
#'   \item{data}{A data.table with the selected WoE features and the target variable.}
#'   \item{selected_features}{A character vector of the selected WoE feature names.}
#'   \item{feature_iv}{A data.table with all features and their total IV.}
#'   \item{report}{A data.table summarizing the feature selection process.}
#'
#' @details
#' The function performs the following steps:
#' 1. Validates input parameters.
#' 2. Extracts and sorts features by their Information Value.
#' 3. Selects features based on the provided IV threshold.
#' 4. Adjusts the selection to meet minimum and maximum feature count requirements.
#' 5. Prepares a final dataset with selected WoE features and the target variable.
#' 6. Generates a summary report of the selection process.
#'
#' Mathematical Background:
#'
#' Weight of Evidence (WoE) and Information Value (IV) are key concepts in predictive modeling,
#' especially in credit scoring. They are derived from information theory and provide a way to
#' measure the predictive power of an independent variable in relation to the dependent variable.
#'
#' Let \eqn{Y} be a binary target variable and \eqn{X} be a predictor variable.
#'
#' For a given bin \eqn{i} of \eqn{X}:
#'
#' \deqn{P(X_i|Y=1) = \frac{\text{Number of events in bin i}}{\text{Total number of events}}}
#'
#' \deqn{P(X_i|Y=0) = \frac{\text{Number of non-events in bin i}}{\text{Total number of non-events}}}
#'
#' The Weight of Evidence for bin \eqn{i} is defined as:
#'
#' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
#'
#' The Information Value for the entire variable \eqn{X} is:
#'
#' \deqn{IV = \sum_{i} (P(X_i|Y=1) - P(X_i|Y=0)) \cdot WoE_i}
#'
#' Interpretation of Information Value:
#'
#' | IV Range  | Predictive Power |
#' |-----------|-------------------|
#' | < 0.02    | Useless           |
#' | 0.02-0.1  | Weak              |
#' | 0.1-0.3   | Medium            |
#' | 0.3-0.5   | Strong            |
#' | > 0.5     | Suspicious        |
#'
#' Note: An IV > 0.5 might indicate overfitting or data leakage and should be investigated.
#'
#' @examples
#' \dontrun{
#' # Assuming 'obwoe_result' is the output from an Optimal Binning and WoE analysis
#' result <- OBSelectOptimalFeatures(
#'   obresult = obwoe_result,
#'   target = "target_variable",
#'   iv_threshold = 0.05,
#'   min_features = 10,
#'   max_features = 30
#' )
#'
#' # Access the final dataset with selected WoE features
#' final_dataset <- result$data
#'
#' # View the selected WoE feature names
#' print(result$selected_features)
#'
#' # View the feature selection summary report
#' print(result$report)
#' }
#'
#' @importFrom data.table setDT setorder data.table
#' @export
OBSelectOptimalFeatures <- function(obresult, target, iv_threshold = 0.02, min_features = 5, max_features = NULL) {
  # Input validation
  if (!is.list(obresult) || !all(c("woedt", "woebins") %in% names(obresult))) {
    stop("'obresult' must be a list containing 'woedt' and 'woebins' elements")
  }
  if (!is.character(target) || length(target) != 1) {
    stop("'target' must be a single character string")
  }
  if (!is.numeric(iv_threshold) || iv_threshold < 0) {
    stop("'iv_threshold' must be a non-negative numeric value")
  }
  if (!is.numeric(min_features) || min_features < 1) {
    stop("'min_features' must be a positive integer")
  }
  if (!is.null(max_features) && (!is.numeric(max_features) || max_features < min_features)) {
    stop("'max_features' must be NULL or a numeric value greater than or equal to 'min_features'")
  }

  # Ensure woedt is a data.table
  obdt <- data.table::setDT(obresult$woedt)

  # Validate target variable
  if (!target %in% names(obdt)) {
    stop(sprintf("Target variable '%s' not found in the dataset", target))
  }

  # Calculate total IV for each feature from woebins
  feature_iv <- obresult$woebins[, .(total_iv = sum(iv)), by = .(feature)]
  data.table::setorder(feature_iv, -total_iv)

  # Select features based on IV threshold
  selected_features <- unique(feature_iv[total_iv >= iv_threshold, feature])

  # Adjust for min_features if necessary
  if (length(selected_features) < min_features) {
    selected_features <- feature_iv[1:min(min_features, nrow(feature_iv)), feature]
  }

  # Adjust for max_features if specified
  if (!is.null(max_features) && length(selected_features) > max_features) {
    selected_features <- selected_features[1:max_features]
  }

  # Add "_woe" suffix to selected features
  selected_woe_features <- paste0(selected_features, "_woe")

  # Validate selected WoE features exist in the dataset
  missing_features <- selected_woe_features[!selected_woe_features %in% names(obdt)]
  if (length(missing_features) > 0) {
    warning(sprintf(
      "The following WoE features were not found in the dataset: %s",
      paste(missing_features, collapse = ", ")
    ))
    selected_woe_features <- selected_woe_features[selected_woe_features %in% names(obdt)]
  }

  # Prepare the final dataset with selected WoE features and target
  final_dt <- obdt[, c(target, selected_woe_features), with = FALSE]

  # Prepare the report
  report <- data.table::data.table(
    total_features = sum(grepl("_woe$", names(obdt))),
    selected_features = length(selected_woe_features),
    iv_threshold = iv_threshold,
    min_iv = feature_iv[feature %in% selected_features, min(total_iv)],
    max_iv = feature_iv[feature %in% selected_features, max(total_iv)],
    mean_iv = feature_iv[feature %in% selected_features, mean(total_iv)]
  )

  # Return the results
  return(list(
    data = final_dt,
    selected_features = selected_woe_features,
    feature_iv = feature_iv,
    report = report
  ))
}


#' Preprocess Data for Optimal Binning
#'
#' @param dt A data.table containing the dataset.
#' @param target Target name
#' @param features Vector of feature names to process.
#' @param control A list of control parameters.
#' @param preprocess Preprocess feature. 'both' feature and report. Can also be 'both' or 'feature'
#'
#' @return A list of preprocessed data for each feature.
#'
#' @export
OBPreprocessData <- function(dt, target, features, control, preprocess = "both") {
  preprocessed_data <- list()
  for (feat in features) {
    preprocessed_data[[feat]] <- OBDataPreprocessor(
      target = dt[[target]],
      feature = dt[[feat]],
      num_miss_value = control$num_miss_value,
      char_miss_value = control$char_miss_value,
      outlier_method = control$outlier_method,
      outlier_process = control$outlier_process,
      preprocess = preprocess,
      iqr_k = control$iqr_k,
      zscore_threshold = control$zscore_threshold,
      grubbs_alpha = control$grubbs_alpha
    )
  }
  return(preprocessed_data)
}

#' Check if WoE values are monotonic
#'
#' @param woe_values Vector of WoE values
#'
#' @return Logical indicating if WoE values are monotonic
#'
#' @keywords internal
#' @export
OBWoEMonotonic <- function(woe_values) {
  diff_woe <- diff(woe_values)
  all(diff_woe >= 0) || all(diff_woe <= 0)
}

#' Map Target Variable
#'
#' @param dt Data table
#' @param target Target variable name
#' @param positive Positive class indicator
#'
#' @return Updated data table with mapped target variable
#'
#' @keywords internal
#' @export
OBMapTargetVariable <- function(dt, target, positive) {
  if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) > 2) {
      stop("Target variable must have exactly two categories")
    }
    positive_value <- strsplit(positive, "\\|")[[1]][1]
    target_col <- dt[[target]]
    dt[, (target) := ifelse(target_col == positive_value, 1, 0)]
  } else if (!all(dt[[target]] %in% c(0, 1))) {
    stop("Target variable must be binary (0 or 1) or a string with two categories")
  }
  return(dt)
}

#' Calculate Special WoE
#'
#' @param target Target values for special cases
#'
#' @return WoE value for special cases
#'
#' @keywords internal
#' @export
OBCalculateSpecialWoE <- function(target) {
  counts <- table(target)
  log(counts[2] / sum(counts) / (counts[1] / sum(counts)))
}

#' Create Special Bin
#'
#' @param dt_special Data for special cases
#' @param woebin Existing WoE bins
#' @param special_woe WoE value for special cases
#'
#' @return Special bin information
#'
#' @keywords internal
#' @export
OBCreateSpecialBin <- function(dt_special, woebin, special_woe) {
  data.table::data.table(
    bin = "Special",
    count = nrow(dt_special),
    count_neg = sum(dt_special$target == 0),
    count_pos = sum(dt_special$target == 1),
    woe = special_woe,
    iv = (sum(dt_special$target == 1) / (sum(woebin$count_pos) + sum(dt_special$target == 1)) -
      sum(dt_special$target == 0) / (sum(woebin$count_neg) + sum(dt_special$target == 0))) * special_woe
  )
}


#' Validate Inputs for Optimal Binning
#'
#' @param dt A data.table containing the dataset.
#' @param target The name of the target variable.
#' @param features Vector of feature names to process.
#' @param method The binning method to use.
#' @param preprocess Logical. Whether to preprocess the data before binning.
#' @param min_bins Minimum number of bins.
#' @param max_bins Maximum number of bins.
#' @param control A list of additional control parameters.
#' @param positive Character string specifying which category should be considered as positive.
#'
#' @return None. Throws an error if any input is invalid.
#'
#' @keywords internal
#' @export
OBValidateInputs <- function(dt, target, features, method, preprocess, min_bins, max_bins, control, positive) {
  # Check if dt is a data.table
  if (!data.table::is.data.table(dt)) {
    stop("The 'dt' argument must be a data.table.")
  }

  # Check if target exists in dt
  if (!target %in% names(dt)) {
    stop("The 'target' variable does not exist in the provided data.table.")
  }

  # Check if target is binary or has two categories
  if (is.numeric(dt[[target]])) {
    if (!all(dt[[target]] %in% c(0, 1))) {
      stop("The 'target' variable must be binary (0 or 1) when numeric.")
    }
  } else if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) != 2) {
      stop("The 'target' variable must have exactly two categories when categorical.")
    }
  } else {
    stop("The 'target' variable must be either numeric (0/1) or categorical (two categories).")
  }

  # Check features (if provided)
  if (!is.null(features)) {
    if (!all(features %in% names(dt))) {
      stop("One or more specified 'features' do not exist in the provided data.table.")
    }
  }

  # Define all possible methods
  # all_methods <- c(
  #   "auto", "cm", "dplc", "gmb", "ldb", "mba", "mblp", "milp", "mob", "obnp", "swb", "udt",
  #   "bb", "bs", "dpb", "eb", "eblc", "efb", "ewb", "ir", "jnbo", "kmb", "mdlp", "mrblp", "plaob", "qb", "sbb", "ubsd"
  # )

  all_methods_char <- unique(c("auto", names(OBGetAlgoName()$char)))
  all_methods_num <- unique(c("auto", names(OBGetAlgoName()$num)))

  all_methods <- sort(unique(c(all_methods_char, all_methods_num)))

  # Check binning method
  if (!all(method %in% all_methods)) {
    stop(paste("Invalid binning method. Choose one of the following:", paste(all_methods, collapse = ", ")))
  }

  # Check preprocess
  if (!is.logical(preprocess)) {
    stop("'preprocess' must be a logical value (TRUE or FALSE).")
  }

  # Check min_bins and max_bins
  # if (!is.numeric(min_bins) || min_bins < 2) {
  #   stop("min_bins must be an integer greater than or equal to 2.")
  # }
  # if (!is.numeric(max_bins) | max_bins < min_bins) {
  #   stop("max_bins must be an integer greater than or equal to min_bins.")
  # }

  # Check control
  if (!is.list(control)) {
    stop("'control' must be a list.")
  }

  # Check specific control parameters
  if (!is.numeric(control$cat_cutoff) || control$cat_cutoff <= 0 || control$cat_cutoff >= 1) {
    stop("control$cat_cutoff must be a number between 0 and 1.")
  }
  if (!is.numeric(control$bin_cutoff) || control$bin_cutoff <= 0 || control$bin_cutoff >= 1) {
    stop("control$bin_cutoff must be a number between 0 and 1.")
  }

  # Check positive argument
  if (!is.character(positive) || !grepl("^(bad|good)\\|1$", positive)) {
    stop("'positive' must be either 'bad|1' or 'good|1'")
  }

  # If all checks pass, the function will return silently
}


#' Select Optimal Binning Algorithm
#'
#' @description
#' This function selects the appropriate binning algorithm based on the method and variable type.
#'
#' @param feature The name of the feature to bin.
#' @param method The binning method to use.
#' @param dt A data.table containing the dataset.
#' @param min_bin Minimum number of bins.
#' @param max_bin Maximum number of bins.
#' @param control A list of additional control parameters.
#'
#' @return A list containing the selected algorithm, its parameters, and the method name.
#'
#' @keywords internal
#' @export
OBSelectAlgorithm <- function(feature, method, dt, min_bin, max_bin, control) {
  # Determine if the feature is categorical or numeric
  is_categorical <- is.factor(dt[[feature]]) || is.character(dt[[feature]])

  # Get available algorithms using OBGetAlgoName()
  available_algorithms <- OBGetAlgoName()

  # Select the appropriate algorithm based on the method and variable type
  data_type <- if (is_categorical) "char" else "num"

  # Find the algorithm that matches the method
  selected_algorithm <- NULL
  for (algo in names(available_algorithms[[data_type]])) {
    selected_algorithm[[algo]] <- available_algorithms[[data_type]][[algo]]
  }

  # Check if a valid algorithm was found
  if (is.null(selected_algorithm)) {
    stop(paste("The", method, "method is not applicable for", if (data_type == "char") "categorical" else "numeric", "variables."))
  }

  # Define default parameters for all algorithms
  default_params <- list(
    min_bins = min_bin,
    max_bins = max_bin,
    bin_cutoff = control$bin_cutoff,
    max_n_prebins = control$max_n_prebins
  )

  # Define specific parameters for certain algorithms
  specific_params <- list(
    optimal_binning_categorical_gab = list(
      population_size = control$population_size,
      num_generations = control$num_generations,
      mutation_rate = control$mutation_rate,
      crossover_rate = control$crossover_rate,
      time_limit_seconds = control$time_limit_seconds
    ),
    optimal_binning_categorical_oslp = list(monotonic = control$monotonic),
    optimal_binning_categorical_sab = list(
      initial_temperature = control$initial_temperature,
      cooling_rate = control$cooling_rate,
      max_iterations = control$max_iterations
    ),
    optimal_binning_numerical_bb = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_cart = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_dpb = list(n_threads = control$n_threads),
    optimal_binning_numerical_eb = list(n_threads = control$n_threads),
    optimal_binning_numerical_mba = list(n_threads = control$n_threads),
    optimal_binning_numerical_milp = list(n_threads = control$n_threads),
    optimal_binning_numerical_mrblp = list(n_threads = control$n_threads)
  )

  # Merge default parameters with specific parameters for the selected algorithm
  algorithm_params <- c(
    default_params,
    specific_params[[selected_algorithm]] %||% list()
  )

  # Merge with user-provided control parameters for the specific algorithm
  if (!is.null(control[[selected_algorithm]])) {
    algorithm_params <- utils::modifyList(algorithm_params, control[[selected_algorithm]])
  }

  # Return the selected algorithm and its parameters
  list(
    algorithm = selected_algorithm,
    params = algorithm_params,
    method = method
  )
}


#' Get Available Optimal Binning Algorithms
#'
#' @description
#' This function retrieves all available optimal binning algorithms from the OBWoE package,
#' separating them into categorical and numerical types.
#'
#' @return A list containing two elements:
#'   \item{char}{A named list of categorical binning algorithms}
#'   \item{num}{A named list of numerical binning algorithms}
#'
#' @details
#' The function searches for all exported functions in the OBWoE package that start with
#' "optimal_binning_categorical_" or "optimal_binning_numerical_". It then creates two separate lists
#' for categorical and numerical algorithms, using the last part of the function name (after the last
#' underscore) as the list item name.
#'
#' @examples
#' \dontrun{
#' algorithms <- OBGetAlgoName()
#' print(algorithms$char) # List of categorical algorithms
#' print(algorithms$num) # List of numerical algorithms
#' }
#'
#' @export
OBGetAlgoName <- function() {
  # Get all exported functions from OBWoE package
  pk <- getNamespaceExports("OBWoE")

  # Helper function to extract the last part of the function name
  get_last_part <- function(x) {
    sapply(strsplit(as.character(x), "_"), function(parts) parts[length(parts)])
  }

  catmethods <- c(
    "dmiv", "cm", "sketch", "swb", "fetb", "milp", "jedi",
    "ivb", "mba", "sblp", "dp", "sab", "udt", "mob"
  )

  # "gmb" (bugs)

  nummethods <- c(
    "cm", "bb", "ir", "kmb", "dp", "ubsd", "lpdb", "ewb",
    "dmiv", "ldb", "sketch", "jedi", "fetb", "udt", "mrblp",
    "mblp", "oslp", "mob", "mdlp"
  )

  # Filter and process categorical algorithms
  obj <- intersect(
    pk[grepl("optimal_binning_categorical_", pk)],
    paste0("optimal_binning_categorical_", catmethods)
  )

  categorical <- lapply(obj, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last_part(f))
  })
  names(categorical) <- get_last_part(obj)

  # Filter and process numerical algorithms
  obj <- intersect(
    pk[grepl("optimal_binning_numerical_", pk)],
    paste0("optimal_binning_numerical_", nummethods)
  )

  numerical <- lapply(obj, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last_part(f))
  })
  names(numerical) <- get_last_part(obj)

  # Return a list with both types of algorithms
  return(list(char = categorical, num = numerical))
}


#' Select the Best Model for Optimal Binning
#'
#' This function selects the best model for optimal binning across multiple features
#' using various binning algorithms for both numerical and categorical variables.
#'
#' @param dt A data.table containing the target variable and features to be binned.
#' @param target The name of the target variable in the data.table.
#' @param features A character vector of feature names to be binned.
#' @param method A method for use. If not available, test all.
#' @param min_bins The minimum number of bins to use in the binning process.
#' @param max_bins The maximum number of bins to use in the binning process.
#' @param control A list of control parameters for the binning algorithms (not used directly in this function).
#' @param progress Logical; if TRUE, display a progress bar during processing (default is TRUE).
#' @param trace Logical; if TRUE, provide more detailed output for debugging (default is FALSE).
#'
#' @return A list containing the results for each feature:
#'   \item{woebin}{The Weight of Evidence (WoE) binning result for the best model.}
#'   \item{woefeature}{The WoE-transformed feature for the best model.}
#'   \item{bestmethod}{The name of the algorithm that produced the best model.}
#'   \item{report}{A data.table summarizing the performance of all tried models.}
#'
#' @details
#' The function iterates through each feature, applying various binning algorithms
#' suitable for either numerical or categorical data. It then selects the best model
#' based on monotonicity, number of zero-count bins, total number of bins, and
#' Information Value (IV).
#'
#' For features with 2 or fewer distinct values, the function forces them to be
#' treated as factors and applies categorical binning methods.
#'
#' If a binning algorithm fails, the function attempts to relax the binning parameters
#' and try again. If it still fails, that method is skipped for that feature.
#'
#' @importFrom data.table data.table rbindlist setorder
#' @importFrom utils modifyList
#' @importFrom progress progress_bar
#'
#' @examples
#' \dontrun{
#' library(data.table)
#' dt <- data.table(
#'   target = sample(0:1, 1000, replace = TRUE),
#'   feature1 = rnorm(1000),
#'   feature2 = sample(letters[1:5], 1000, replace = TRUE)
#' )
#' results <- OBSelectBestModel(
#'   dt = dt,
#'   target = "target",
#'   features = c("feature1", "feature2"),
#'   min_bins = 3,
#'   max_bins = 10
#' )
#' }
#'
#' @export
OBSelectBestModel <- function(dt, target, features, method = NULL, min_bins, max_bins, control, progress = TRUE, trace = FALSE) {
  results <- list()

  algoes <- OBGetAlgoName()

  numerical_methods <- lapply(algoes$num, function(a) {
    a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins))
    return(a)
  })

  categorical_methods <- lapply(algoes$char, function(a) {
    a$params <- utils::modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins, bin_separator = control$bin_separator))
    return(a)
  })

  if (progress) {
    pb <- progress::progress_bar$new(
      format = "Processing :what [:bar] :percent | ETA: :eta | Elapsed: :elapsed",
      total = length(features) * 3,
      clear = FALSE,
      width = 80
    )
  }

  for (feat in features) {
    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", paste0("Feature: ", feat))))
    }

    dt_feature <- data.table::data.table(target = dt[[target]], feature = dt[[feat]])

    is_string <- is.character(dt_feature$feature) || is.factor(dt_feature$feature)
    methods_to_try <- if (is_string) categorical_methods else numerical_methods

    if (!is.null(method) && length(intersect("auto", as.character(method))) == 0) {
      if (any(method %in% names(methods_to_try))) {
        methods_to_try <- methods_to_try[method]
      } else {
        warning(sprintf(
          "The provided method '%s' is not available for this variable type (%s). Using automatic method selection.",
          paste0(method, collapse = ";"),
          if (is_string) "categorical" else "numeric"
        ))
      }
    }

    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", "Trying methods")))
    }

    OO <- lapply(methods_to_try, function(m) {
      tryCatch(
        {
          binning_result <- suppressWarnings(suppressMessages(
            do.call(
              m$algorithm,
              c(list(target = dt_feature$target, feature = dt_feature$feature), m$params)
            )
          ))

          if (!is.null(binning_result)) {
            binning_result$algorithm <- m$algorithm
            binning_result$method <- m$method
            return(binning_result)
          } else {
            return(NULL)
          }
        },
        error = function(e) {
          if (trace) {
            message(sprintf("Error in method %s for feature %s: %s", m$algorithm, feat, e$message))
          }
          return(NULL)
        }
      )
    })

    # Remove NULL results
    OO <- Filter(Negate(is.null), OO)
    mm <- data.table::rbindlist(
      lapply(names(OO), function(m) {
        tryCatch(
          {
            data.table::data.table(
              model_method = m,
              model_algorithm = OO[[m]]$algorithm,
              total_iv = sum(OO[[m]]$iv, na.rm = TRUE),
              total_bins = length(OO[[m]]$bin),
              total_zero_pos = sum(OO[[m]]$count_pos == 0, na.rm = TRUE),
              total_zero_neg = sum(OO[[m]]$count_neg == 0, na.rm = TRUE),
              is_monotonic = as.numeric(OBWoEMonotonic(OO[[m]]$woe))
            )
          },
          error = function(e) {
            data.table::data.table(
              model_method = m,
              model_algorithm = NA_character_,
              total_iv = NA_real_,
              total_bins = NA_integer_,
              total_zero_pos = NA_integer_,
              total_zero_neg = NA_integer_,
              is_monotonic = NA_real_
            )
          }
        )
      }),
      fill = TRUE,
      use.names = TRUE
    )

    fn_rank <- function(m) {
      m$rk0 <- rank(-m$is_monotonic)
      m$rk1 <- rank(-m$total_iv)
      m$rk2 <- rank(m$total_zero_pos)
      m$rk3 <- rank(m$total_zero_neg)
      m$id <- apply(m[, c("rk0", "rk0", "rk1", "rk2", "rk3")], 1, mean)
      m <- m[order(m$id)]
      m$rk0 <- m$rk1 <- m$rk2 <- m$rk3 <- NULL
      return(m)
    }

    mm <- fn_rank(mm)
    best_model <- OO[[unique(head(mm, 1)$model_method)]]

    woefeature <- if (!is_string) {
      OBApplyWoENum(best_model, dt_feature$feature, include_upper_bound = control$include_upper_bound)
    } else {
      OBApplyWoECat(best_model, dt_feature$feature, bin_separator = control$bin_separator)
    }

    woebin <- OBGainsTableFeature(woefeature, dt_feature$target)

    bestmethod <- best_model$algorithm

    report <- mm

    if (progress) {
      pb$tick(tokens = list(what = sprintf("%-5s", "Finalizing")))
    }

    results[[feat]] <- list(
      woebin = data.table::setDT(woebin),
      woefeature = data.table::setDT(woefeature),
      report = data.table::setDT(report),
      bestmethod = bestmethod
    )
  }

  return(results)
}
