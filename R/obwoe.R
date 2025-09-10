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
#'
#' ## **Categorical Variable Algorithms**
#'
#' | Algorithm                                  | Abbreviation | Theoretical Foundation        | Key Features                                                              |
#' | :----------------------------------------- | :----------- | :---------------------------- | :------------------------------------------------------------------------ |
#' | ChiMerge                                   | CM           | Statistical Tests             | Uses chi-square tests to merge.                                 |
#' | Dynamic Programming with Local Constraints | DPLC         | Mathematical Programming      | Maximizes IV with global constraints.                         |
#' | Fisher's Exact Test Binning                | FETB         | Statistical Tests             | Uses Fisher's exact test for statistical merging.               |
#' | Greedy Merge Binning                       | GMB          | Iterative Optimization        | Iteratively merges bins to optimize.                            |
#' | Information Value Binning                  | IVB          | Information Theory            | Dynamic programming for IV optimization.                      |
#' | Joint Entropy-Driven Information           | JEDI         | Information Theory            | Adaptive merging with entropy.                                  |
#' | Monotonic Binning Algorithm                | MBA          | Information Theory            | Combines WoE/IV with monotonicity constraints.                  |
#' | Mixed Integer Linear Programming           | MILP         | Mathematical Programming      | Mathematical optimization for binning.                          |
#' | Monotonic Optimal Binning                  | MOB          | Iterative Optimization        | Specialized for monotonicity.                                   |
#' | Simulated Annealing Binning                | SAB          | Metaheuristic Optimization    | Simulated annealing for global optimization.                    |
#' | Similarity-Based Logistic Partitioning     | SBLP         | Distance-Based Methods        | Similarity measures for optimal binning.                      |
#' | Sliding Window Binning                     | SWB          | Iterative Optimization        | Sliding window approach for binning.                            |
#' | User-Defined Technique                     | UDT          | Hybrid Methods                | Flexible hybrid approach.                                       |
#' | JEDI Multinomial WoE                       | JEDI_MWOE    | Information Theory            | Extension of JEDI for multinomial targets.                      |
#'
#' ## **Numerical Variable Algorithms**
#'
#' | Algorithm                                                  | Abbreviation | Theoretical Foundation        | Key Features                                                              |
#' | :--------------------------------------------------------- | :----------- | :---------------------------- | :------------------------------------------------------------------------ |
#' | Branch and Bound                                           | BB           | Mathematical Programming      | Efficient search in solution space.                       |
#' | ChiMerge                                                   | CM           | Statistical Methods           | Chi-square-based merging.                                       |
#' | Dynamic Programming with Local Constraints                 | DPLC         | Mathematical Programming      | Constrained optimization.                                       |
#' | Equal-Width Binning                                        | EWB          | Simple Discretization         | Equal-width intervals for binning.                              |
#' | Fisher's Exact Test Binning                                | FETB         | Statistical Tests             | Fisher's test for statistical merging.                          |
#' | Joint Entropy-Driven Interval                              | JEDI         | Information Theory            | Entropy optimization with merging.                              |
#' | K-means Binning                                            | KMB          | Clustering                    | K-means inspired clustering.                                    |
#' | Local Density Binning                                      | LDB          | Density Estimation            | Adapts to local density structure.                              |
#' | Local Polynomial Density Binning                           | LPDB         | Density Estimation            | Polynomial density estimation.                                  |
#' | Monotonic Binning with Linear Programming                  | MBLP         | Mathematical Programming      | Linear programming with monotonicity.                           |
#' | Minimum Description Length Principle                       | MDLP         | Information Theory            | MDL criterion with monotonicity.                                |
#' | Monotonic Optimal Binning                                  | MOB          | Iterative Optimization        | Specialized monotonicity.                                       |
#' | Monotonic Risk Binning with LR Pre-binning                 | MRBLP        | Hybrid Methods                | Likelihood ratio pre-binning.                                   |
#' | Optimal Supervised Learning Partitioning                   | OSLP         | Supervised Learning           | Specialized supervised approach.                                |
#' | Unsupervised Binning with Standard Deviation               | UBSD         | Statistical Methods           | Standard deviation-based binning.                               |
#' | Unsupervised Decision Tree                                 | UDT          | Decision Trees                | Decision tree inspired binning.                                 |
#' | Isotonic Regression                                        | IR           | Statistical Methods           | Pool Adjacent Violators algorithm (PAVA).                       |
#' | Fast MDLP with Monotonicity                                | FAST_MDLPM   | Information Theory            | Optimized MDL implementation.                                   |
#' | JEDI Multinomial WoE                                       | JEDI_MWOE    | Information Theory            | Multinomial extension of JEDI.                                  |
#' | Sketch-based Binning                                       | SKETCH       | Approximate Computing         | KLL sketch for efficient computation.                           |
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
#'   target = "creditability", method = "jedi",
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
#'   method = c("jedi", "cm"), preprocess = TRUE, outputall = FALSE,
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
#'   method = "jedi", # Local Polynomial Density Binning
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
obwoe <- function(dt, target, features = NULL, min_bins = 3L, max_bins = 4L, method = "jedi",
                  positive = "bad|1", preprocess = TRUE, progress = TRUE, trace = FALSE,
                  outputall = TRUE, control = list()) {
  # 1. Default control parameters
  default_control <- list(
    cat_cutoff = 0.05,
    bin_cutoff = 0.05,
    min_bads = 0.05,
    pvalue_threshold = 0.05,
    max_n_prebins = 20,
    monotonicity_direction = "increase",
    lambda = 0.1,
    min_bin_size = 0.05,
    min_iv_gain = 0.01,
    max_depth = 10,
    num_miss_value = -999.0,
    char_miss_value = "N/A",
    outlier_method = "iqr",
    outlier_process = FALSE,
    iqr_k = 1.5,
    zscore_threshold = 3,
    grubbs_alpha = 0.05,
    n_threads = 1L,
    is_monotonic = TRUE,
    population_size = 50L,
    max_generations = 100L,
    mutation_rate = 0.1,
    initial_temperature = 1,
    cooling_rate = 0.995,
    max_iterations = 1000L,
    include_upper_bound = TRUE,
    bin_separator = "%;%",
    laplace_smoothing = 0.5,
    sketch_k = 200,
    sketch_width = 2000,
    sketch_depth = 5,
    polynomial_degree = 3,
    auto_monotonicity = TRUE,
    monotonic_trend = "auto",
    use_chi2_algorithm = FALSE,
    chi_merge_threshold = 0.05,
    force_monotonic_direction = 0L
  )
  control <- modifyList(default_control, control)

  # 2. Validate inputs
  OBValidateInputs(dt, target, features, method, preprocess, min_bins, max_bins, control, positive)

  # 3. Ensure a data.frame copy
  if (!is.data.frame(dt)) {
    dt <- as.data.frame(dt, stringsAsFactors = FALSE)
  }

  # 4. Feature selection
  if (is.null(features)) {
    features <- setdiff(names(dt), c(target, "target"))
  }

  # 5. Map target variable
  data <- OBMapTargetVariable(dt, target, positive)

  # 6. Preprocess data
  prep_mode <- if (preprocess) "both" else "report"
  pre_list <- OBPreprocessData(data, target, features, control, preprocess = prep_mode)
  # Preprocess report
  report_preprocess <- do.call(rbind, lapply(names(pre_list), function(feat) {
    cbind(
      feature = feat,
      as.data.frame(pre_list[[feat]]$report, stringsAsFactors = FALSE)
    )
  }))
  rownames(report_preprocess) <- NULL
  # Overwrite features if full preprocess
  if (preprocess) {
    for (feat in features) {
      data[[feat]] <- pre_list[[feat]]$preprocess$feature_preprocessed
    }
  }

  # 7. Confirm binary target
  if (!all(data[[target]] %in% c(0, 1))) {
    stop("Target variable must be binary and coded as 0 and 1 after mapping.")
  }

  # 8. Detect unsupported or single-class features
  skip_feats <- c()
  single_class <- c()
  for (feat in features) {
    dims <- OBCheckDistinctsLength(data[[feat]], data[[target]])
    if (dims[1] <= 2) {
      data[[feat]] <- as.factor(data[[feat]])
    }
    if (!is.numeric(data[[feat]]) && !is.factor(data[[feat]]) && !is.character(data[[feat]])) {
      skip_feats <- c(skip_feats, feat)
    } else if (is.factor(data[[feat]]) || is.character(data[[feat]])) {
      vals <- data[[target]][!is.na(data[[feat]])]
      if (length(unique(vals)) == 1) single_class <- c(single_class, feat)
    }
  }
  features <- setdiff(features, c(skip_feats, single_class))
  if (length(features) == 0) {
    stop("No features to process after removing unsupported or single-class features.")
  }

  # 9. Run optimal binning
  results <- OBSelectBestModel(data, target, features, method, min_bins, max_bins, control, progress, trace)

  # 10. Assemble woe bin table
  woebin <- do.call(rbind, lapply(names(results), function(feat) {
    cbind(
      feature = feat,
      as.data.frame(results[[feat]]$woebin, stringsAsFactors = FALSE)
    )
  }))
  rownames(woebin) <- NULL

  if (!outputall) {
    return(woebin)
  }

  # 11. Build WoE-transformed data.frame
  data_woe <- data.frame(data[[target]], stringsAsFactors = FALSE)
  colnames(data_woe) <- target
  woe_feats <- lapply(results, function(x) as.data.frame(x$woefeature, stringsAsFactors = FALSE))
  data_woe <- cbind(data_woe, do.call(cbind, woe_feats))

  # 12. Best model report
  report_best_model <- do.call(rbind, lapply(names(results), function(feat) {
    cbind(
      feature = feat,
      as.data.frame(results[[feat]]$report, stringsAsFactors = FALSE)
    )
  }))
  rownames(report_best_model) <- NULL

  # 13. Return full output
  list(
    data = data_woe,
    woebin = woebin,
    report_best_model = report_best_model,
    report_preprocess = report_preprocess
  )
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
OBSelectOptimalFeatures <- function(obresult, target, iv_threshold = 0.02, min_features = 5L, max_features = NULL) {
  # Input checks
  if (!is.list(obresult) || !all(c("woedt", "woebins") %in% names(obresult))) {
    stop("'obresult' must be a list containing 'woedt' and 'woebins'")
  }
  # tail
  if (!is.character(target) || length(target) != 1) {
    stop("'target' must be a single character string")
  }
  if (!is.numeric(iv_threshold) || iv_threshold < 0) {
    stop("'iv_threshold' must be non-negative numeric")
  }
  if (!is.numeric(min_features) || min_features < 1) {
    stop("'min_features' must be a positive integer")
  }
  if (!is.null(max_features) && (!is.numeric(max_features) || max_features < min_features)) {
    stop("'max_features' must be NULL or >= min_features")
  }

  # Ensure data.frame
  obdt <- obresult$woedt
  if (!is.data.frame(obdt)) obdt <- as.data.frame(obdt, stringsAsFactors = FALSE)

  if (!target %in% names(obdt)) {
    stop(sprintf("Target '%s' not found in the data", target))
  }

  # Compute total IV per feature
  iv_df <- stats::aggregate(iv ~ feature, data = obresult$woebins, sum)
  names(iv_df) <- c("feature", "total_iv")
  iv_df <- iv_df[order(-iv_df$total_iv), ]

  # Select by threshold
  sel_feats <- iv_df$feature[iv_df$total_iv >= iv_threshold]
  if (length(sel_feats) < min_features) {
    sel_feats <- iv_df$feature[seq_len(min(min_features, nrow(iv_df)))]
  }
  if (!is.null(max_features) && length(sel_feats) > max_features) {
    sel_feats <- sel_feats[seq_len(max_features)]
  }

  woe_cols <- paste0(sel_feats, "_woe")
  missing <- setdiff(woe_cols, names(obdt))
  if (length(missing) > 0) {
    warning(sprintf("Missing WoE columns: %s", paste(missing, collapse = ", ")))
    woe_cols <- intersect(woe_cols, names(obdt))
  }

  final_data <- obdt[, c(target, woe_cols), drop = FALSE]

  # Summary report
  report <- data.frame(
    total_features    = sum(grepl("_woe$", names(obdt))),
    selected_features = length(woe_cols),
    iv_threshold      = iv_threshold,
    min_iv            = if (length(sel_feats) > 0) min(iv_df$total_iv[iv_df$feature %in% sel_feats]) else NA,
    max_iv            = if (length(sel_feats) > 0) max(iv_df$total_iv[iv_df$feature %in% sel_feats]) else NA,
    mean_iv           = if (length(sel_feats) > 0) mean(iv_df$total_iv[iv_df$feature %in% sel_feats]) else NA,
    stringsAsFactors  = FALSE
  )

  list(
    data              = final_data,
    selected_features = woe_cols,
    feature_iv        = iv_df,
    report            = report
  )
}


#' Preprocess Data for Optimal Binning
#'
#' @param dt A data.frame containing the dataset.
#' @param target String name of the target column.
#' @param features Character vector of feature names.
#' @param control List of control parameters.
#' @param preprocess One of "both" or "report".
#' @return Named list of preprocessing results for each feature.
#' @export
OBPreprocessData <- function(dt, target, features, control, preprocess = c("both", "report")) {
  preprocess <- match.arg(preprocess)
  out <- vector("list", length(features))
  names(out) <- features
  for (feat in features) {
    out[[feat]] <- OBDataPreprocessor(
      target            = dt[[target]],
      feature           = dt[[feat]],
      num_miss_value    = control$num_miss_value,
      char_miss_value   = control$char_miss_value,
      outlier_method    = control$outlier_method,
      outlier_process   = control$outlier_process,
      preprocess        = preprocess,
      iqr_k             = control$iqr_k,
      zscore_threshold  = control$zscore_threshold,
      grubbs_alpha      = control$grubbs_alpha
    )
  }
  out
}


#' Check if WoE values are monotonic
#'
#' @param woe_values Numeric vector of WoE.
#' @return TRUE if entirely nondecreasing or nonincreasing.
#' @export
OBWoEMonotonic <- function(woe_values) {
  d <- diff(woe_values)
  all(d >= 0) || all(d <= 0)
}


#' Map Target Variable to 0/1
#'
#' @param dt A data.frame.
#' @param target Name of the target column.
#' @param positive Which level is positive, "bad|1" or "good|1".
#' @return data.frame with recoded target.
#' @export
OBMapTargetVariable <- function(dt, target, positive) {
  if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) != 2) {
      stop("Target variable must have exactly two categories")
    }
    pos_val <- strsplit(positive, "\\|", perl = TRUE)[[1]][1]
    dt[[target]] <- ifelse(dt[[target]] == pos_val, 1, 0)
  } else if (!all(dt[[target]] %in% c(0, 1))) {
    stop("Target variable must be binary (0/1) or two-level factor/character")
  }
  dt
}


#' Calculate Special WoE for Edge Cases
#'
#' @param target Numeric or factor vector of two values.
#' @return Numeric WoE.
#' @export
OBCalculateSpecialWoE <- function(target) {
  tbl <- table(target)
  if (length(tbl) != 2) stop("Target must have exactly two values")
  p1 <- tbl[2] / sum(tbl)
  p0 <- tbl[1] / sum(tbl)
  log(p1 / p0)
}


#' Create Special Bin Entry
#'
#' @param dt_special data.frame of special-case rows.
#' @param woebin data.frame of existing bins (must have count_pos, count_neg).
#' @param special_woe Numeric WoE value.
#' @return Single-row data.frame with special bin info.
#' @export
OBCreateSpecialBin <- function(dt_special, woebin, special_woe) {
  total_pos <- sum(woebin$count_pos, na.rm = TRUE)
  total_neg <- sum(woebin$count_neg, na.rm = TRUE)
  new_pos <- sum(dt_special$target == 1, na.rm = TRUE)
  new_neg <- sum(dt_special$target == 0, na.rm = TRUE)
  data.frame(
    bin = "Special",
    count = nrow(dt_special),
    count_neg = new_neg,
    count_pos = new_pos,
    woe = special_woe,
    iv = ((new_pos / (total_pos + new_pos) -
      new_neg / (total_neg + new_neg)) * special_woe),
    stringsAsFactors = FALSE
  )
}


#' Validate Inputs for Optimal Binning
#'
#' @param dt A data.frame.
#' @param target Target column name.
#' @param features Character vector or NULL.
#' @param method String or vector of methods.
#' @param preprocess Logical.
#' @param min_bins Integer.
#' @param max_bins Integer.
#' @param control List of control parameters.
#' @param positive "bad|1" or "good|1".
#' @return NULL; stops on error.
#' @export
OBValidateInputs <- function(dt, target, features, method, preprocess, min_bins, max_bins, control, positive) {
  if (!is.data.frame(dt)) stop("The 'dt' argument must be a data.frame.")
  if (!target %in% names(dt)) stop("The 'target' variable does not exist.")
  # Check target
  if (is.numeric(dt[[target]])) {
    if (!all(dt[[target]] %in% c(0, 1))) {
      stop("Numeric target must be binary (0/1).")
    }
  } else if (is.character(dt[[target]]) || is.factor(dt[[target]])) {
    if (length(unique(dt[[target]])) != 2) {
      stop("Categorical target must have exactly two levels.")
    }
  } else {
    stop("Target must be numeric (0/1) or a two-level factor/character.")
  }
  # Features exist
  if (!is.null(features) && !all(features %in% names(dt))) {
    stop("Some 'features' do not exist in the data.frame.")
  }
  # Methods valid
  all_methods <- sort(unique(c("auto", names(OBGetAlgoName()$char), names(OBGetAlgoName()$num))))
  if (!all(method %in% all_methods)) {
    stop(sprintf("Invalid method. Choose from: %s", paste(all_methods, collapse = ", ")))
  }
  if (!is.logical(preprocess)) stop("'preprocess' must be TRUE or FALSE.")
  if (!is.numeric(min_bins) || min_bins < 2) stop("'min_bins' must be >= 2.")
  if (!is.numeric(max_bins) || max_bins < min_bins) stop("'max_bins' must be >= min_bins.")
  if (!is.list(control)) stop("'control' must be a list.")
  if (!is.numeric(control$cat_cutoff) || control$cat_cutoff <= 0 || control$cat_cutoff >= 1) {
    stop("control$cat_cutoff must be between 0 and 1.")
  }
  if (!grepl("^(bad|good)\\|1$", positive)) {
    stop("'positive' must be 'bad|1' or 'good|1'.")
  }
}


#' Select Optimal Binning Algorithm (Unused in main flow)
#'
#' @param feature Name of the feature.
#' @param method Desired method.
#' @param dt data.frame.
#' @param min_bin Integer.
#' @param max_bin Integer.
#' @param control List.
#' @return List(algorithm, params, method).
#' @export
OBSelectAlgorithm <- function(feature, method, dt, min_bin, max_bin, control) {
  is_cat <- is.factor(dt[[feature]]) || is.character(dt[[feature]])
  algos <- OBGetAlgoName()
  dtype <- if (is_cat) "char" else "num"
  sel <- algos[[dtype]][[method]]
  if (is.null(sel)) {
    stop(sprintf("Method '%s' not applicable for %s variables.", method, if (is_cat) "categorical" else "numeric"))
  }
  default_params <- list(
    min_bins = min_bin, max_bins = max_bin,
    bin_cutoff = control$bin_cutoff,
    max_n_prebins = control$max_n_prebins
  )
  specific <- list(
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
      cooling_rate        = control$cooling_rate,
      max_iterations      = control$max_iterations
    ),
    optimal_binning_numerical_bb = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_cart = list(is_monotonic = control$is_monotonic),
    optimal_binning_numerical_dpb = list(n_threads = control$n_threads),
    optimal_binning_numerical_eb = list(n_threads = control$n_threads),
    optimal_binning_numerical_mba = list(n_threads = control$n_threads),
    optimal_binning_numerical_milp = list(n_threads = control$n_threads),
    optimal_binning_numerical_mrblp = list(n_threads = control$n_threads)
  )
  spec <- specific[[sel$algorithm]]
  if (is.null(spec)) spec <- list()
  params <- modifyList(default_params, spec)
  if (!is.null(control[[sel$algorithm]])) {
    params <- modifyList(params, control[[sel$algorithm]])
  }
  list(algorithm = sel$algorithm, params = params, method = sel$method)
}


#' Get Available Optimal Binning Algorithms
#'
#' @return List with $char (categorical) and $num (numerical) named lists.
#' @export
OBGetAlgoName <- function() {
  pk <- getNamespaceExports("OptimalBinningWoE")
  get_last <- function(x) utils::tail(strsplit(x, "_")[[1]], 1)
  catm <- c(
    "dmiv", "cm", "sketch", "swb", "fetb", "milp", "jedi",
    "ivb", "mba", "sblp", "dp", "sab", "udt", "mob"
  )
  numm <- c(
    "cm", "bb", "ir", "kmb", "dp", "ubsd", "lpdb", "ewb",
    "dmiv", "ldb", "sketch", "jedi", "fetb", "udt", "mrblp",
    "mblp", "oslp", "mob", "mdlp"
  )

  cat_objs <- intersect(
    pk[grepl("^optimal_binning_categorical_", pk)],
    paste0("optimal_binning_categorical_", catm)
  )
  char <- stats::setNames(lapply(cat_objs, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last(f))
  }), sapply(cat_objs, get_last))

  num_objs <- intersect(
    pk[grepl("^optimal_binning_numerical_", pk)],
    paste0("optimal_binning_numerical_", numm)
  )
  num <- stats::setNames(lapply(num_objs, function(f) {
    o <- formals(f)
    o <- o[setdiff(names(o), c("target", "feature"))]
    list(algorithm = f, params = o, method = get_last(f))
  }), sapply(num_objs, get_last))

  list(char = char, num = num)
}


#' Select the Best Model for Optimal Binning
#'
#' @param dt data.frame with target + feature columns.
#' @param target Name of the target column.
#' @param features Character vector of feature names.
#' @param method String or vector; methods to try or "auto".
#' @param min_bins Integer.
#' @param max_bins Integer.
#' @param control List of control parameters.
#' @param progress Logical; show progress bar.
#' @param trace Logical; print method errors.
#' @return Named list of results per feature: each has woebin, woefeature, report, bestmethod.
#' @export
OBSelectBestModel <- function(dt, target, features, method = NULL,
                              min_bins, max_bins, control, progress = TRUE, trace = FALSE) {
  results <- list()
  algos <- OBGetAlgoName()

  num_methods <- lapply(algos$num, function(a) {
    a$params <- modifyList(a$params, list(min_bins = min_bins, max_bins = max_bins))
    a
  })
  char_methods <- lapply(algos$char, function(a) {
    a$params <- modifyList(
      a$params,
      list(
        min_bins = min_bins,
        max_bins = max_bins,
        bin_separator = control$bin_separator
      )
    )
    a
  })

  if (progress) {
    pb <- progress::progress_bar$new(
      format = "Processing :what [:bar] :percent | ETA: :eta | Elapsed: :elapsed",
      total  = length(features) * 3,
      clear  = FALSE,
      width  = 80
    )
  }

  for (feat in features) {
    if (progress) pb$tick(tokens = list(what = sprintf("%-5s", paste0("Feature: ", feat))))
    df_feat <- data.frame(
      target = dt[[target]],
      feature = dt[[feat]],
      stringsAsFactors = FALSE
    )
    is_str <- is.character(df_feat$feature) || is.factor(df_feat$feature)
    methods_to_try <- if (is_str) char_methods else num_methods

    # filter by provided methods
    if (!is.null(method) && !"auto" %in% method) {
      valid <- intersect(method, names(methods_to_try))
      if (length(valid) > 0) {
        methods_to_try <- methods_to_try[valid]
      } else {
        warning(sprintf(
          "Methods '%s' not available for %s; using auto.",
          paste(method, collapse = ";"),
          if (is_str) "categorical" else "numeric"
        ))
      }
    }
    if (progress) pb$tick(tokens = list(what = sprintf("%-5s", "Trying methods")))

    # apply each algorithm
    OO <- lapply(methods_to_try, function(m) {
      tryCatch(
        {
          res <- suppressWarnings(suppressMessages(
            do.call(
              m$algorithm,
              c(
                list(
                  target = df_feat$target,
                  feature = df_feat$feature
                ),
                m$params
              )
            )
          ))
          if (!is.null(res)) {
            res$algorithm <- m$algorithm
            res$method <- m$method
            res
          } else {
            NULL
          }
        },
        error = function(e) {
          if (trace) message(sprintf("Error in %s for %s: %s", m$algorithm, feat, e$message))
          NULL
        }
      )
    })
    OO <- Filter(Negate(is.null), OO)

    # summarize each result
    mm_list <- mapply(function(name, res) {
      tryCatch(
        {
          data.frame(
            model_method = name,
            model_algorithm = res$algorithm,
            total_iv = sum(res$iv, na.rm = TRUE),
            total_bins = length(res$bin),
            total_zero_pos = sum(res$count_pos == 0, na.rm = TRUE),
            total_zero_neg = sum(res$count_neg == 0, na.rm = TRUE),
            is_monotonic = as.numeric(OBWoEMonotonic(res$woe)),
            stringsAsFactors = FALSE
          )
        },
        error = function(e) {
          data.frame(
            model_method = name,
            model_algorithm = NA_character_,
            total_iv = NA_real_,
            total_bins = NA_integer_,
            total_zero_pos = NA_integer_,
            total_zero_neg = NA_integer_,
            is_monotonic = NA_real_,
            stringsAsFactors = FALSE
          )
        }
      )
    }, name = names(OO), res = OO, SIMPLIFY = FALSE)
    mm <- do.call(rbind, mm_list)
    rownames(mm) <- NULL

    # ranking function (preserves original weighting bug: rk0 twice)
    fn_rank <- function(df) {
      df$rk0 <- rank(-df$is_monotonic)
      df$rk1 <- rank(-df$total_iv)
      df$rk2 <- rank(df$total_zero_pos)
      df$rk3 <- rank(df$total_zero_neg)
      df$id <- (df$rk0 + df$rk0 + df$rk1 + df$rk2 + df$rk3) / 5
      df <- df[order(df$id), ]
      df[c("rk0", "rk1", "rk2", "rk3")] <- NULL
      df
    }
    mm <- fn_rank(mm)

    best_name <- mm$model_method[1]
    best_res <- OO[[best_name]]

    # apply WoE transformation
    if (!is_str) {
      w_feat <- OBApplyWoENum(best_res, df_feat$feature, include_upper_bound = control$include_upper_bound)
    } else {
      w_feat <- OBApplyWoECat(best_res, df_feat$feature, bin_separator = control$bin_separator)
    }
    w_feat <- as.data.frame(w_feat, stringsAsFactors = FALSE)

    w_bin <- as.data.frame(OBGainsTableFeature(w_feat, df_feat$target), stringsAsFactors = FALSE)

    if (progress) pb$tick(tokens = list(what = sprintf("%-5s", "Finalizing")))

    results[[feat]] <- list(
      woebin = w_bin,
      woefeature = w_feat,
      report = mm,
      bestmethod = best_res$algorithm
    )
  }

  results
}
