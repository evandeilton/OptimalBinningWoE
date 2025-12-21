# ============================================================================ #
# step_obwoe: Optimal Binning and WoE Transformation for tidymodels
# ============================================================================ #

#' @title Valid Binning Algorithms
#'
#' @description
#' Internal function returning the vector of all valid algorithm identifiers
#' supported by the OptimalBinningWoE package. Used for validation and
#' parameter definition.
#'
#' @return A character vector of valid algorithm names including "auto".
#'
#' @export
.valid_algorithms <- function() {
  c(
    "auto",
    # Universal (numerical and categorical)
    "jedi", "jedi_mwoe", "cm", "dp", "dmiv", "fetb", "mob", "sketch", "udt",
    # Categorical only
    "gmb", "ivb", "mba", "milp", "sab", "sblp", "swb",
    # Numerical only
    "bb", "ewb", "fast_mdlp", "ir", "kmb", "ldb", "lpdb", "mblp",
    "mdlp", "mrblp", "oslp", "ubsd"
  )
}


#' @title Numerical-Only Algorithms
#'
#' @description
#' Internal function returning algorithm identifiers that support only
#' numerical features.
#'
#' @return A character vector of numerical-only algorithm names.
#'
#' @export
.numerical_only_algorithms <- function() {
  c(
    "bb", "ewb", "fast_mdlp", "ir", "kmb", "ldb", "lpdb", "mblp",
    "mdlp", "mrblp", "oslp", "ubsd"
  )
}


#' @title Categorical-Only Algorithms
#'
#' @description
#' Internal function returning algorithm identifiers that support only
#' categorical features.
#'
#' @return A character vector of categorical-only algorithm names.
#'
#' @export
.categorical_only_algorithms <- function() {
  c("gmb", "ivb", "mba", "milp", "sab", "sblp", "swb")
}


#' @title Universal Algorithms
#'
#' @description
#' Internal function returning algorithm identifiers that support both
#' numerical and categorical features.
#'
#' @return A character vector of universal algorithm names.
#'
#' @export
.universal_algorithms <- function() {
  c("jedi", "jedi_mwoe", "cm", "dp", "dmiv", "fetb", "mob", "sketch", "udt")
}


#' @title Optimal Binning and WoE Transformation Step
#'
#' @description
#' \code{step_obwoe()} creates a \emph{specification} of a recipe step that
#' discretizes predictor variables using one of 28 state-of-the-art optimal
#' binning algorithms and transforms them into Weight of Evidence (WoE) values.
#' This step fully integrates the \strong{OptimalBinningWoE} package into the
#' \code{tidymodels} framework, supporting supervised discretization for both
#' binary and multinomial classification targets with extensive hyperparameter
#' tuning capabilities.
#'
#' @inheritParams recipes::step_center
#' @param ... One or more selector functions to choose variables for this step.
#'   See \code{\link[recipes]{selections}} for available selectors. Common
#'   choices include \code{all_predictors()}, \code{all_numeric_predictors()},
#'   or \code{all_nominal_predictors()}. Ensure the selected variables are
#'   compatible with the chosen \code{algorithm} (e.g., do not apply
#'   \code{"mdlp"} to categorical data).
#' @param role For variables created by this step, what role should they have?
#'
#'   Default is \code{"predictor"}.
#' @param trained A logical indicating whether the step has been trained
#'   (fitted). This should not be set manually.
#' @param outcome A character string specifying the name of the binary or
#'   multinomial response variable. This argument is \strong{required} as all
#'   binning algorithms are supervised. The outcome must exist in the training
#'   data provided to \code{prep()}. The outcome should be encoded as a factor
#'   (standard for tidymodels classification) or as integers 0/1 for binary,
#'   0/1/2/... for multinomial.
#' @param algorithm Character string specifying the binning algorithm to use.
#'   Use \code{"auto"} (default) for automatic selection based on target type:
#'   \code{"jedi"} for binary targets, \code{"jedi_mwoe"} for multinomial.
#'
#'   Available algorithms are organized by supported feature types:
#'
#'   \strong{Universal (numerical and categorical):}
#'   \code{"auto"}, \code{"jedi"}, \code{"jedi_mwoe"}, \code{"cm"}, \code{"dp"},
#'   \code{"dmiv"}, \code{"fetb"}, \code{"mob"}, \code{"sketch"}, \code{"udt"}
#'
#'   \strong{Numerical only:}
#'   \code{"bb"}, \code{"ewb"}, \code{"fast_mdlp"}, \code{"ir"}, \code{"kmb"},
#'   \code{"ldb"}, \code{"lpdb"}, \code{"mblp"}, \code{"mdlp"}, \code{"mrblp"},
#'   \code{"oslp"}, \code{"ubsd"}
#'
#'   \strong{Categorical only:}
#'   \code{"gmb"}, \code{"ivb"}, \code{"mba"}, \code{"milp"}, \code{"sab"},
#'   \code{"sblp"}, \code{"swb"}
#'
#'   This parameter is tunable with \code{tune()}.
#' @param min_bins Integer specifying the minimum number of bins to create.
#'   Must be at least 2. Default is 2. This parameter is tunable with
#'   \code{tune()}.
#' @param max_bins Integer specifying the maximum number of bins to create.
#'   Must be greater than or equal to \code{min_bins}. Default is 10. This
#'   parameter is tunable with \code{tune()}.
#' @param bin_cutoff Numeric value between 0 and 1 (exclusive) specifying the
#'   minimum proportion of total observations that each bin must contain. Bins
#'   with fewer observations are merged with adjacent bins. This serves as a
#'   regularization mechanism to prevent overfitting and ensure statistical
#'   stability of WoE estimates. Default is 0.05 (5\%). This parameter is
#'   tunable with \code{tune()}.
#' @param output Character string specifying the transformation output format:
#'   \describe{
#'     \item{\code{"woe"}}{Replaces the original variable with WoE values
#'       (default). This is the standard choice for logistic regression
#'       scorecards.
#'     }
#'     \item{\code{"bin"}}{Replaces the original variable with bin labels
#'       (character). Useful for tree-based models or exploratory analysis.
#'     }
#'     \item{\code{"both"}}{Keeps the original column and adds two new columns
#'       with suffixes \code{_woe} and \code{_bin}. Useful for model comparison
#'       or audit trails.
#'     }
#'   }
#' @param suffix_woe Character string suffix appended to create WoE column names
#'   when \code{output = "both"}. Default is \code{"_woe"}.
#' @param suffix_bin Character string suffix appended to create bin column names
#'   when \code{output = "both"}. Default is \code{"_bin"}.
#' @param na_woe Numeric value to assign to observations that cannot be mapped
#'   to a bin during \code{bake()}. This includes missing values (\code{NA}) and
#'   unseen categories not present in the training data. Default is 0, which
#'   represents neutral evidence (neither good nor bad).
#' @param control A named list of additional control parameters passed to
#'   \code{\link{control.obwoe}}. These provide fine-grained control over
#'   algorithm behavior such as convergence thresholds and maximum pre-bins.
#'   Parameters specified directly in \code{step_obwoe()} (e.g.,
#'   \code{bin_cutoff}) take precedence over values in this list.
#' @param binning_results Internal storage for fitted binning models after
#'   \code{prep()}. Do not set manually.
#' @param skip Logical. Should this step be skipped when \code{bake()} is
#'   called on new data? Default is \code{FALSE}. Setting to \code{TRUE} is
#'   rarely needed for WoE transformations but may be useful in specialized
#'   workflows.
#' @param id A unique character string to identify this step. If not provided,
#'   a random identifier is generated.
#'
#' @return An updated \code{recipe} object with the new step appended.
#'
#' @details
#' \subsection{Weight of Evidence Transformation}{
#'
#' Weight of Evidence (WoE) is a supervised encoding technique that transforms

#' categorical and continuous variables into a scale that measures the
#' predictive strength of each value or bin relative to the target variable.
#' For a bin \eqn{i}, the WoE is defined as:
#'
#' \deqn{WoE_i = \ln\left(\frac{\text{Distribution of Events}_i}{\text{Distribution of Non-Events}_i}\right)}
#'
#' Positive WoE values indicate the bin has a higher proportion of events
#' (e.g., defaults) than the overall population, while negative values indicate
#' lower risk.
#' }
#'
#' \subsection{Algorithm Selection Strategy}{
#'
#' The \code{algorithm} parameter provides access to 28 binning algorithms:
#'
#' \itemize{
#'   \item Use \code{algorithm = "auto"} (default) for automatic selection:
#'     \code{"jedi"} for binary targets, \code{"jedi_mwoe"} for multinomial.
#'   \item Use \code{algorithm = "mob"} (Monotonic Optimal Binning) when
#'     monotonic WoE trends are required for regulatory compliance (Basel/IFRS 9).
#'   \item Use \code{algorithm = "mdlp"} for entropy-based discretization of
#'     numerical variables (requires \code{all_numeric_predictors()}).
#'   \item Use \code{algorithm = "dp"} (Dynamic Programming) for exact optimal
#'     solutions when computational cost is acceptable.
#' }
#'
#' If an incompatible algorithm is applied to a variable (e.g., \code{"mdlp"}
#' on a factor), the step will issue a warning during \code{prep()} and skip
#' that variable, leaving it untransformed.
#' }
#'
#' \subsection{Handling New Data}{
#'
#' During \code{bake()}, observations are mapped to bins learned during
#' \code{prep()}:
#'
#' \itemize{
#'   \item \strong{Numerical variables}: Values are assigned to bins based on
#'     the learned cutpoints using interval notation.
#'   \item \strong{Categorical variables}: Categories are matched to their
#'     corresponding bins. Categories not seen during training receive the
#'     \code{na_woe} value.
#'   \item \strong{Missing values}: Always receive the \code{na_woe} value.
#' }
#' }
#'
#' \subsection{Tuning with tune}{
#'
#' This step is fully compatible with the \code{tune} package. The following
#' parameters support \code{tune()}:
#'
#' \itemize{
#'   \item \code{algorithm}: See \code{\link{obwoe_algorithm}}.
#'   \item \code{min_bins}: See \code{\link{obwoe_min_bins}}.
#'   \item \code{max_bins}: See \code{\link{obwoe_max_bins}}.
#'   \item \code{bin_cutoff}: See \code{\link{obwoe_bin_cutoff}}.
#' }
#' }
#'
#' \subsection{Case Weights}{
#'
#' This step does not currently support case weights. All observations are
#' treated with equal weight during binning optimization.
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
#' Navas-Palencia, G. (2020). Optimal Binning: Mathematical Programming
#' Formulation and Solution Approach. \emph{Expert Systems with Applications},
#' 158, 113508. \doi{10.1016/j.eswa.2020.113508}
#'
#' @seealso
#' \code{\link{obwoe}} for the underlying binning engine,
#' \code{\link{control.obwoe}} for advanced control parameters,
#' \code{\link{obwoe_algorithm}}, \code{\link{obwoe_min_bins}},
#' \code{\link{obwoe_max_bins}}, \code{\link{obwoe_bin_cutoff}} for tuning
#' parameter definitions,
#' \code{\link[recipes]{recipe}}, \code{\link[recipes]{prep}},
#' \code{\link[recipes]{bake}} for the tidymodels recipe framework.
#'
#' @examples
#' \donttest{
#' library(recipes)
#'
#' # Simulated credit data
#' set.seed(123)
#' credit_data <- data.frame(
#'   age = rnorm(500, 45, 12),
#'   income = exp(rnorm(500, 10, 0.6)),
#'   employment = sample(c("Employed", "Self-Employed", "Unemployed"),
#'     500,
#'     replace = TRUE, prob = c(0.7, 0.2, 0.1)
#'   ),
#'   education = factor(c("HighSchool", "Bachelor", "Master", "PhD")[
#'     sample(1:4, 500, replace = TRUE, prob = c(0.3, 0.4, 0.2, 0.1))
#'   ]),
#'   default = factor(rbinom(500, 1, 0.15),
#'     levels = c(0, 1),
#'     labels = c("No", "Yes")
#'   )
#' )
#'
#' # Example 1: Basic usage with automatic algorithm selection
#' rec_basic <- recipe(default ~ ., data = credit_data) %>%
#'   step_obwoe(all_predictors(), outcome = "default")
#'
#' rec_prepped <- prep(rec_basic)
#' baked_data <- bake(rec_prepped, new_data = NULL)
#' head(baked_data)
#'
#' # View binning details
#' tidy(rec_prepped, number = 1)
#'
#' # Example 2: Numerical-only algorithm on numeric predictors
#' rec_mdlp <- recipe(default ~ age + income, data = credit_data) %>%
#'   step_obwoe(all_numeric_predictors(),
#'     outcome = "default",
#'     algorithm = "mdlp",
#'     min_bins = 3,
#'     max_bins = 6
#'   )
#'
#' # Example 3: Output both bins and WoE
#' rec_both <- recipe(default ~ age, data = credit_data) %>%
#'   step_obwoe(age,
#'     outcome = "default",
#'     output = "both"
#'   )
#'
#' baked_both <- bake(prep(rec_both), new_data = NULL)
#' names(baked_both)
#' # Contains: default, age, age_woe, age_bin
#'
#' # Example 4: Custom control parameters
#' rec_custom <- recipe(default ~ ., data = credit_data) %>%
#'   step_obwoe(all_predictors(),
#'     outcome = "default",
#'     algorithm = "mob",
#'     bin_cutoff = 0.03,
#'     control = list(
#'       max_n_prebins = 30,
#'       convergence_threshold = 1e-8
#'     )
#'   )
#'
#' # Example 5: Tuning specification (for use with tune package)
#' # rec_tune <- recipe(default ~ ., data = credit_data) %>%
#' #   step_obwoe(all_predictors(),
#' #              outcome = "default",
#' #              algorithm = tune(),
#' #              min_bins = tune(),
#' #              max_bins = tune())
#' }
#'
#' @export
#' @importFrom recipes add_step step rand_id recipes_eval_select
#' @importFrom recipes check_new_data sel2char bake prep tidy tunable required_pkgs
#' @importFrom rlang enquos abort warn is_quosure
#' @importFrom tibble tibble as_tibble
step_obwoe <- function(recipe,
                       ...,
                       role = "predictor",
                       trained = FALSE,
                       outcome = NULL,
                       algorithm = "auto",
                       min_bins = 2L,
                       max_bins = 10L,
                       bin_cutoff = 0.05,
                       output = c("woe", "bin", "both"),
                       suffix_woe = "_woe",
                       suffix_bin = "_bin",
                       na_woe = 0,
                       control = list(),
                       binning_results = NULL,
                       skip = FALSE,
                       id = recipes::rand_id("obwoe")) {
  output <- match.arg(output)


  # Validate outcome (required for supervised binning)
  if (is.null(outcome)) {
    rlang::abort(
      c(
        "The `outcome` argument is required for supervised binning.",
        i = "Specify the name of the binary/multinomial target variable."
      )
    )
  }

  if (!is.character(outcome) || length(outcome) != 1L) {
    rlang::abort("`outcome` must be a single character string.")
  }

  # Validate algorithm (allow tune() placeholder)
  if (!rlang::is_quosure(algorithm) && !inherits(algorithm, "call")) {
    if (!is.character(algorithm) || length(algorithm) != 1L) {
      rlang::abort("`algorithm` must be a single character string or tune().")
    }

    valid_algos <- .valid_algorithms()
    if (!algorithm %in% valid_algos) {
      rlang::abort(
        c(
          sprintf("`algorithm = '%s'` is not recognized.", algorithm),
          i = sprintf(
            "Valid options: %s",
            paste(valid_algos, collapse = ", ")
          )
        )
      )
    }
  }

  # Validate min_bins (allow tune() placeholder)
  if (!rlang::is_quosure(min_bins) && !inherits(min_bins, "call")) {
    if (!is.numeric(min_bins) || length(min_bins) != 1L) {
      rlang::abort("`min_bins` must be a single integer or tune().")
    }
    min_bins <- as.integer(min_bins)
    if (min_bins < 2L) {
      rlang::abort("`min_bins` must be at least 2.")
    }
  }

  # Validate max_bins (allow tune() placeholder)
  if (!rlang::is_quosure(max_bins) && !inherits(max_bins, "call")) {
    if (!is.numeric(max_bins) || length(max_bins) != 1L) {
      rlang::abort("`max_bins` must be a single integer or tune().")
    }
    max_bins <- as.integer(max_bins)

    # Cross-validate with min_bins only if both are concrete values
    if (!rlang::is_quosure(min_bins) && !inherits(min_bins, "call")) {
      if (max_bins < min_bins) {
        rlang::abort("`max_bins` must be greater than or equal to `min_bins`.")
      }
    }
  }

  # Validate bin_cutoff (allow tune() placeholder)
  if (!rlang::is_quosure(bin_cutoff) && !inherits(bin_cutoff, "call")) {
    if (!is.numeric(bin_cutoff) || length(bin_cutoff) != 1L) {
      rlang::abort("`bin_cutoff` must be a single numeric value or tune().")
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      rlang::abort("`bin_cutoff` must be between 0 and 1 (exclusive).")
    }
  }

  # Validate na_woe
  if (!is.numeric(na_woe) || length(na_woe) != 1L) {
    rlang::abort("`na_woe` must be a single numeric value.")
  }

  # Validate suffixes
  if (!is.character(suffix_woe) || length(suffix_woe) != 1L) {
    rlang::abort("`suffix_woe` must be a single character string.")
  }
  if (!is.character(suffix_bin) || length(suffix_bin) != 1L) {
    rlang::abort("`suffix_bin` must be a single character string.")
  }

  # Validate control
  if (!is.list(control)) {
    rlang::abort("`control` must be a named list.")
  }

  recipes::add_step(
    recipe,
    step_obwoe_new(
      terms = rlang::enquos(...),
      role = role,
      trained = trained,
      outcome = outcome,
      algorithm = algorithm,
      min_bins = min_bins,
      max_bins = max_bins,
      bin_cutoff = bin_cutoff,
      output = output,
      suffix_woe = suffix_woe,
      suffix_bin = suffix_bin,
      na_woe = na_woe,
      control = control,
      binning_results = binning_results,
      skip = skip,
      id = id
    )
  )
}


#' @title Internal Constructor for step_obwoe
#'
#' @description
#' Creates a new step_obwoe object. This is an internal function and should
#' not be called directly by users.
#'
#' @inheritParams step_obwoe
#' @param terms A list of quosures specifying the variables to transform.
#'
#' @return A step_obwoe object.
#'
#' @keywords internal
step_obwoe_new <- function(terms,
                           role,
                           trained,
                           outcome,
                           algorithm,
                           min_bins,
                           max_bins,
                           bin_cutoff,
                           output,
                           suffix_woe,
                           suffix_bin,
                           na_woe,
                           control,
                           binning_results,
                           skip,
                           id) {
  recipes::step(
    subclass = "obwoe",
    terms = terms,
    role = role,
    trained = trained,
    outcome = outcome,
    algorithm = algorithm,
    min_bins = min_bins,
    max_bins = max_bins,
    bin_cutoff = bin_cutoff,
    output = output,
    suffix_woe = suffix_woe,
    suffix_bin = suffix_bin,
    na_woe = na_woe,
    control = control,
    binning_results = binning_results,
    skip = skip,
    id = id
  )
}


#' @title Prepare the Optimal Binning Step
#'
#' @description
#' Fits the optimal binning models on training data. This method is called
#' by \code{\link[recipes]{prep}} and should not be invoked directly.
#'
#' @param x A step_obwoe object.
#' @param training A tibble or data frame containing the training data.
#' @param info A tibble with column metadata from the recipe.
#' @param ... Additional arguments (currently unused).
#'
#' @return A trained step_obwoe object with \code{binning_results} populated.
#'
#' @export
#' @method prep step_obwoe
#' @importFrom utils modifyList
prep.step_obwoe <- function(x, training, info = NULL, ...) {
  col_names <- recipes::recipes_eval_select(x$terms, training, info)

  # Validate outcome exists in training data
  if (!x$outcome %in% names(training)) {
    rlang::abort(
      sprintf("Outcome column '%s' not found in training data.", x$outcome)
    )
  }

  # Remove outcome from selected columns if accidentally captured

  col_names <- setdiff(col_names, x$outcome)

  # Return early if no columns selected
  if (length(col_names) == 0L) {
    return(
      step_obwoe_new(
        terms = x$terms,
        role = x$role,
        trained = TRUE,
        outcome = x$outcome,
        algorithm = x$algorithm,
        min_bins = x$min_bins,
        max_bins = x$max_bins,
        bin_cutoff = x$bin_cutoff,
        output = x$output,
        suffix_woe = x$suffix_woe,
        suffix_bin = x$suffix_bin,
        na_woe = x$na_woe,
        control = x$control,
        binning_results = list(),
        skip = x$skip,
        id = x$id
      )
    )
  }

  # Resolve "auto" algorithm based on target type
  resolved_algorithm <- x$algorithm
  if (identical(resolved_algorithm, "auto")) {
    target_vec <- training[[x$outcome]]
    if (is.factor(target_vec)) {
      n_classes <- length(levels(target_vec))
    } else {
      n_classes <- length(unique(target_vec[!is.na(target_vec)]))
    }
    resolved_algorithm <- if (n_classes > 2L) "jedi_mwoe" else "jedi"
  }

  # Build control object with proper precedence
  # Explicit step parameters override control list values
  ctrl_base <- as.list(x$control)
  ctrl_override <- list(bin_cutoff = x$bin_cutoff)
  ctrl_merged <- modifyList(ctrl_base, ctrl_override)
  final_control <- do.call(control.obwoe, ctrl_merged)

  # Identify algorithm constraints for type checking
  num_only <- .numerical_only_algorithms()
  cat_only <- .categorical_only_algorithms()

  # Process each feature
  binning_results <- list()

  for (col in col_names) {
    feat_vec <- training[[col]]

    # Determine feature type
    is_numeric_feat <- is.numeric(feat_vec) && !is.factor(feat_vec)

    # Check algorithm-feature compatibility
    if (!is_numeric_feat && resolved_algorithm %in% num_only) {
      rlang::warn(
        sprintf(
          paste0(
            "Algorithm '%s' does not support categorical features. ",
            "Skipping variable '%s'."
          ),
          resolved_algorithm, col
        )
      )
      next
    }

    if (is_numeric_feat && resolved_algorithm %in% cat_only) {
      rlang::warn(
        sprintf(
          paste0(
            "Algorithm '%s' does not support numerical features. ",
            "Skipping variable '%s'."
          ),
          resolved_algorithm, col
        )
      )
      next
    }

    # Call obwoe for this feature
    res <- tryCatch(
      {
        obwoe(
          data = training[, c(col, x$outcome), drop = FALSE],
          target = x$outcome,
          feature = col,
          algorithm = resolved_algorithm,
          min_bins = x$min_bins,
          max_bins = x$max_bins,
          control = final_control
        )
      },
      error = function(e) {
        rlang::warn(
          sprintf(
            "Failed to bin variable '%s' with algorithm '%s': %s. Skipping.",
            col, resolved_algorithm, conditionMessage(e)
          )
        )
        return(NULL)
      }
    )

    # Validate result
    if (is.null(res)) {
      next
    }

    # Check for internal algorithm errors
    has_error <- any(res$summary$error, na.rm = TRUE)
    if (has_error) {
      rlang::warn(
        sprintf(
          "Algorithm '%s' reported an error for variable '%s'. Skipping.",
          resolved_algorithm, col
        )
      )
      next
    }

    # Extract and store only essential information for bake()
    feat_res <- res$results[[col]]

    stored_result <- list(
      type = feat_res$type,
      bin = feat_res$bin,
      woe = feat_res$woe,
      iv = if (!is.null(feat_res$iv)) feat_res$iv else rep(NA_real_, length(feat_res$bin)),
      cutpoints = feat_res$cutpoints,
      total_iv = res$summary$total_iv[1L],
      converged = if (!is.null(feat_res$converged)) feat_res$converged else NA
    )

    # Pre-compute category mapping for categorical features (efficiency)
    if (feat_res$type == "categorical") {
      cat_map_keys <- character(0L)
      cat_map_bins <- character(0L)
      cat_map_woes <- numeric(0L)

      for (i in seq_along(feat_res$bin)) {
        bin_label <- feat_res$bin[i]
        woe_val <- feat_res$woe[i]
        # Split merged categories
        parts <- trimws(strsplit(bin_label, "%;%", fixed = TRUE)[[1L]])
        cat_map_keys <- c(cat_map_keys, parts)
        cat_map_bins <- c(cat_map_bins, rep(bin_label, length(parts)))
        cat_map_woes <- c(cat_map_woes, rep(woe_val, length(parts)))
      }

      stored_result$cat_map_keys <- cat_map_keys
      stored_result$cat_map_bins <- cat_map_bins
      stored_result$cat_map_woes <- cat_map_woes
    }

    binning_results[[col]] <- stored_result
  }

  step_obwoe_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    outcome = x$outcome,
    algorithm = resolved_algorithm,
    min_bins = x$min_bins,
    max_bins = x$max_bins,
    bin_cutoff = x$bin_cutoff,
    output = x$output,
    suffix_woe = x$suffix_woe,
    suffix_bin = x$suffix_bin,
    na_woe = x$na_woe,
    control = x$control,
    binning_results = binning_results,
    skip = x$skip,
    id = x$id
  )
}


#' @title Apply the Optimal Binning Transformation
#'
#' @description
#' Applies the learned binning and WoE transformation to new data. This method
#' is called by \code{\link[recipes]{bake}} and should not be invoked directly.
#'
#' @param object A trained step_obwoe object.
#' @param new_data A tibble or data frame to transform.
#' @param ... Additional arguments (currently unused).
#'
#' @return A tibble with transformed columns according to the \code{output}
#'   parameter.
#'
#' @export
#' @method bake step_obwoe
#' @importFrom tibble as_tibble
bake.step_obwoe <- function(object, new_data, ...) {
  features <- names(object$binning_results)
  recipes::check_new_data(features, object, new_data)

  # Early return if no features were successfully binned
  if (length(features) == 0L) {
    return(tibble::as_tibble(new_data))
  }

  # Track columns to insert for output = "both"
  # We process all features first, then handle column ordering
  new_cols_data <- list()

  for (col in features) {
    res <- object$binning_results[[col]]
    vals <- new_data[[col]]
    n <- length(vals)

    # Initialize output vectors
    vec_woe <- rep(object$na_woe, n)
    vec_bin <- rep(NA_character_, n)

    if (res$type == "numerical") {
      # Numerical transformation using cutpoints
      if (is.null(res$cutpoints) || length(res$cutpoints) == 0L) {
        # Single bin case
        vec_bin <- rep(res$bin[1L], n)
        vec_woe <- rep(res$woe[1L], n)
        # Handle NAs
        na_mask <- is.na(vals)
        vec_woe[na_mask] <- object$na_woe
        vec_bin[na_mask] <- NA_character_
      } else {
        # Multiple bins
        breaks <- c(-Inf, res$cutpoints, Inf)
        vals_num <- as.numeric(vals)

        idx <- cut(
          vals_num,
          breaks = breaks,
          labels = FALSE,
          include.lowest = TRUE,
          right = TRUE
        )

        valid <- !is.na(idx)
        vec_bin[valid] <- res$bin[idx[valid]]
        vec_woe[valid] <- res$woe[idx[valid]]
        # NAs already have na_woe from initialization
      }
    } else {
      # Categorical transformation using pre-computed mapping
      vals_char <- as.character(vals)

      # Vectorized lookup using match
      idx <- match(vals_char, res$cat_map_keys)
      valid <- !is.na(idx)

      vec_bin[valid] <- res$cat_map_bins[idx[valid]]
      vec_woe[valid] <- res$cat_map_woes[idx[valid]]
      # NAs and unseen categories already have na_woe from initialization
    }

    # Apply transformation based on output mode
    if (object$output == "woe") {
      new_data[[col]] <- vec_woe
    } else if (object$output == "bin") {
      new_data[[col]] <- vec_bin
    } else {
      # output == "both": store new columns for later insertion
      new_cols_data[[col]] <- list(
        woe = vec_woe,
        bin = vec_bin
      )
    }
  }

  # Handle output = "both" with proper column ordering
  if (object$output == "both" && length(new_cols_data) > 0L) {
    # Process in reverse order of column position to maintain correct indices
    col_positions <- match(names(new_cols_data), names(new_data))
    process_order <- order(col_positions, decreasing = TRUE)

    for (idx in process_order) {
      col <- names(new_cols_data)[idx]
      col_idx <- which(names(new_data) == col)

      woe_name <- paste0(col, object$suffix_woe)
      bin_name <- paste0(col, object$suffix_bin)

      n_cols <- ncol(new_data)

      if (col_idx < n_cols) {
        # Insert after the original column
        before_cols <- names(new_data)[seq_len(col_idx)]
        after_cols <- names(new_data)[(col_idx + 1L):n_cols]

        new_data[[woe_name]] <- new_cols_data[[col]]$woe
        new_data[[bin_name]] <- new_cols_data[[col]]$bin

        new_order <- c(before_cols, woe_name, bin_name, after_cols)
        new_data <- new_data[, new_order, drop = FALSE]
      } else {
        # Column is last, just append
        new_data[[woe_name]] <- new_cols_data[[col]]$woe
        new_data[[bin_name]] <- new_cols_data[[col]]$bin
      }
    }
  }

  tibble::as_tibble(new_data)
}


#' @title Print Method for step_obwoe
#'
#' @description
#' Prints a concise summary of the step_obwoe object.
#'
#' @param x A step_obwoe object.
#' @param width Maximum width for printing term names.
#' @param ... Additional arguments (currently unused).
#'
#' @return Invisibly returns \code{x}.
#'
#' @export
#' @method print step_obwoe
print.step_obwoe <- function(x, width = max(20L, options()$width - 30L), ...) {
  title <- "Optimal Binning WoE"

  if (x$trained) {
    n_features <- length(x$binning_results)

    if (n_features > 0L) {
      # Calculate total IV across all features
      total_iv <- sum(
        vapply(
          x$binning_results,
          function(r) {
            if (!is.null(r$total_iv) && !is.na(r$total_iv)) {
              r$total_iv
            } else if (!is.null(r$iv)) {
              sum(r$iv, na.rm = TRUE)
            } else {
              0
            }
          },
          numeric(1L)
        ),
        na.rm = TRUE
      )

      cat(
        sprintf(
          "%s [trained, %d feature%s, total IV=%.4f, algorithm='%s']\n",
          title,
          n_features,
          if (n_features == 1L) "" else "s",
          total_iv,
          x$algorithm
        )
      )
    } else {
      cat(sprintf("%s [trained, 0 features]\n", title))
    }
  } else {
    terms_text <- recipes::sel2char(x$terms)
    terms_combined <- paste(terms_text, collapse = ", ")

    # Truncate if too long
    if (nchar(terms_combined) > width) {
      terms_combined <- paste0(substr(terms_combined, 1L, width - 3L), "...")
    }

    cat(sprintf("%s (%s) [algorithm='%s']\n", title, terms_combined, x$algorithm))
  }

  invisible(x)
}


#' @title Tidy Method for step_obwoe
#'
#' @description
#' Returns a tibble with information about the binning transformation. For
#' trained steps, returns one row per bin per feature, including bin labels,
#' WoE values, and IV contributions. For untrained steps, returns a placeholder
#' tibble.
#'
#' @param x A step_obwoe object.
#' @param ... Additional arguments (currently unused).
#'
#' @return A tibble with columns:
#' \describe{
#'   \item{terms}{Character. Feature name.}
#'   \item{bin}{Character. Bin label or interval.}
#'   \item{woe}{Numeric. Weight of Evidence value for the bin.}
#'   \item{iv}{Numeric. Information Value contribution of the bin.}
#'   \item{id}{Character. Step identifier.}
#' }
#'
#' @export
#' @method tidy step_obwoe
#' @importFrom tibble tibble as_tibble
tidy.step_obwoe <- function(x, ...) {
  if (x$trained) {
    if (length(x$binning_results) == 0L) {
      # No features processed successfully
      return(
        tibble::tibble(
          terms = character(0L),
          bin = character(0L),
          woe = numeric(0L),
          iv = numeric(0L),
          id = character(0L)
        )
      )
    }

    # Build tibble with one row per bin
    result_list <- lapply(names(x$binning_results), function(feat) {
      res <- x$binning_results[[feat]]

      n_bins <- length(res$bin)

      tibble::tibble(
        terms = rep(feat, n_bins),
        bin = res$bin,
        woe = res$woe,
        iv = if (length(res$iv) == n_bins) res$iv else rep(NA_real_, n_bins),
        id = rep(x$id, n_bins)
      )
    })

    out <- do.call(rbind, result_list)
    tibble::as_tibble(out)
  } else {
    # Untrained: return placeholder
    tibble::tibble(
      terms = recipes::sel2char(x$terms),
      bin = NA_character_,
      woe = NA_real_,
      iv = NA_real_,
      id = x$id
    )
  }
}


#' @title Tunable Parameters for step_obwoe
#'
#' @description
#' Returns information about which parameters of step_obwoe can be tuned
#' using the \code{tune} package.
#'
#' @param x A step_obwoe object.
#' @param ... Additional arguments (currently unused).
#'
#' @return A tibble describing tunable parameters.
#'
#' @export
#' @method tunable step_obwoe
#' @importFrom tibble tibble
tunable.step_obwoe <- function(x, ...) {
  tibble::tibble(
    name = c("algorithm", "min_bins", "max_bins", "bin_cutoff"),
    call_info = list(
      list(pkg = "OptimalBinningWoE", fun = "obwoe_algorithm"),
      list(pkg = "OptimalBinningWoE", fun = "obwoe_min_bins"),
      list(pkg = "OptimalBinningWoE", fun = "obwoe_max_bins"),
      list(pkg = "OptimalBinningWoE", fun = "obwoe_bin_cutoff")
    ),
    source = "recipe",
    component = "step_obwoe",
    component_id = x$id
  )
}


#' @title Required Packages for step_obwoe
#'
#' @description
#' Lists the packages required to execute the step_obwoe transformation.
#'
#' @param x A step_obwoe object.
#' @param ... Additional arguments (currently unused).
#'
#' @return A character vector of package names.
#'
#' @export
#' @method required_pkgs step_obwoe
required_pkgs.step_obwoe <- function(x, ...) {
  c("OptimalBinningWoE", "recipes")
}


# ============================================================================ #
# Dials Parameters for Hyperparameter Tuning
# ============================================================================ #

#' @title Binning Algorithm Parameter
#'
#' @description
#' A qualitative tuning parameter for selecting the optimal binning algorithm
#' in \code{\link{step_obwoe}}.
#'
#' @param values A character vector of algorithm names to include in the
#'   parameter space. If \code{NULL} (default), includes all 29 algorithms
#'   (28 specific algorithms plus \code{"auto"}).
#'
#' @return A \code{dials} qualitative parameter object.
#'
#' @details
#' The algorithms are organized into three groups:
#'
#' \strong{Universal} (support both numerical and categorical features):
#' \code{"auto"}, \code{"jedi"}, \code{"jedi_mwoe"}, \code{"cm"}, \code{"dp"},
#' \code{"dmiv"}, \code{"fetb"}, \code{"mob"}, \code{"sketch"}, \code{"udt"}
#'
#' \strong{Numerical only}:
#' \code{"bb"}, \code{"ewb"}, \code{"fast_mdlp"}, \code{"ir"}, \code{"kmb"},
#' \code{"ldb"}, \code{"lpdb"}, \code{"mblp"}, \code{"mdlp"}, \code{"mrblp"},
#' \code{"oslp"}, \code{"ubsd"}
#'
#' \strong{Categorical only}:
#' \code{"gmb"}, \code{"ivb"}, \code{"mba"}, \code{"milp"}, \code{"sab"},
#' \code{"sblp"}, \code{"swb"}
#'
#' When tuning with mixed feature types, consider restricting \code{values}
#' to universal algorithms only.
#'
#' @seealso \code{\link{step_obwoe}}, \code{\link{obwoe}}
#'
#' @examples
#' # Default: all algorithms
#' obwoe_algorithm()
#'
#' # Restrict to universal algorithms for mixed data
#' obwoe_algorithm(values = c("jedi", "mob", "dp", "cm"))
#'
#' # Numerical-only algorithms
#' obwoe_algorithm(values = c("mdlp", "fast_mdlp", "ewb", "ir"))
#'
#' @export
#' @importFrom dials new_qual_param
obwoe_algorithm <- function(values = NULL) {
  if (is.null(values)) {
    values <- .valid_algorithms()
  }

  dials::new_qual_param(
    type = "character",
    values = values,
    label = c(obwoe_algorithm = "Binning Algorithm")
  )
}


#' @title Minimum Bins Parameter
#'
#' @description
#' A quantitative tuning parameter for the minimum number of bins in
#' \code{\link{step_obwoe}}.
#'
#' @param range A two-element integer vector specifying the minimum and maximum
#'   values for the parameter. Default is \code{c(2L, 5L)}.
#' @param trans A transformation object from the \code{scales} package, or
#'   \code{NULL} for no transformation. Default is \code{NULL}.
#'
#' @return A \code{dials} quantitative parameter object.
#'
#' @details
#' The minimum number of bins constrains the algorithm to create at least
#' this many bins. Setting \code{min_bins = 2} allows maximum flexibility,
#' while higher values ensure more granular discretization.
#'
#' For credit scoring applications, \code{min_bins} is typically set between
#' 2 and 4 to avoid forcing artificial splits on weakly predictive variables.
#'
#' @seealso \code{\link{step_obwoe}}, \code{\link{obwoe_max_bins}}
#'
#' @examples
#' obwoe_min_bins()
#' obwoe_min_bins(range = c(3L, 7L))
#'
#' @export
#' @importFrom dials new_quant_param
obwoe_min_bins <- function(range = c(2L, 5L), trans = NULL) {
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(obwoe_min_bins = "Minimum Bins")
  )
}


#' @title Maximum Bins Parameter
#'
#' @description
#' A quantitative tuning parameter for the maximum number of bins in
#' \code{\link{step_obwoe}}.
#'
#' @param range A two-element integer vector specifying the minimum and maximum
#'   values for the parameter. Default is \code{c(5L, 20L)}.
#' @param trans A transformation object from the \code{scales} package, or
#'   \code{NULL} for no transformation. Default is \code{NULL}.
#'
#' @return A \code{dials} quantitative parameter object.
#'
#' @details
#' The maximum number of bins limits algorithm complexity and helps prevent
#' overfitting. Higher values allow more granular discretization but may
#' capture noise rather than signal.
#'
#' For credit scoring applications, \code{max_bins} is typically set between
#' 5 and 10 to balance predictive power with interpretability. Values above
#' 15 are rarely necessary and may indicate overfitting.
#'
#' @seealso \code{\link{step_obwoe}}, \code{\link{obwoe_min_bins}}
#'
#' @examples
#' obwoe_max_bins()
#' obwoe_max_bins(range = c(4L, 12L))
#'
#' @export
#' @importFrom dials new_quant_param
obwoe_max_bins <- function(range = c(5L, 20L), trans = NULL) {
  dials::new_quant_param(
    type = "integer",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(obwoe_max_bins = "Maximum Bins")
  )
}


#' @title Bin Cutoff Parameter
#'
#' @description
#' A quantitative tuning parameter for the minimum bin support (proportion
#' of observations per bin) in \code{\link{step_obwoe}}.
#'
#' @param range A two-element numeric vector specifying the minimum and maximum
#'   values for the parameter. Default is \code{c(0.01, 0.10)}.
#' @param trans A transformation object from the \code{scales} package, or
#'   \code{NULL} for no transformation. Default is \code{NULL}.
#'
#' @return A \code{dials} quantitative parameter object.
#'
#' @details
#' The bin cutoff specifies the minimum proportion of observations that each
#' bin must contain. Bins with fewer observations are merged with adjacent
#' bins. This serves as a regularization mechanism:
#'
#' \itemize{
#'   \item Lower values (e.g., 0.01) allow smaller bins, capturing subtle
#'     patterns but risking unstable WoE estimates.
#'   \item Higher values (e.g., 0.10) enforce larger bins, producing more
#'     stable estimates but potentially missing important patterns.
#' }
#'
#' For credit scoring, values between 0.02 and 0.05 are typical. Regulatory
#' guidelines often require minimum bin sizes for model stability.
#'
#' @seealso \code{\link{step_obwoe}}, \code{\link{control.obwoe}}
#'
#' @examples
#' obwoe_bin_cutoff()
#' obwoe_bin_cutoff(range = c(0.02, 0.08))
#'
#' @export
#' @importFrom dials new_quant_param
obwoe_bin_cutoff <- function(range = c(0.01, 0.10), trans = NULL) {
  dials::new_quant_param(
    type = "double",
    range = range,
    inclusive = c(TRUE, TRUE),
    trans = trans,
    label = c(obwoe_bin_cutoff = "Bin Support Cutoff")
  )
}


#' # ============================================================================#
#' # step_obwoe: Full-Spectrum Optimal Binning for tidymodels
#' # ============================================================================#
#'
#' #' @title Optimal Binning and WoE Transformation Step
#' #'
#' #' @description
#' #' \code{step_obwoe()} creates a \emph{specification} of a recipe step that
#' #' discretizes predictor variables using one of 28 state-of-the-art optimal
#' #' binning algorithms and transforms them into Weight of Evidence (WoE) values.
#' #' This step fully integrates the \strong{OptimalBinningWoE} package into the
#' #' \code{tidymodels} framework, supporting supervised discretization for both
#' #' binary and multinomial targets with extensive hyperparameter tuning capabilities.
#' #'
#' #' @inheritParams recipes::step_center
#' #' @param ... One or more selector functions to choose variables for this step.
#' #'   See \code{\link[recipes]{selections}}. Common choices: \code{all_predictors()},
#' #'   \code{all_numeric_predictors()}, or \code{all_nominal_predictors()}.
#' #'   \strong{Note:} Ensure the selected variables are compatible with the chosen
#' #'   \code{algorithm} (e.g., do not apply \code{"mdlp"} to categorical data).
#' #' @param role For variables created by this step, what role should they have?
#' #'   Default is \code{"predictor"}.
#' #' @param trained A logical indicating whether the step has been trained.
#' #' @param outcome A character string specifying the name of the response variable.
#' #'   This is \strong{required} as all algorithms are supervised. The outcome
#' #'   must exist in the training data provided to \code{prep()}.
#' #' @param algorithm Character string specifying the binning algorithm.
#' #'   Available options:
#' #'   \describe{
#' #'     \item{\strong{Universal (Num + Cat)}:}{\code{"auto"} (default), \code{"jedi"},
#' #'       \code{"jedi_mwoe"}, \code{"cm"}, \code{"dp"}, \code{"dmiv"}, \code{"fetb"},
#' #'       \code{"mob"}, \code{"sketch"}, \code{"udt"}}
#' #'     \item{\strong{Numerical Only}:}{\code{"bb"}, \code{"ewb"}, \code{"fast_mdlp"},
#' #'       \code{"ir"}, \code{"kmb"}, \code{"ldb"}, \code{"lpdb"}, \code{"mblp"},
#' #'       \code{"mdlp"}, \code{"mrblp"}, \code{"oslp"}, \code{"ubsd"}}
#' #'     \item{\strong{Categorical Only}:}{\code{"gmb"}, \code{"ivb"}, \code{"mba"},
#' #'       \code{"milp"}, \code{"sab"}, \code{"sblp"}, \code{"swb"}}
#' #'   }
#' #' @param min_bins Integer. Minimum number of bins (constraints). Default is 2.
#' #' @param max_bins Integer. Maximum number of bins (granularity). Default is 10.
#' #' @param bin_cutoff Numeric. Minimum fraction of observations per bin (0-1).
#' #'   Default is 0.05.
#' #' @param output Character string specifying the transformation output:
#' #'   \describe{
#' #'     \item{\code{"woe"}}{Replaces the original variable with WoE values (Default).}
#' #'     \item{\code{"bin"}}{Replaces the original variable with bin labels.}
#' #'     \item{\code{"both"}}{Keeps original, adds \code{_woe} and \code{_bin} columns.}
#' #'   }
#' #' @param suffix_woe Character. Suffix for WoE columns (used if \code{output="both"}).
#' #' @param suffix_bin Character. Suffix for bin columns (used if \code{output="both"}).
#' #' @param na_woe Numeric. WoE value to assign to missing values or unseen categories
#' #'   during \code{bake()}. Default is 0.
#' #' @param control Optional list of advanced parameters created by
#' #'   \code{\link{control.obwoe}}. If provided, overrides \code{bin_cutoff}.
#' #' @param binning_results Internal storage for fitted models. Do not set manually.
#' #' @param skip Logical. Should the step be skipped when baking? Default \code{FALSE}.
#' #' @param id Unique identifier for the step.
#' #'
#' #' @return An updated \code{recipe} object.
#' #'
#' #' @details
#' #' \subsection{Algorithm Selection Strategy}{
#' #' The \code{algorithm} parameter exposes the full power of the underlying engine.
#' #' Users should carefully select the algorithm matching their data types:
#' #' \itemize{
#' #'   \item Use \code{algorithm = "auto"} (default) for safe, automatic selection
#' #'     (\code{"jedi"} for binary targets).
#' #'   \item Use \code{algorithm = "mdlp"} or \code{"fast_mdlp"} for entropy-based
#' #'     discretization of numerical variables (requires \code{all_numeric_predictors()}).
#' #'   \item Use \code{algorithm = "mob"} (Monotonic Optimal Binning) when monotonic
#' #'     WoE trends are required for regulatory models (Basel/IFRS 9).
#' #' }
#' #' If an incompatible algorithm is applied to a variable (e.g., \code{"mdlp"} on a factor),
#' #' the step will issue a warning during \code{prep()} and skip that variable, leaving
#' #' it untransformed.
#' #' }
#' #'
#' #' \subsection{Tuning with Dials}{
#' #' This step is fully compatible with the \code{tune} package. The following
#' #' parameters are tunable:
#' #' \itemize{
#' #'   \item \code{algorithm}: See \code{\link{obwoe_algorithm}}.
#' #'   \item \code{min_bins}, \code{max_bins}: Structural constraints.
#' #'   \item \code{bin_cutoff}: Regularization parameter.
#' #' }
#' #' }
#' #'
#' #' @references
#' #' Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing
#' #' Intelligent Credit Scoring. \emph{John Wiley & Sons}.
#' #'
#' #' Navas-Palencia, G. (2020). Optimal Binning: Mathematical Programming
#' #' Formulation and Solution Approach. \emph{Expert Systems with Applications}.
#' #'
#' #' @seealso
#' #' \code{\link{obwoe}}, \code{\link{obwoe_algorithm}} for tuning grids.
#' #'
#' #' @examples
#' #' \donttest{
#' #' library(recipes)
#' #'
#' #' # Fake credit data
#' #' set.seed(123)
#' #' df <- data.frame(
#' #'   age = rnorm(100, 45, 10),
#' #'   income = exp(rnorm(100, 9, 0.5)),
#' #'   job = sample(c("Fixed", "PartTime", "Freelance"), 100, replace = TRUE),
#' #'   bad = rbinom(100, 1, 0.2)
#' #' )
#' #'
#' #' # Example 1: Universal JEDI algorithm on mixed data
#' #' rec_univ <- recipe(bad ~ ., data = df) %>%
#' #'   step_obwoe(all_predictors(), outcome = "bad", algorithm = "jedi")
#' #'
#' #' rec_univ_prepped <- prep(rec_univ)
#' #' baked_univ <- bake(rec_univ_prepped, new_data = NULL)
#' #' head(baked_univ)
#' #'
#' #' # Example 2: Specific Numerical Algorithm (MDLP)
#' #' # Note: We purposefully select only numeric predictors to avoid errors
#' #' rec_num <- recipe(bad ~ ., data = df) %>%
#' #'   step_obwoe(all_numeric_predictors(), outcome = "bad", algorithm = "mdlp")
#' #'
#' #' # Example 3: Outputting Bins and WoE (Both)
#' #' rec_both <- recipe(bad ~ ., data = df) %>%
#' #'   step_obwoe(age, outcome = "bad", output = "both")
#' #'
#' #' baked_both <- bake(prep(rec_both), new_data = df)
#' #' names(baked_both) # Contains age, age_woe, age_bin
#' #'
#' #' # Example 4: Tuning specification
#' #' # rec_tune <- recipe(bad ~ ., data = df) %>%
#' #' #   step_obwoe(all_predictors(), outcome = "bad",
#' #' #              algorithm = tune(), min_bins = tune())
#' #' }
#' #'
#' #' @export
#' #' @importFrom recipes add_step step rand_id recipes_eval_select check_type
#' #' @importFrom recipes check_new_data sel2char bake prep required_pkgs tidy
#' #' @importFrom recipes tunable
#' #' @importFrom rlang enquos abort warn
#' #' @importFrom tibble tibble as_tibble is_tibble
#' step_obwoe <- function(recipe,
#'                        ...,
#'                        role = "predictor",
#'                        trained = FALSE,
#'                        outcome = NULL,
#'                        algorithm = "auto",
#'                        min_bins = 2L,
#'                        max_bins = 10L,
#'                        bin_cutoff = 0.05,
#'                        output = c("woe", "bin", "both"),
#'                        suffix_woe = "_woe",
#'                        suffix_bin = "_bin",
#'                        na_woe = 0,
#'                        control = list(),
#'                        binning_results = NULL,
#'                        skip = FALSE,
#'                        id = recipes::rand_id("obwoe")) {
#'   output <- match.arg(output)
#'
#'   if (is.null(outcome)) {
#'     rlang::abort("The 'outcome' argument (target variable name) is required for supervised binning.")
#'   }
#'
#'   recipes::add_step(
#'     recipe,
#'     step_obwoe_new(
#'       terms = rlang::enquos(...),
#'       role = role,
#'       trained = trained,
#'       outcome = outcome,
#'       algorithm = algorithm,
#'       min_bins = min_bins,
#'       max_bins = max_bins,
#'       bin_cutoff = bin_cutoff,
#'       output = output,
#'       suffix_woe = suffix_woe,
#'       suffix_bin = suffix_bin,
#'       na_woe = na_woe,
#'       control = control,
#'       binning_results = binning_results,
#'       skip = skip,
#'       id = id
#'     )
#'   )
#' }
#'
#' step_obwoe_new <- function(terms, role, trained, outcome, algorithm,
#'                            min_bins, max_bins, bin_cutoff, output, suffix_woe,
#'                            suffix_bin, na_woe, control, binning_results,
#'                            skip, id) {
#'   recipes::step(
#'     subclass = "obwoe",
#'     terms = terms,
#'     role = role,
#'     trained = trained,
#'     outcome = outcome,
#'     algorithm = algorithm,
#'     min_bins = min_bins,
#'     max_bins = max_bins,
#'     bin_cutoff = bin_cutoff,
#'     output = output,
#'     suffix_woe = suffix_woe,
#'     suffix_bin = suffix_bin,
#'     na_woe = na_woe,
#'     control = control,
#'     binning_results = binning_results,
#'     skip = skip,
#'     id = id
#'   )
#' }
#'
#' #' @export
#' #' @importFrom utils modifyList
#' #' @method prep step_obwoe
#' prep.step_obwoe <- function(x, training, info = NULL, ...) {
#'   col_names <- recipes::recipes_eval_select(x$terms, training, info)
#'
#'   # Ensure outcome is available
#'   if (!x$outcome %in% names(training)) {
#'     rlang::abort(sprintf("Outcome column '%s' not found in training data.", x$outcome))
#'   }
#'
#'   # Avoid processing the outcome itself if selected by catch-all selectors
#'   col_names <- setdiff(col_names, x$outcome)
#'
#'   if (length(col_names) == 0) {
#'     # No variables selected, return empty results
#'     return(step_obwoe_new(
#'       terms = x$terms, role = x$role, trained = TRUE,
#'       outcome = x$outcome, algorithm = x$algorithm,
#'       min_bins = x$min_bins, max_bins = x$max_bins,
#'       bin_cutoff = x$bin_cutoff, output = x$output,
#'       suffix_woe = x$suffix_woe, suffix_bin = x$suffix_bin,
#'       na_woe = x$na_woe, control = x$control,
#'       binning_results = list(), skip = x$skip, id = x$id
#'     ))
#'   }
#'
#'   # Prepare control object
#'   # Merge explicit params with list params, explicit takes precedence defaults
#'   ctrl_args <- list(
#'     bin_cutoff = x$bin_cutoff
#'     # max_n_prebins etc can be in x$control
#'   )
#'   final_control <- do.call(
#'     control.obwoe,
#'     modifyList(ctrl_args, as.list(x$control))
#'   )
#'
#'   binning_results <- list()
#'
#'   for (col in col_names) {
#'     # We call obwoe() for each feature individually to handle errors per-feature
#'     # without stopping the whole recipe.
#'     res <- tryCatch(
#'       {
#'         obwoe(
#'           data = training[, c(col, x$outcome), drop = FALSE],
#'           target = x$outcome,
#'           feature = col,
#'           algorithm = x$algorithm,
#'           min_bins = x$min_bins,
#'           max_bins = x$max_bins,
#'           control = final_control
#'         )
#'       },
#'       error = function(e) {
#'         rlang::warn(sprintf(
#'           "Failed to bin variable '%s' with algorithm '%s': %s. Skipping.",
#'           col, x$algorithm, e$message
#'         ))
#'         return(NULL)
#'       }
#'     )
#'
#'     if (!is.null(res) && is.null(res$summary$error) || isFALSE(res$summary$error)) {
#'       binning_results[[col]] <- res
#'     } else if (!is.null(res) && !is.null(res$summary$error) && isTRUE(res$summary$error)) {
#'       # Algorithm internal error (captured by obwoe but returned as object)
#'       rlang::warn(sprintf(
#'         "Algorithm '%s' failed for variable '%s' (internal error). Skipping.",
#'         x$algorithm, col
#'       ))
#'     }
#'   }
#'
#'   step_obwoe_new(
#'     terms = x$terms,
#'     role = x$role,
#'     trained = TRUE,
#'     outcome = x$outcome,
#'     algorithm = x$algorithm,
#'     min_bins = x$min_bins,
#'     max_bins = x$max_bins,
#'     bin_cutoff = x$bin_cutoff,
#'     output = x$output,
#'     suffix_woe = x$suffix_woe,
#'     suffix_bin = x$suffix_bin,
#'     na_woe = x$na_woe,
#'     control = x$control,
#'     binning_results = binning_results,
#'     skip = x$skip,
#'     id = x$id
#'   )
#' }
#'
#' #' @rdname step_obwoe
#' #' @param object A prepped step_obwoe object.
#' #' @param new_data A tibble of new data to be processed.
#' #' @method bake step_obwoe
#' #' @importFrom recipes bake
#' #' @export
#' bake.step_obwoe <- function(object, new_data, ...) {
#'   # Get successfully binned features
#'   features <- names(object$binning_results)
#'   recipes::check_new_data(features, object, new_data)
#'
#'   # If no features were successfully binned, return data as is
#'   if (length(features) == 0) {
#'     return(tibble::as_tibble(new_data))
#'   }
#'
#'   for (col in features) {
#'     # Extract the obwoe object for this column
#'     model_obj <- object$binning_results[[col]]
#'     res <- model_obj$results[[col]] # Extract specific feature result
#'
#'     # Perform transformation logic locally to be fast and safe
#'     vals <- new_data[[col]]
#'     n <- length(vals)
#'
#'     # Prepare vectors
#'     vec_woe <- rep(object$na_woe, n)
#'     vec_bin <- rep(NA_character_, n)
#'
#'     if (res$type == "numerical") {
#'       # Numerical Transformation
#'       cuts <- c(-Inf, res$cutpoints, Inf)
#'       # Suppress warnings for NA values during cut
#'       idx <- suppressWarnings(cut(as.numeric(vals), breaks = cuts, labels = FALSE, include.lowest = TRUE, right = TRUE))
#'
#'       # Valid indices
#'       valid <- !is.na(idx)
#'       vec_bin[valid] <- res$bin[idx[valid]]
#'       vec_woe[valid] <- res$woe[idx[valid]]
#'     } else {
#'       # Categorical Transformation
#'       vals_char <- as.character(vals)
#'       # Build fast lookup environment
#'       map_bin <- new.env(hash = TRUE, parent = emptyenv())
#'       map_woe <- new.env(hash = TRUE, parent = emptyenv())
#'
#'       for (i in seq_along(res$bin)) {
#'         b_label <- res$bin[i]
#'         b_woe <- res$woe[i]
#'         parts <- strsplit(b_label, "%;%", fixed = TRUE)[[1]]
#'         for (p in parts) {
#'           p <- trimws(p)
#'           assign(p, b_label, envir = map_bin)
#'           assign(p, b_woe, envir = map_woe)
#'         }
#'       }
#'
#'       # Vectorized lookup not trivial with environments, standard loop or match
#'       # Using match is faster than env for moderate cardinality
#'       # Flatten the mapping
#'       # Re-do mapping strategy for bake efficiency:
#'       all_cats <- unlist(lapply(res$bin, function(x) trimws(strsplit(x, "%;%", fixed = TRUE)[[1]])))
#'       all_bins <- rep(res$bin, times = sapply(res$bin, function(x) length(strsplit(x, "%;%", fixed = TRUE)[[1]])))
#'       all_woes <- rep(res$woe, times = sapply(res$bin, function(x) length(strsplit(x, "%;%", fixed = TRUE)[[1]])))
#'
#'       matches <- match(vals_char, all_cats)
#'       valid <- !is.na(matches)
#'
#'       vec_bin[valid] <- all_bins[matches[valid]]
#'       vec_woe[valid] <- all_woes[matches[valid]]
#'     }
#'
#'     # Handle Output Formats
#'     if (object$output == "woe") {
#'       new_data[[col]] <- vec_woe
#'     } else if (object$output == "bin") {
#'       new_data[[col]] <- vec_bin
#'     } else {
#'       # "both"
#'       new_data[[paste0(col, object$suffix_woe)]] <- vec_woe
#'       new_data[[paste0(col, object$suffix_bin)]] <- vec_bin
#'       # Original column stays
#'     }
#'   }
#'
#'   tibble::as_tibble(new_data)
#' }
#'
#' #' @export
#' #' @method print step_obwoe
#' print.step_obwoe <- function(x, width = max(20, options()$width - 30), ...) {
#'   title <- "Optimal Binning & WoE Transformation"
#'
#'   if (x$trained) {
#'     n_features <- length(x$binning_results)
#'     cat(sprintf(
#'       "%s [trained] on %d features using '%s'\n",
#'       title, n_features, x$algorithm
#'     ))
#'   } else {
#'     terms <- recipes::sel2char(x$terms)
#'     cat(sprintf(
#'       "%s on %s (algorithm: %s)\n",
#'       title, paste(terms, collapse = ", "), x$algorithm
#'     ))
#'   }
#'   invisible(x)
#' }
#'
#' #' @export
#' #' @method tidy step_obwoe
#' tidy.step_obwoe <- function(x, ...) {
#'   if (x$trained) {
#'     # Extract info from binning_results
#'     res <- do.call(rbind, lapply(names(x$binning_results), function(feat) {
#'       obj <- x$binning_results[[feat]]
#'       # summary is a data.frame inside the obwoe object
#'       tibble::tibble(
#'         terms = feat,
#'         algorithm = obj$summary$algorithm,
#'         n_bins = obj$summary$n_bins,
#'         iv = obj$summary$total_iv,
#'         id = x$id
#'       )
#'     }))
#'     if (is.null(res)) {
#'       # If trained but no results (all failed)
#'       tibble::tibble(terms = character(), algorithm = character(), n_bins = integer(), iv = numeric(), id = character())
#'     } else {
#'       tibble::as_tibble(res)
#'     }
#'   } else {
#'     tibble::tibble(
#'       terms = recipes::sel2char(x$terms),
#'       algorithm = x$algorithm,
#'       n_bins = NA_integer_,
#'       iv = NA_real_,
#'       id = x$id
#'     )
#'   }
#' }
#'
#' #' @export
#' #' @method tunable step_obwoe
#' tunable.step_obwoe <- function(x, ...) {
#'   tibble::tibble(
#'     name = c("algorithm", "min_bins", "max_bins", "bin_cutoff"),
#'     call_info = list(
#'       list(pkg = "OptimalBinningWoE", fun = "obwoe_algorithm"),
#'       list(pkg = "OptimalBinningWoE", fun = "obwoe_min_bins"),
#'       list(pkg = "OptimalBinningWoE", fun = "obwoe_max_bins"),
#'       list(pkg = "OptimalBinningWoE", fun = "obwoe_bin_cutoff")
#'     ),
#'     source = "recipe",
#'     component = "step_obwoe",
#'     component_id = x$id
#'   )
#' }
#'
#' #' @export
#' #' @method required_pkgs step_obwoe
#' required_pkgs.step_obwoe <- function(x, ...) {
#'   c("OptimalBinningWoE", "recipes")
#' }
#'
#' # ============================================================================#
#' # Dials Parameters (Hyperparameter Tuning)
#' # ============================================================================#
#'
#' #' @title Tuning Parameters for Optimal Binning
#' #' @description
#' #' \code{dials} parameter objects to enable hyperparameter tuning of the
#' #' \code{step_obwoe} recipe step.
#' #'
#' #' @param values A character vector of algorithms. If \code{NULL}, returns all 28 supported algorithms.
#' #' @param range A two-element vector specifying min/max limits.
#' #' @param trans A trans object from the scales package.
#' #'
#' #' @return A \code{dials} parameter object.
#' #' @name obwoe-dials
#' NULL
#'
#' #' @rdname obwoe-dials
#' #' @export
#' #' @importFrom dials new_qual_param
#' obwoe_algorithm <- function(values = NULL) {
#'   if (is.null(values)) {
#'     # Full list of 28 algorithms + auto
#'     values <- c(
#'       "auto",
#'       # Universal
#'       "jedi", "jedi_mwoe", "cm", "dp", "dmiv", "fetb", "mob", "sketch", "udt",
#'       # Categorical
#'       "gmb", "ivb", "mba", "milp", "sab", "sblp", "swb",
#'       # Numerical
#'       "bb", "ewb", "fast_mdlp", "ir", "kmb", "ldb", "lpdb", "mblp",
#'       "mdlp", "mrblp", "oslp", "ubsd"
#'     )
#'   }
#'   dials::new_qual_param(
#'     type = "character",
#'     values = values,
#'     label = c(obwoe_algorithm = "Binning Algorithm")
#'   )
#' }
#'
#' #' @rdname obwoe-dials
#' #' @export
#' #' @importFrom dials new_quant_param
#' obwoe_min_bins <- function(range = c(2L, 5L), trans = NULL) {
#'   dials::new_quant_param(
#'     type = "integer",
#'     range = range,
#'     inclusive = c(TRUE, TRUE),
#'     trans = trans,
#'     label = c(obwoe_min_bins = "Minimum Bins")
#'   )
#' }
#'
#' #' @rdname obwoe-dials
#' #' @export
#' #' @importFrom dials new_quant_param
#' obwoe_max_bins <- function(range = c(5L, 20L), trans = NULL) {
#'   dials::new_quant_param(
#'     type = "integer",
#'     range = range,
#'     inclusive = c(TRUE, TRUE),
#'     trans = trans,
#'     label = c(obwoe_max_bins = "Maximum Bins")
#'   )
#' }
#'
#' #' @rdname obwoe-dials
#' #' @export
#' #' @importFrom dials new_quant_param
#' obwoe_bin_cutoff <- function(range = c(0.01, 0.10), trans = NULL) {
#'   dials::new_quant_param(
#'     type = "double",
#'     range = range,
#'     inclusive = c(TRUE, TRUE),
#'     trans = trans,
#'     label = c(obwoe_bin_cutoff = "Bin Support Cutoff")
#'   )
#' }
#' # Internal util to merge lists (backport from utils if needed, or simple implementation)
