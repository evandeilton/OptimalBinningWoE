#' @title Optimized Logistic Regression
#' @description
#' Fits logistic regression models using an optimized C++ implementation via Rcpp.
#'
#' @param formula An object of class \code{formula} describing the model to be fitted.
#' @param data A data frame or data.table containing the model data.
#' @param max_iter Maximum number of iterations for the optimization algorithm. Default is 1000.
#' @param tol Convergence tolerance for the optimization algorithm. Default is 1e-6.
#'
#' @return An object of class \code{oblr} containing the results of the logistic regression fit, including:
#' \describe{
#'   \item{coefficients}{Vector of estimated coefficients.}
#'   \item{se}{Standard errors of the coefficients.}
#'   \item{z_scores}{Z-statistics for the coefficients.}
#'   \item{p_values}{P-values for the coefficients.}
#'   \item{loglikelihood}{Log-likelihood of the model.}
#'   \item{convergence}{Convergence indicator.}
#'   \item{iterations}{Number of iterations performed.}
#'   \item{message}{Convergence message.}
#'   \item{data}{List containing the design matrix X, response y, and the function call.}
#' }
#'
#' @details
#' The \code{oblr} function fits a logistic regression model using an optimized C++
#' implementation via Rcpp. This implementation is designed to be efficient, especially
#' for large or sparse datasets.
#'
#' The logistic regression model is defined as:
#'
#' \deqn{P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + ... + \beta_p X_p)}}}
#'
#' where \eqn{\beta} are the coefficients to be estimated.
#'
#' The optimization method used is L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno),
#' a variant of the BFGS method that uses a limited amount of memory. This method is
#' particularly effective for optimization problems with many variables.
#'
#' The estimation process involves the following steps:
#' 1. Data preparation: The design matrix X is created using \code{sparse.model.matrix}
#'    from the Matrix package, which is efficient for sparse data.
#' 2. Optimization: The C++ function \code{fit_logistic_regression} is called to perform
#'    the optimization using L-BFGS.
#' 3. Statistics calculation: Standard errors, z-statistics, and p-values are calculated
#'    using the Hessian matrix returned by the optimization function.
#'
#' Convergence is determined by the relative change in the objective function
#' (log-likelihood) between successive iterations, compared to the specified tolerance.
#'
#' @examples
#' \dontrun{
#' library(data.table)
#'
#' # Create example data
#' set.seed(123)
#' n <- 10000
#' X1 <- rnorm(n)
#' X2 <- rnorm(n)
#' Y <- rbinom(n, 1, plogis(1 + 0.5 * X1 - 0.5 * X2))
#' dt <- data.table(Y, X1, X2)
#'
#' # Fit logistic regression model
#' model <- oblr(Y ~ X1 + X2, data = dt, max_iter = 1000, tol = 1e-6)
#'
#' # View results
#' print(model)
#' }
#' @importFrom Matrix sparse.model.matrix
#' @export
oblr <- function(formula, data, max_iter = 1000, tol = 1e-6) {
  # Capture the function call
  call_ <- match.call()

  # Validate input arguments
  if (!inherits(formula, "formula")) {
    stop("'formula' must be an object of class 'formula'")
  }
  if (!is.data.frame(data) && !data.table::is.data.table(data)) {
    stop("'data' must be a data.frame or data.table")
  }
  if (!is.numeric(max_iter) || max_iter <= 0) {
    stop("'max_iter' must be a positive number")
  }
  if (!is.numeric(tol) || tol <= 0) {
    stop("'tol' must be a positive number")
  }

  # Prepare the data
  mf <- model.frame(formula, data)
  y <- model.response(mf)
  X <- Matrix::sparse.model.matrix(formula, data)

  # Fit the model using the optimized implementation
  fit <- fit_logistic_regression(X, y, maxit = max_iter, eps_f = tol)

  # Add the call to the return object
  fit$data <- list(X = X, y = y, call = call_)

  class(fit) <- "oblr"
  # Return the fit object
  return(fit)
}

#' @title Print Method for oblr Objects
#' @description
#' Prints a brief summary of the `oblr` model, including estimated coefficients and convergence information.
#'
#' @param x An object of class `oblr`.
#' @param digits Number of significant digits to display. Defaults to the maximum between 3 and \code{getOption("digits") - 3}.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @export
print.oblr <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Call:\n")
  print(x$data$call)
  cat("\nCoefficients:\n")

  co <- x$coefficients
  names(co) <- colnames(x$data$X)
  print(co, digits = digits)

  cat("\nConvergence:", if (x$convergence == 0) "Successful" else "Not Successful", "\n")
  cat("Iterations:", x$iterations, "\n")
  if (!is.null(x$message)) {
    cat("Message:", x$message, "\n")
  }
  invisible(x)
}

#' @title Summary Method for oblr Objects
#' @description
#' Provides a detailed summary of the `oblr` model, including coefficients, standard errors, z-values, p-values, and model fit statistics.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return An object of class `summary.oblr` containing the model summary.
#'
#' @export
summary.oblr <- function(object, ...) {
  coef <- object$coefficients
  se <- object$se
  z <- object$z_scores
  p <- object$p_values
  y <- object$data$y
  X <- object$data$X
  n <- length(y)

  coef.table <- cbind(coef, se, z, p)
  dimnames(coef.table) <- list(
    names(coef),
    c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
  )
  rownames(coef.table) <- colnames(X)

  # Calculate null and residual deviance

  fitted.values <- 1 / (1 + exp(-as.vector(X %*% coef)))

  null.deviance <- -2 * sum(y * log(mean(y)) + (1 - y) * log(1 - mean(y)))
  deviance <- -2 * object$loglikelihood

  df.null <- n - 1
  df.residual <- n - length(coef)

  # Calculate AIC e BIC
  aic <- deviance + 2 * length(coef)
  bic <- deviance + log(n) * length(coef)

  # Prepare the summary object
  ans <- list(
    call = object$data$call,
    family = binomial(),
    coefficients = coef.table,
    deviance = deviance,
    null.deviance = null.deviance,
    df.residual = df.residual,
    df.null = df.null,
    aic = aic,
    bic = bic,
    logLik = object$loglikelihood,
    iter = object$iterations,
    converged = object$convergence,
    n = n
  )

  class(ans) <- "summary.oblr"
  return(ans)
}


#' @title Print Method for summary.oblr Objects
#' @description
#' Prints a detailed summary of the `oblr` model, including coefficients, standard errors, z-values, p-values, and model fit statistics.
#'
#' @param x An object of class `summary.oblr`.
#' @param digits Number of significant digits to display. Defaults to the maximum between 3 and \code{getOption("digits") - 3}.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @export
print.summary.oblr <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat("Call           :\n")
  print(x$call)

  cat("\nCoefficients :\n")
  printCoefmat(x$coefficients, digits = digits, signif.stars = TRUE)
  cat("\n")

  cat("Deviance       :", format(x$deviance, digits = digits), "\n")
  cat("AIC            :", format(x$aic, digits = digits), "\n")
  cat("BIC            :", format(x$bic, digits = digits), "\n")
  cat("Log-Likelihood :", format(x$logLik, digits = digits), "\n")
  # cat("Iterations     :", x$iter, "\n")
  cat("Convergence    :", if (x$converged == 1) "Successful" else "Not Successful", "\n")

  invisible(x)
}

#' @title Coefficients Method for oblr Objects
#' @description
#' Extracts the estimated coefficients from the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A numeric vector of estimated coefficients.
#'
#' @export
coef.oblr <- function(object, ...) {
  co <- object$coefficients
  names(co) <- colnames(object$data$X)
  return(co)
}

#' @title Log-Likelihood Method for oblr Objects
#' @description
#' Returns the log-likelihood of the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return An object of class `logLik` containing the log-likelihood.
#'
#' @export
logLik.oblr <- function(object, ...) {
  structure(object$loglikelihood,
    df = length(object$coefficients),
    class = "logLik"
  )
}

#' @title AIC Method for oblr Objects
#' @description
#' Calculates the Akaike Information Criterion (AIC) for the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#' @param k The penalty per parameter to be used in the AIC calculation. Default is 2.
#'
#' @return A numeric value representing the AIC.
#'
#' @export
AIC.oblr <- function(object, ..., k = 2) {
  return(-2 * object$loglikelihood + k * length(object$coefficients))
}

#' @title BIC Method for oblr Objects
#' @description
#' Calculates the Bayesian Information Criterion (BIC) for the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A numeric value representing the BIC.
#'
#' @export
BIC.oblr <- function(object, ...) {
  n <- length(object$data$y)
  return(-2 * object$loglikelihood + log(n) * length(object$coefficients))
}

#' Register the S3 method
#' @param object obrl class fit
#' @param ... Additional arguments passed to or from other methods.
#' @export
BIC <- function(object, ...) {
  UseMethod("BIC.oblr")
}

#' @title Variance-Covariance Matrix Method for oblr Objects
#' @description
#' Returns the variance-covariance matrix of the estimated coefficients from the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A variance-covariance matrix of the estimated coefficients.
#'
#' @export
vcov.oblr <- function(object, ...) {
  if (is.null(object$se)) {
    stop("Standard errors are not available.")
  }
  ans <- solve(object$hessian)
  colnames(ans) <- rownames(ans) <- colnames(object$data$X)
  return(ans)
}


#' @title Predict Method for oblr Objects
#' @description
#' Generates predictions from the fitted `oblr` model. Can return probabilities, link values, or class predictions.
#'
#' @param object An object of class `oblr`.
#' @param newdata A data frame or data.table containing new data for prediction. If \code{NULL}, uses the data from the fit.
#' @param type The type of prediction to return: \code{"link"} for the linear predictor, \code{"proba"} for probabilities, or \code{"class"} for class predictions.
#' @param cutoff The probability cutoff for class prediction. Default is 0.5. Only used when \code{type = "class"}.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A numeric vector of predictions. For \code{type = "class"}, returns a factor with levels 0 and 1.
#' @importFrom Matrix sparse.model.matrix
#' @export
predict.oblr <- function(object, newdata = NULL, type = c("proba", "class", "link"), cutoff = 0.5, ...) {
  type <- match.arg(type)

  if (!is.numeric(cutoff) || cutoff < 0 || cutoff > 1) {
    stop("cutoff must be a numeric value between 0 and 1")
  }

  if (is.null(newdata)) {
    X <- object$data$X
  } else {
    mf <- model.frame(object$data$call$formula, newdata)
    X <- Matrix::sparse.model.matrix(object$data$call$formula, newdata)
  }

  eta <- as.numeric(X %*% object$coefficients)

  if (type == "link") {
    return(eta)
  } else if (type == "proba") {
    return(plogis(eta))
  } else if (type == "class") {
    probs <- plogis(eta)
    classes <- factor(ifelse(probs > cutoff, 1, 0), levels = c(0, 1))
    return(classes)
  }
}


#' @title Residuals Method for oblr Objects
#' @description
#' Calculates residuals from the `oblr` model, such as deviance, Pearson, and others.
#'
#' @details
#' The following types of residuals can be calculated:
#'
#' - **Raw Residuals**: The difference between observed and predicted values:
#' \deqn{e_i = y_i - \hat{y}_i}
#'
#' - **Deviance Residuals**: Deviance residuals measure the contribution of each observation to the model deviance. For logistic regression, it is defined as:
#' \deqn{e_i^{\text{Deviance}} = \text{sign}(y_i - \hat{y}_i) \sqrt{2 \left[ y_i \log\left(\frac{y_i}{\hat{y}_i}\right) + (1 - y_i) \log\left(\frac{1 - y_i}{1 - \hat{y}_i}\right) \right]}}
#' where \eqn{\hat{y}_i} is the predicted probability, and \eqn{y_i} is the observed value.
#'
#' - **Pearson Residuals**: These residuals scale the raw residuals by the estimated standard deviation:
#' \deqn{e_i^{\text{Pearson}} = \frac{y_i - \hat{y}_i}{\sqrt{\hat{y}_i (1 - \hat{y}_i)}}}
#' Pearson residuals are used to assess goodness of fit in generalized linear models.
#'
#' - **Standardized Residuals**: These residuals standardize the raw residuals by dividing by the estimated standard deviation, adjusting for the fitted values:
#' \deqn{e_i^{\text{Standardized}} = \frac{e_i}{\sqrt{\hat{y}_i (1 - \hat{y}_i)}}}
#'
#' - **Internally Studentized Residuals**: These residuals account for the leverage (influence) of each observation on its own fitted value:
#' \deqn{e_i^{\text{Internally Studentized}} = \frac{e_i}{\sqrt{\hat{y}_i (1 - \hat{y}_i)(1 - h_i)}}}
#' where \eqn{h_i} is the leverage for the \eqn{i}-th observation, calculated from the hat matrix.
#'
#' - **Externally Studentized Residuals**: These residuals are similar to internally studentized residuals but exclude the \eqn{i}-th observation when estimating the variance:
#' \deqn{e_i^{\text{Externally Studentized}} = \frac{e_i}{\hat{\sigma}_{(i)} \sqrt{1 - h_i}}}
#' where \eqn{\hat{\sigma}_{(i)}} is the estimated standard error excluding the \eqn{i}-th observation.
#'
#' - **Leverage-Adjusted Residuals**: These residuals adjust the raw residuals by the leverage value \eqn{h_i}:
#' \deqn{e_i^{\text{Leverage-Adjusted}} = \frac{e_i}{\sqrt{1 - h_i}}}
#'
#' @param object An object of class `oblr`.
#' @param type The type of residuals to calculate: \code{"deviance"}, \code{"pearson"}, \code{"raw"}, \code{"standardized"}, \code{"studentized_internal"}, \code{"studentized_external"}, \code{"leverage_adjusted"}.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A numeric vector of residuals.
#'
#' @export
residuals.oblr <- function(object, type = c("deviance", "pearson", "raw", "standardized", "studentized_internal", "studentized_external", "leverage_adjusted"), ...) {
  type <- match.arg(type)

  if (is.null(object$data$X)) {
    stop("No data available to compute residuals.")
  }

  # Extracting necessary data
  y <- object$data$y
  p <- plogis(as.numeric(object$data$X %*% object$coefficients))
  n <- length(y)

  # Hat matrix diagonal (leverage values)
  X <- object$data$X
  h_ii <- diag(as.matrix(X %*% solve(t(X) %*% X) %*% t(X))) # Leverage for each observation

  # Calculating residuals based on type
  if (type == "raw") {
    # Raw residuals: difference between observed and predicted values
    res <- y - p
  } else if (type == "deviance") {
    # Deviance residuals
    res <- ifelse(y == 1, -log(p), -log(1 - p))
    res <- 2 * (res - (y * log(y + (y == 0)) + (1 - y) * log(1 - y + (y == 1))))
  } else if (type == "pearson") {
    # Pearson residuals
    res <- (y - p) / sqrt(p * (1 - p))
  } else if (type == "standardized") {
    # Standardized residuals: raw residuals divided by the standard error
    raw_res <- y - p
    res <- raw_res / sqrt(p * (1 - p))
  } else if (type == "studentized_internal") {
    # Internally studentized residuals: standardized residuals adjusted for leverage
    raw_res <- y - p
    res <- raw_res / (sqrt(p * (1 - p)) * sqrt(1 - h_ii))
  } else if (type == "studentized_external") {
    # Externally studentized residuals: studentized residuals excluding the i-th observation
    raw_res <- y - p
    sigma_i <- sqrt(sum((y - p)^2) / (n - length(object$coefficients)))
    res <- raw_res / (sigma_i * sqrt(1 - h_ii))
  } else if (type == "leverage_adjusted") {
    # Leverage-adjusted residuals: adjusts raw residuals by leverage
    raw_res <- y - p
    res <- raw_res / sqrt(1 - h_ii)
  }

  return(res)
}

#' @title Fitted Values Method for oblr Objects
#' @description
#' Returns the fitted values (predicted probabilities) from the `oblr` model.
#'
#' @param object An object of class `oblr`.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A numeric vector of fitted values.
#'
#' @export
fitted.oblr <- function(object, ...) {
  return(predict(object, type = "proba"))
}


#' @title Update Method for oblr Objects
#' @description
#' Updates the `oblr` model with new parameters without refitting the entire model.
#'
#' @param object An object of class `oblr`.
#' @param formula. A new formula for the model. If not specified, the original formula is retained.
#' @param data. New data for fitting the model. If not specified, the original data is retained.
#' @param ... Additional arguments passed to or from other methods.
#'
#' @return A new object of class `oblr` fitted with the updated parameters.
#'
#' @export
update.oblr <- function(object, formula., data., ...) {
  call <- object$data$call
  if (!missing(formula.)) {
    call$formula <- formula.
  }
  if (!missing(data.)) {
    call$data <- data.
  }
  call[[1]] <- as.name("oblr")
  eval(call, parent.frame())
}


#' Anova Method for oblr Objects
#'
#' This function performs an analysis of variance (or more precisely, an analysis of deviance)
#' for one or more fitted logistic regression model objects of class 'oblr'.
#'
#' @param object An object of class "oblr", typically the result of a call to oblr().
#' @param ... Additional objects of class "oblr", or a single object of class "list" containing only objects of class "oblr".
#' @param test A character string specifying the test statistic to be used.
#'             Can be one of "Chisq" (default) for likelihood ratio test,
#'             "F" for F-test, or "none" to skip significance testing.
#'
#' @return An object of class "anova" inheriting from class "data.frame".
#'
#' @importFrom stats pchisq pf
#'
#' @export
anova.oblr <- function(object, ..., test = c("Chisq", "F", "none")) {
  test <- match.arg(test)

  # Collect all models
  models <- c(list(object), list(...))

  # If a single list is passed, unpack it
  # if (length(models) == 2 && is.list(models[[2]])) {
  #   models <- c(list(object), models[[2]])
  # }

  # Check if all objects are of class 'oblr'
  if (!all(sapply(models, inherits, "oblr"))) {
    stop("All objects must be of class 'oblr'")
  }

  # Extract relevant information from each model
  n_obs <- length(models[[1]]$data$y)
  resid_df <- sapply(models, function(m) n_obs - length(m$coefficients))
  resid_dev <- -2 * sapply(models, function(m) m$loglikelihood)

  # Create the ANOVA table
  anova_table <- data.frame(
    Resid.Df = resid_df,
    Resid.Dev = resid_dev
  )

  # Add model names if available
  if (!is.null(names(models))) {
    row.names(anova_table) <- names(models)
  } else {
    row.names(anova_table) <- paste("Model", seq_along(models))
  }

  # Calculate changes in df and deviance
  if (length(models) > 1) {
    anova_table$Df <- c(NA, -diff(resid_df))
    anova_table$Deviance <- c(NA, -diff(resid_dev))

    if (test == "Chisq") {
      anova_table$`Pr(>Chi)` <- c(NA, pchisq(anova_table$Deviance[-1],
        df = anova_table$Df[-1],
        lower.tail = FALSE
      ))
    } else if (test == "F") {
      f_stats <- anova_table$Deviance[-1] / anova_table$Df[-1] /
        (anova_table$Resid.Dev[-1] / anova_table$Resid.Df[-1])
      anova_table$`F value` <- c(NA, f_stats)
      anova_table$`Pr(>F)` <- c(NA, pf(f_stats,
        df1 = anova_table$Df[-1],
        df2 = anova_table$Resid.Df[-1],
        lower.tail = FALSE
      ))
    }
  }

  # Set the class of the anova table
  class(anova_table) <- c("anova", "data.frame")

  attr(anova_table, "heading") <- c(
    "Analysis of Deviance Table\n",
    paste("Model:", deparse(models[[1]]$data$call),
      collapse = "\n"
    )
  )

  return(anova_table)
}

#' @title Compute Performance Metrics for Logistic Regression Models
#' @description Calculates various performance metrics for an oblr model.
#' @param object An object of class "oblr".
#' @param newdata A data frame or data.table containing new data for evaluation. If NULL, uses the data from the fit.
#' @param cutoff The probability cutoff for class prediction. Default is 0.5.
#' @return A data.table with the calculated metrics.
#' @details
#' This function calculates the following metrics:
#'
#' 1. Log-likelihood (LogLik):
#' \deqn{LogLik = \sum_{i=1}^n [y_i \log(p_i) + (1-y_i) \log(1-p_i)]}
#' where \eqn{y_i} are the observed values and \eqn{p_i} are the predicted probabilities.
#'
#' 2. Akaike Information Criterion (AIC):
#' \deqn{AIC = 2k - 2LogLik}
#' where \eqn{k} is the number of parameters in the model.
#'
#' 3. Bayesian Information Criterion (BIC):
#' \deqn{BIC = k\log(n) - 2LogLik}
#' where \eqn{n} is the number of observations.
#'
#' 4. Area Under the ROC Curve (AUC):
#' AUC is the area under the Receiver Operating Characteristic curve, which plots the true positive rate against the false positive rate.
#'
#' 5. Gini Coefficient:
#' \deqn{Gini = 2 * AUC - 1}
#'
#' 6. Kolmogorov-Smirnov Statistic (KS):
#' \deqn{KS = \max|F_1(x) - F_0(x)|}
#' where \eqn{F_1(x)} and \eqn{F_0(x)} are the cumulative distribution functions for the positive and negative classes, respectively.
#'
#' 7. Accuracy:
#' \deqn{Accuracy = \frac{TP + TN}{TP + TN + FP + FN}}
#' where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.
#'
#' 8. Recall (Sensitivity):
#' \deqn{Recall = \frac{TP}{TP + FN}}
#'
#' 9. Precision:
#' \deqn{Precision = \frac{TP}{TP + FP}}
#'
#' 10. F1-Score:
#' \deqn{F1 = 2 * \frac{Precision * Recall}{Precision + Recall}}
#'
#' These metrics provide a comprehensive view of the model's performance,
#' including its predictive capability (AUC, KS), fit to the data (LogLik, AIC, BIC),
#' and performance in classification tasks (Accuracy, Recall, Precision, F1-Score).
#'
#' @importFrom data.table data.table
#' @importFrom pROC roc auc
#' @export
computeMetrics <- function(object, newdata = NULL, cutoff = 0.5) {
  if (!inherits(object, "oblr")) {
    stop("The object must be of class 'oblr'")
  }

  # Get predictions
  y_true <- if (is.null(newdata)) object$data$y else newdata[[all.vars(object$data$call$formula)[1]]]
  y_pred_proba <- predict(object, newdata = newdata, type = "proba")
  y_pred_class <- predict(object, newdata = newdata, type = "class", cutoff = cutoff)

  # Calculate metrics
  loglik <- object$loglikelihood
  k <- length(object$coefficients)
  n <- length(y_true)

  aic <- 2 * k - 2 * loglik
  bic <- log(n) * k - 2 * loglik

  roc_obj <- pROC::roc(y_true, y_pred_proba, quiet = TRUE)
  auc <- as.numeric(pROC::auc(roc_obj))
  gini <- 2 * auc - 1

  ks <- max(abs(cumsum(y_true[order(y_pred_proba)]) / sum(y_true) -
    cumsum(!y_true[order(y_pred_proba)]) / sum(!y_true)))

  confusion_matrix <- table(Actual = y_true, Predicted = y_pred_class)
  accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

  if (nrow(confusion_matrix) == 2 && ncol(confusion_matrix) == 2) {
    recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
    precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
    f1_score <- 2 * (precision * recall) / (precision + recall)
  } else {
    recall <- precision <- f1_score <- NA
  }

  metrics <- data.table::data.table(
    LogLik = loglik,
    AIC = aic,
    BIC = bic,
    AUC = auc,
    Gini = gini,
    KS = ks,
    Accuracy = accuracy,
    Recall = recall,
    Precision = precision,
    F1_Score = f1_score
  )

  return(metrics)
}
