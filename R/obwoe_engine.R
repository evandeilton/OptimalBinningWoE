# ============================================================================ #
# Model engines for obwoe_scorecard()                                          #
# ============================================================================ #

#' @title Internal: The Engine Contract
#'
#' @description
#' Every engine is three functions and nothing else:
#' \describe{
#'   \item{\code{fit(x, y, args)}}{\code{x} is a numeric matrix of Weight of
#'     Evidence columns with no intercept column, \code{y} an integer 0/1
#'     vector. Returns an opaque fitted object.}
#'   \item{\code{link(object, x)}}{Returns the log-odds of the \strong{event},
#'     one per row of \code{x}. Mandatory.}
#'   \item{\code{coef(object)}}{Returns a named numeric vector with
#'     \code{"(Intercept)"} first, \strong{or \code{NULL}} to declare that the
#'     model is not additive in the WoE predictors.}
#' }
#'
#' The \code{NULL} return is the honest-degradation switch. A scorecard's
#' points table exists only because the model is a sum of per-variable terms;
#' a tree ensemble is not, so it cannot be decomposed into points and the
#' pipeline must say so rather than fabricate a table.
#'
#' @return A named list of engine definitions.
#'
#' @keywords internal
.ob_engine_registry <- function() {
  list(
    glm = list(
      pkgs = character(0),
      additive = TRUE,
      fit = function(x, y, args) {
        d <- data.frame(.y = as.integer(y), as.data.frame(x, check.names = FALSE),
          check.names = FALSE
        )
        do.call(stats::glm, c(
          list(formula = .y ~ ., family = stats::binomial(), data = d),
          args
        ))
      },
      link = function(object, x) {
        as.numeric(stats::predict(object,
          newdata = as.data.frame(x, check.names = FALSE), type = "link"
        ))
      },
      coef = function(object) stats::coef(object),
      diagnostics = function(object) {
        cf <- stats::coef(summary(object))
        list(
          converged = isTRUE(object$converged),
          se = cf[, 2],
          z = cf[, 3],
          p_value = cf[, 4]
        )
      }
    ),

    # The package's own C++ L-BFGS logistic regression: no extra dependency,
    # and it returns standard errors directly.
    obwoe = list(
      pkgs = character(0),
      additive = TRUE,
      fit = function(x, y, args) {
        xm <- cbind(`(Intercept)` = 1, as.matrix(x))
        f <- do.call(fit_logistic_regression, c(
          list(X_r = xm, y_r = as.numeric(y)), args
        ))
        f$.names <- colnames(xm)
        f
      },
      link = function(object, x) {
        as.numeric(cbind(1, as.matrix(x)) %*% object$coefficients)
      },
      coef = function(object) stats::setNames(object$coefficients, object$.names),
      diagnostics = function(object) list(
        converged = isTRUE(object$convergence),
        se = stats::setNames(object$se, object$.names),
        z = stats::setNames(object$z_scores, object$.names),
        p_value = stats::setNames(object$p_values, object$.names)
      )
    ),

    # Penalised logistic regression: still additive in the WoE, so it still
    # produces a points table. Useful when the shortlist stays correlated.
    glmnet = list(
      pkgs = "glmnet",
      additive = TRUE,
      fit = function(x, y, args) {
        a <- utils::modifyList(
          list(x = as.matrix(x), y = as.integer(y), family = "binomial", alpha = 0),
          args
        )
        f <- do.call(glmnet::cv.glmnet, a)
        f$.s <- if (!is.null(args$s)) args$s else "lambda.min"
        f
      },
      link = function(object, x) {
        as.numeric(stats::predict(object,
          newx = as.matrix(x), s = object$.s, type = "link"
        ))
      },
      coef = function(object) {
        cf <- as.matrix(stats::coef(object, s = object$.s))
        stats::setNames(as.numeric(cf), rownames(cf))
      },
      diagnostics = function(object) list(converged = TRUE)
    )
  )
}


#' @title Internal: Resolve an Engine
#'
#' @param engine Character string naming a registered engine, or a list
#'   supplying \code{fit}, \code{link} and \code{coef} directly.
#' @param fallback Logical. Fall back to \code{"glm"} when the engine's package
#'   is not installed?
#'
#' @return An engine definition, with \code{name}, \code{requested} and
#'   \code{used} recorded.
#'
#' @details
#' A missing engine package is an error, not a silent substitution. A scorecard
#' is a governance artefact: a workbook documenting a model the analyst did not
#' ask for is worse than a call that fails. \code{fallback = TRUE} opts into the
#' substitution and records both the requested and the used engine so the
#' workbook can say which one produced it.
#'
#' @keywords internal
.ob_engine_get <- function(engine = "glm", fallback = FALSE) {
  if (is.list(engine)) {
    missing_fns <- setdiff(c("fit", "link", "coef"), names(engine))
    if (length(missing_fns) > 0L) {
      stop(sprintf(
        "A custom engine must supply %s; missing: %s.",
        "'fit', 'link' and 'coef'", paste(sQuote(missing_fns), collapse = ", ")
      ))
    }
    out <- utils::modifyList(
      list(pkgs = character(0), additive = NA, diagnostics = function(object) list()),
      engine
    )
    out$name <- "custom"
    out$requested <- "custom"
    out$used <- "custom"
    return(out)
  }

  reg <- .ob_engine_registry()
  if (!is.character(engine) || length(engine) != 1L || !engine %in% names(reg)) {
    stop(sprintf(
      "'engine' must be one of %s, or a list supplying fit/link/coef.",
      paste(sQuote(names(reg)), collapse = ", ")
    ))
  }

  e <- reg[[engine]]
  missing_pkgs <- e$pkgs[!vapply(e$pkgs, requireNamespace, logical(1),
    quietly = TRUE
  )]

  if (length(missing_pkgs) > 0L) {
    if (!isTRUE(fallback)) {
      stop(sprintf(
        paste0(
          "engine = \"%s\" needs the %s package, which is not installed. ",
          "Install it, use engine = \"glm\", or set engine_fallback = TRUE ",
          "in control.obwoe_scorecard()."
        ),
        engine, paste(sQuote(missing_pkgs), collapse = ", ")
      ))
    }
    warning(sprintf(
      "engine = \"%s\" needs %s, which is not installed; falling back to \"glm\".",
      engine, paste(sQuote(missing_pkgs), collapse = ", ")
    ))
    out <- .ob_engine_get("glm")
    out$requested <- engine
    return(out)
  }

  e$name <- engine
  e$requested <- engine
  e$used <- engine
  e
}
