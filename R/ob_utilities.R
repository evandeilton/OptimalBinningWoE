# ============================================================================#
# R Wrappers for C++ High-Performance Gains Table Engines
# ============================================================================#

#' @title Compute Comprehensive Gains Table from Binning Results
#'
#' @description
#' This function serves as a high-performance engine (implemented in C++) to calculate
#' a comprehensive set of credit scoring and classification metrics based on
#' pre-aggregated binning results. It takes a list of bin counts and computes
#' metrics such as Information Value (IV), Weight of Evidence (WoE), Kolmogorov-Smirnov (KS),
#' Gini, Lift, and various entropy-based divergence measures.
#'
#' @param binning_result A named \code{list} or \code{data.frame} containing the
#'   following atomic vectors (all must have the same length):
#'   \describe{
#'     \item{\code{id}}{Numeric vector of bin identifiers. Determines the sort order
#'       for cumulative metrics (e.g., KS, Recall).}
#'     \item{\code{bin}}{Character vector of bin labels/intervals.}
#'     \item{\code{count}}{Numeric vector of total observations per bin (\eqn{O_i}).}
#'     \item{\code{count_pos}}{Numeric vector of positive (event) counts per bin (\eqn{E_i}).}
#'     \item{\code{count_neg}}{Numeric vector of negative (non-event) counts per bin (\eqn{NE_i}).}
#'   }
#'
#' @return A \code{data.frame} with the following columns (metrics calculated per bin):
#'   \describe{
#'     \item{\strong{Identifiers}}{
#'       \code{id}, \code{bin}
#'     }
#'     \item{\strong{Counts & Rates}}{
#'       \code{count}, \code{pos}, \code{neg},
#'       \code{pos_rate} (\eqn{\pi_i}), \code{neg_rate} (\eqn{1-\pi_i}),
#'       \code{count_perc} (\eqn{O_i / O_{total}})
#'     }
#'     \item{\strong{Distributions (Shares)}}{
#'       \code{pos_perc} (\eqn{D_1(i)}: Share of Bad),
#'       \code{neg_perc} (\eqn{D_0(i)}: Share of Good)
#'     }
#'     \item{\strong{Cumulative Statistics}}{
#'       \code{cum_pos}, \code{cum_neg},
#'       \code{cum_pos_perc} (\eqn{CDF_1}), \code{cum_neg_perc} (\eqn{CDF_0}),
#'       \code{cum_count_perc}
#'     }
#'     \item{\strong{Credit Scoring Metrics}}{
#'       \code{woe}, \code{iv}, \code{total_iv}, \code{ks}, \code{lift},
#'       \code{odds_pos}, \code{odds_ratio}
#'     }
#'     \item{\strong{Advanced Metrics}}{
#'       \code{gini_contribution}, \code{log_likelihood},
#'       \code{kl_divergence}, \code{js_divergence}
#'     }
#'     \item{\strong{Classification Metrics}}{
#'       \code{precision}, \code{recall}, \code{f1_score}
#'     }
#'   }
#'
#' @details
#' \subsection{Mathematical Definitions}{
#'
#' Let \eqn{E_i} and \eqn{NE_i} be the number of events and non-events in bin \eqn{i},
#' and \eqn{E_{total}}, \eqn{NE_{total}} be the population totals.
#'
#' \strong{Weight of Evidence (WoE) & Information Value (IV):}
#' \deqn{WoE_i = \ln\left(\frac{E_i / E_{total}}{NE_i / NE_{total}}\right)}
#' \deqn{IV_i = \left(\frac{E_i}{E_{total}} - \frac{NE_i}{NE_{total}}\right) \times WoE_i}
#'
#' \strong{Kolmogorov-Smirnov (KS):}
#' \deqn{KS_i = \left| \sum_{j=1}^i \frac{E_j}{E_{total}} - \sum_{j=1}^i \frac{NE_j}{NE_{total}} \right|}
#'
#' \strong{Lift:}
#' \deqn{Lift_i = \frac{E_i / (E_i + NE_i)}{E_{total} / (E_{total} + NE_{total})}}
#'
#' \strong{Kullback-Leibler Divergence (Bernoulli):}
#' Measures the divergence between the bin's event rate \eqn{p_i} and the global event rate \eqn{p_{global}}:
#' \deqn{KL_i = p_i \ln\left(\frac{p_i}{p_{global}}\right) + (1-p_i) \ln\left(\frac{1-p_i}{1-p_{global}}\right)}
#' }
#'
#' @examples
#' # Manually constructed binning result
#' bin_res <- list(
#'   id = 1:3,
#'   bin = c("Low", "Medium", "High"),
#'   count = c(100, 200, 50),
#'   count_pos = c(5, 30, 20),
#'   count_neg = c(95, 170, 30)
#' )
#'
#' gt <- ob_gains_table(bin_res)
#' print(gt[, c("bin", "woe", "iv", "ks")])
#'
#' @export
ob_gains_table <- function(binning_result) {
  .Call("_OptimalBinningWoE_OBGainsTable", binning_result, PACKAGE = "OptimalBinningWoE")
}

#' @title Compute Gains Table for a Binned Feature Vector
#'
#' @description
#' Calculates a full gains table by aggregating a raw binned dataframe against a
#' binary target. Unlike \code{\link{ob_gains_table}} which expects pre-aggregated counts,
#' this function takes observation-level data, aggregates it by the specified
#' group variable (bin, WoE, or ID), and then computes all statistical metrics.
#'
#' @param binned_df A \code{data.frame} resulting from a binning transformation (e.g., via
#'   \code{obwoe_apply}), containing at least the following columns:
#'   \describe{
#'     \item{\code{feature}}{Original feature values (optional, for reference).}
#'     \item{\code{bin}}{Character vector of bin labels.}
#'     \item{\code{woe}}{Numeric vector of Weight of Evidence values.}
#'     \item{\code{idbin}}{Numeric vector of bin IDs (required for correct sorting).}
#'   }
#' @param target A numeric vector of binary outcomes (0 for non-event, 1 for event).
#'   Must have the same length as \code{binned_df}. Missing values are not allowed.
#' @param group_var Character string specifying the aggregation key. Options:
#'   \itemize{
#'     \item \code{"bin"}: Group by bin label (default).
#'     \item \code{"woe"}: Group by WoE value.
#'     \item \code{"idbin"}: Group by bin ID.
#'   }
#'
#' @return A \code{data.frame} containing the same extensive set of metrics as
#'   \code{\link{ob_gains_table}}, aggregated by \code{group_var} and sorted by \code{idbin}.
#'
#' @details
#' \subsection{Aggregation and Sorting}{
#' The function first aggregates the binary target by the specified \code{group_var}.
#' Crucially, it uses the \code{idbin} column to sort the resulting groups. This ensures
#' that cumulative metrics (like KS and Gini) are calculated based on the logical
#' order of the bins (e.g., low score to high score), not alphabetical order.
#' }
#'
#' \subsection{Advanced Metrics}{
#' In addition to standard credit scoring metrics, this function computes:
#' \itemize{
#'   \item \strong{Jensen-Shannon Divergence}: A symmetrized and smoothed version of
#'     KL divergence, useful for measuring stability between the bin distribution
#'     and the population distribution.
#'   \item \strong{F1-Score, Precision, Recall}: Treating each bin as a potential
#'     classification threshold.
#' }
#' }
#'
#' @references
#' Siddiqi, N. (2006). \emph{Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring}. Wiley.
#'
#' Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency.
#' \emph{The Annals of Mathematical Statistics}.
#'
#' @examples
#' \donttest{
#' # Mock data representing a binned feature
#' df_binned <- data.frame(
#'   feature = c(10, 20, 30, 10, 20, 50),
#'   bin = c("Low", "Mid", "High", "Low", "Mid", "High"),
#'   woe = c(-0.5, 0.2, 1.1, -0.5, 0.2, 1.1),
#'   idbin = c(1, 2, 3, 1, 2, 3)
#' )
#' target <- c(0, 0, 1, 1, 0, 1)
#'
#' # Calculate gains table grouped by bin ID
#' gt <- ob_gains_table_feature(df_binned, target, group_var = "idbin")
#'
#' # Inspect key metrics
#' print(gt[, c("id", "count", "pos_rate", "lift", "js_divergence")])
#' }
#'
#' @export
ob_gains_table_feature <- function(binned_df, target, group_var = "bin") {
  .Call("_OptimalBinningWoE_OBGainsTableFeature", binned_df, as.numeric(target), group_var, PACKAGE = "OptimalBinningWoE")
}
