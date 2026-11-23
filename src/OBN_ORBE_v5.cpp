// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * @file OBN_ORBE_v5.cpp
 * @brief Order-Restricted Bayesian Evidence binning for numerical features
 *
 * Atoms are the sorted distinct values of the feature. The bins and their
 * number maximize the Beta-Binomial evidence of the partition under a shape
 * constraint on the posterior WoE.
 */

#include "common/orbe_core.h"

using namespace OptimalBinning;

namespace {

// Interval label in the format shared by the numerical engines.
std::string interval_label(double lower, double upper, bool first, bool last) {
  std::ostringstream oss;
  oss.precision(6);
  oss << std::fixed;
  if (first && last) oss << "(-Inf;+Inf]";
  else if (first) oss << "(-Inf;" << upper << "]";
  else if (last) oss << "(" << lower << ";+Inf]";
  else oss << "(" << lower << ";" << upper << "]";
  return oss.str();
}

orbe::Shape parse_trend(const std::string& trend, std::vector<orbe::Shape>& shapes) {
  if (trend == "auto") shapes = {orbe::Shape::Ascending, orbe::Shape::Descending};
  else if (trend == "ascending") shapes = {orbe::Shape::Ascending};
  else if (trend == "descending") shapes = {orbe::Shape::Descending};
  else if (trend == "peak") shapes = {orbe::Shape::Peak};
  else if (trend == "valley") shapes = {orbe::Shape::Valley};
  else if (trend == "none") shapes = {orbe::Shape::None};
  else throw std::invalid_argument(
    "monotonic_trend must be one of: 'auto', 'ascending', 'descending', 'peak', 'valley', 'none'.");
  return shapes.front();
}

} // namespace

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_orbe(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 2,
    int max_bins = 10,
    double bin_cutoff = 0.05,
    int max_n_prebins = 200,
    std::string monotonic_trend = "auto",
    double prior_strength = 1.0,
    double credible_level = 0.95) {

  try {
    // ---- Validation -------------------------------------------------------
    const R_xlen_t len = feature.size();
    if (len != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (len == 0) throw std::invalid_argument("Feature and target must not be empty.");
    if (min_bins < 1) throw std::invalid_argument("min_bins must be at least 1.");
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (!(bin_cutoff >= 0.0 && bin_cutoff < 1.0)) {
      throw std::invalid_argument("bin_cutoff must be in [0, 1).");
    }
    if (max_n_prebins < 2) throw std::invalid_argument("max_n_prebins must be at least 2.");
    if (!(prior_strength > 0.0) || !std::isfinite(prior_strength)) {
      throw std::invalid_argument("prior_strength must be a positive finite number.");
    }
    if (!(credible_level > 0.0 && credible_level < 1.0)) {
      throw std::invalid_argument("credible_level must be in (0, 1).");
    }
    std::vector<orbe::Shape> shapes;
    const orbe::Shape first_shape = parse_trend(monotonic_trend, shapes);

    // ---- Rows with a non-missing feature (NA / NaN rows are excluded) -----
    std::vector<std::pair<double, int>> rows;
    rows.reserve(static_cast<size_t>(len));
    double tot_e = 0.0;
    for (R_xlen_t i = 0; i < len; ++i) {
      const int t = target[i];
      if (t == NA_INTEGER) throw std::invalid_argument("Target contains missing values (NA).");
      if (t != 0 && t != 1) throw std::invalid_argument("Target must be binary (0 or 1).");
      const double x = feature[i];
      if (std::isnan(x)) continue;
      rows.emplace_back(x + 0.0, t);  // -0.0 == +0.0
      tot_e += t;
    }
    if (rows.empty()) throw std::invalid_argument("Feature has no non-missing values.");
    const double tot_n = static_cast<double>(rows.size());
    if (tot_e == 0.0 || tot_e == tot_n) {
      throw std::invalid_argument(
        "Target must contain both classes (0 and 1) among the non-missing feature values.");
    }
    std::sort(rows.begin(), rows.end());

    // ---- Distinct values --------------------------------------------------
    std::vector<double> d_val, d_n, d_e;
    for (size_t i = 0; i < rows.size();) {
      size_t j = i;
      double e = 0.0;
      while (j < rows.size() && rows[j].first == rows[i].first) { e += rows[j].second; ++j; }
      d_val.push_back(rows[i].first);
      d_n.push_back(static_cast<double>(j - i));
      d_e.push_back(e);
      i = j;
    }
    // -Inf cannot be a cutpoint: it joins the smallest finite value
    if (d_val.size() > 1 && std::isinf(d_val[0]) && d_val[0] < 0) {
      d_n[1] += d_n[0];
      d_e[1] += d_e[0];
      d_val.erase(d_val.begin());
      d_n.erase(d_n.begin());
      d_e.erase(d_e.begin());
    }
    const size_t n_distinct = d_val.size();

    // ---- Atoms: distinct values, grouped by frequency above the cap -------
    const bool unimodal = first_shape == orbe::Shape::Peak || first_shape == orbe::Shape::Valley;
    const size_t cap = static_cast<size_t>(orbe::atom_cap(max_n_prebins, max_bins, unimodal));
    std::vector<double> upper, cum_n(1, 0.0), cum_e(1, 0.0);
    if (n_distinct <= cap) {
      for (size_t i = 0; i < n_distinct; ++i) {
        upper.push_back(d_val[i]);
        cum_n.push_back(cum_n.back() + d_n[i]);
        cum_e.push_back(cum_e.back() + d_e[i]);
      }
    } else {
      const double step = tot_n / static_cast<double>(cap);
      double next = step, run_n = 0.0, run_e = 0.0;
      for (size_t i = 0; i < n_distinct; ++i) {
        run_n += d_n[i];
        run_e += d_e[i];
        if (run_n >= next - 1e-9 || i + 1 == n_distinct) {
          upper.push_back(d_val[i]);
          cum_n.push_back(run_n);
          cum_e.push_back(run_e);
          while (next <= run_n + 1e-9) next += step;
        }
      }
    }
    const int u = static_cast<int>(upper.size());

    // ---- Prior ------------------------------------------------------------
    double a, b;
    orbe::prior_parameters(tot_e / tot_n, prior_strength, a, b);
    // Number of bins uniform on 1..m+1, cuts uniform among the m boundaries
    // between distinct values.
    const double m = static_cast<double>(n_distinct) - 1.0;
    const int kmax = std::min(max_bins, u);
    std::vector<double> log_prior(static_cast<size_t>(kmax) + 1, 0.0);
    for (int k = 1; k <= kmax; ++k) {
      log_prior[static_cast<size_t>(k)] = -std::log(m + 1.0) - R::lchoose(m, k - 1.0);
    }

    // ---- Fit --------------------------------------------------------------
    const double n_min = orbe::min_bin_rows(bin_cutoff, tot_n);
    const orbe::Fit fit = orbe::fit(cum_n, cum_e, min_bins, max_bins, n_min, a, b,
                                    log_prior, shapes, true);
    const size_t k = fit.ends.size();

    // ---- Output -----------------------------------------------------------
    std::vector<double> bin_n(k), bin_e(k);
    Rcpp::CharacterVector labels(k);
    Rcpp::NumericVector cutpoints(k > 0 ? k - 1 : 0);
    int lo = 0;
    double lower = R_NegInf;
    for (size_t i = 0; i < k; ++i) {
      const int hi = fit.ends[i];
      bin_n[i] = cum_n[static_cast<size_t>(hi)] - cum_n[static_cast<size_t>(lo)];
      bin_e[i] = cum_e[static_cast<size_t>(hi)] - cum_e[static_cast<size_t>(lo)];
      const double up = upper[static_cast<size_t>(hi) - 1];
      labels[i] = interval_label(lower, up, i == 0, i + 1 == k);
      if (i + 1 < k) cutpoints[i] = up;
      lower = up;
      lo = hi;
    }
    const orbe::BinSummary s = orbe::summarize(bin_n, bin_e, a, b, credible_level);

    Rcpp::NumericVector ids(k);
    Rcpp::IntegerVector count(k), count_pos(k), count_neg(k);
    for (size_t i = 0; i < k; ++i) {
      ids[i] = static_cast<double>(i + 1);
      count[i] = static_cast<int>(bin_n[i]);
      count_pos[i] = static_cast<int>(bin_e[i]);
      count_neg[i] = static_cast<int>(bin_n[i] - bin_e[i]);
    }

    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = labels,
      Rcpp::Named("woe") = Rcpp::wrap(s.woe),
      Rcpp::Named("iv") = Rcpp::wrap(s.iv),
      Rcpp::Named("count") = count,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg,
      Rcpp::Named("event_rate") = Rcpp::wrap(s.event_rate),
      Rcpp::Named("woe_se") = Rcpp::wrap(s.woe_se),
      Rcpp::Named("woe_lower") = Rcpp::wrap(s.woe_lower),
      Rcpp::Named("woe_upper") = Rcpp::wrap(s.woe_upper),
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("total_iv") = s.total_iv,
      Rcpp::Named("total_iv_corrected") = s.total_iv_corrected,
      Rcpp::Named("log_evidence") = fit.log_evidence,
      Rcpp::Named("log_evidence_by_bins") = Rcpp::wrap(fit.score_by_k),
      Rcpp::Named("monotonic_trend") = orbe::shape_to_string(fit.shape),
      Rcpp::Named("order_probability") = fit.order_probability,
      Rcpp::Named("converged") = true,
      Rcpp::Named("iterations") = kmax
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}
