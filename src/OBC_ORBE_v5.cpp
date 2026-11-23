// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file OBC_ORBE_v5.cpp
 * @brief Order-Restricted Bayesian Evidence binning for categorical features
 *
 * Atoms are the categories ordered by their posterior-mean event rate. The
 * groups are contiguous in that order and maximize the Beta-Binomial evidence
 * of the grouping, so a category merges when the data do not support a rate
 * of its own.
 */

#include "common/orbe_core.h"

using namespace OptimalBinning;

namespace {

struct Level {
  std::string name;
  double n = 0.0;
  double e = 0.0;
  double rate = 0.0;  // posterior-mean event rate
};

} // namespace

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_orbe(
    Rcpp::IntegerVector target,
    Rcpp::StringVector feature,
    int min_bins = 2,
    int max_bins = 10,
    double bin_cutoff = 0.05,
    int max_n_prebins = 200,
    std::string bin_separator = "%;%",
    double prior_strength = 1.0,
    double credible_level = 0.95) {

  try {
    // ---- Validation -------------------------------------------------------
    const R_xlen_t len = feature.size();
    if (len != target.size()) {
      throw std::invalid_argument("feature and target must have the same size.");
    }
    if (len == 0) throw std::invalid_argument("Input vectors cannot be empty.");
    if (min_bins < 1) throw std::invalid_argument("min_bins must be >= 1.");
    if (max_bins < min_bins) throw std::invalid_argument("max_bins must be >= min_bins.");
    if (!(bin_cutoff >= 0.0 && bin_cutoff < 1.0)) {
      throw std::invalid_argument("bin_cutoff must be in [0, 1).");
    }
    if (max_n_prebins < 2) throw std::invalid_argument("max_n_prebins must be >= 2.");
    if (!(prior_strength > 0.0) || !std::isfinite(prior_strength)) {
      throw std::invalid_argument("prior_strength must be a positive finite number.");
    }
    if (!(credible_level > 0.0 && credible_level < 1.0)) {
      throw std::invalid_argument("credible_level must be in (0, 1).");
    }

    // ---- Counts per category (a missing value is the category "NA") -------
    std::map<std::string, Level> table;
    double tot_e = 0.0;
    for (R_xlen_t i = 0; i < len; ++i) {
      const int t = target[i];
      if (t == NA_INTEGER) throw std::invalid_argument("Target cannot contain missing values.");
      if (t != 0 && t != 1) throw std::invalid_argument("Target must contain only values 0 and 1.");
      const std::string key =
        (feature[i] == NA_STRING) ? std::string("NA") : Rcpp::as<std::string>(feature[i]);
      Level& lv = table[key];
      lv.name = key;
      lv.n += 1.0;
      lv.e += t;
      tot_e += t;
    }
    const double tot_n = static_cast<double>(len);
    if (tot_e == 0.0 || tot_e == tot_n) {
      throw std::invalid_argument("Target must contain both values 0 and 1.");
    }

    // ---- Order by posterior-mean rate; ties by count, then by name --------
    double a, b;
    orbe::prior_parameters(tot_e / tot_n, prior_strength, a, b);
    std::vector<Level> levels;
    levels.reserve(table.size());
    for (auto& kv : table) {
      kv.second.rate = (a + kv.second.e) / (a + b + kv.second.n);
      levels.push_back(kv.second);
    }
    std::sort(levels.begin(), levels.end(), [](const Level& x, const Level& y) {
      if (x.rate != y.rate) return x.rate < y.rate;
      if (x.n != y.n) return x.n < y.n;
      return x.name < y.name;
    });
    const size_t n_levels = levels.size();

    // A separator inside a category name makes the bin labels ambiguous
    size_t hits = 0;
    std::string example;
    for (const Level& lv : levels) {
      if (!bin_separator.empty() && lv.name.find(bin_separator) != std::string::npos) {
        if (hits == 0 || lv.name < example) example = lv.name;
        ++hits;
      }
    }
    if (hits > 0) {
      Rcpp::warning("bin_separator \"%s\" occurs inside %d categor%s of the feature "
                    "(e.g. \"%s\"): the bin labels are ambiguous and cannot be split "
                    "back into categories. Choose a bin_separator that does not occur "
                    "in the category names.",
                    bin_separator, static_cast<int>(hits), (hits == 1 ? "y" : "ies"), example);
    }

    // ---- Atoms: categories, grouped by frequency above the cap ------------
    const size_t cap = static_cast<size_t>(orbe::atom_cap(max_n_prebins, max_bins, false));
    std::vector<size_t> atom_end;  // one past the last level of each atom
    std::vector<double> cum_n(1, 0.0), cum_e(1, 0.0);
    if (n_levels <= cap) {
      for (size_t i = 0; i < n_levels; ++i) {
        atom_end.push_back(i + 1);
        cum_n.push_back(cum_n.back() + levels[i].n);
        cum_e.push_back(cum_e.back() + levels[i].e);
      }
    } else {
      const double step = tot_n / static_cast<double>(cap);
      double next = step, run_n = 0.0, run_e = 0.0;
      for (size_t i = 0; i < n_levels; ++i) {
        run_n += levels[i].n;
        run_e += levels[i].e;
        if (run_n >= next - 1e-9 || i + 1 == n_levels) {
          atom_end.push_back(i + 1);
          cum_n.push_back(run_n);
          cum_e.push_back(run_e);
          while (next <= run_n + 1e-9) next += step;
        }
      }
    }
    const int u = static_cast<int>(atom_end.size());

    // ---- Prior: number of groups uniform on 1..L, and each category is
    // assigned to one of the k groups uniformly
    const int kmax = std::min(max_bins, u);
    const double n_lev = static_cast<double>(n_levels);
    std::vector<double> log_prior(static_cast<size_t>(kmax) + 1, 0.0);
    for (int k = 1; k <= kmax; ++k) {
      log_prior[static_cast<size_t>(k)] = -std::log(n_lev) - n_lev * std::log(static_cast<double>(k));
    }

    // ---- Fit: groups in strictly increasing posterior WoE -----------------
    const double n_min = orbe::min_bin_rows(bin_cutoff, tot_n);
    const std::vector<orbe::Shape> shapes = {orbe::Shape::Ascending};
    const orbe::Fit fit = orbe::fit(cum_n, cum_e, min_bins, max_bins, n_min, a, b,
                                    log_prior, shapes, false);
    const size_t k = fit.ends.size();

    // ---- Output -----------------------------------------------------------
    std::vector<double> bin_n(k), bin_e(k);
    Rcpp::CharacterVector labels(k);
    int lo = 0;
    for (size_t i = 0; i < k; ++i) {
      const int hi = fit.ends[i];
      bin_n[i] = cum_n[static_cast<size_t>(hi)] - cum_n[static_cast<size_t>(lo)];
      bin_e[i] = cum_e[static_cast<size_t>(hi)] - cum_e[static_cast<size_t>(lo)];
      const size_t from = lo == 0 ? 0 : atom_end[static_cast<size_t>(lo) - 1];
      const size_t to = atom_end[static_cast<size_t>(hi) - 1];
      std::string label;
      for (size_t l = from; l < to; ++l) {
        if (l > from) label += bin_separator;
        label += levels[l].name;
      }
      labels[i] = label;
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
      Rcpp::Named("total_iv") = s.total_iv,
      Rcpp::Named("total_iv_corrected") = s.total_iv_corrected,
      Rcpp::Named("log_evidence") = fit.log_evidence,
      Rcpp::Named("log_evidence_by_bins") = Rcpp::wrap(fit.score_by_k),
      Rcpp::Named("order_probability") = fit.order_probability,
      Rcpp::Named("converged") = true,
      Rcpp::Named("iterations") = kmax
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}
