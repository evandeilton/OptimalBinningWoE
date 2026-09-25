// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <set>
#include <string>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

/**
 * @brief A finite cut c with a <= c < b between two consecutive distinct
 * values a < b, for the right-closed (lower, upper] convention.
 *
 * The midpoint (a + b) / 2 is used whenever it is valid, so ordinary data is
 * unaffected; it overflows to +/-Inf near the largest finite double and, for
 * two adjacent doubles, can round up to b. Then a/2 + b/2, and finally a, are
 * used instead.
 */
inline double ubsd_safe_cut(double a, double b) {
  double m = (a + b) / 2.0;
  if (!std::isfinite(m)) m = a / 2.0 + b / 2.0;
  if (std::isfinite(m) && m >= a && m < b) return m;
  if (std::isfinite(a)) return a;
  // a == -Inf: the only finite value that keeps -Inf alone is the lowest double
  return std::numeric_limits<double>::lowest();
}

} // namespace


/**
 * @brief Optimal Binning for Numerical Variables using Unsupervised Binning with Standard Deviation (UBSD)
 *
 * Algorithm Overview:
 * 1. Initial edges from mean +/- k * sd and equal-width points (unsupervised)
 * 2. Assignment to right-closed bins (lower, upper]
 * 3. Greedy merging of small (and empty) bins, direction by IV
 * 4. Iterated monotonicity enforcement and reduction to max_bins until the
 *    total IV is stable
 * 5. Laplace smoothing for stable WoE in sparse bins
 */
class OBN_UBSD {
private:
  // Input data
  std::vector<double> feature;
  std::vector<double> target;

  // Algorithm parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;

  // Results
  std::vector<NumericalBin> bins;
  std::vector<double> cutpoints;
  double total_iv;
  bool converged;
  int iterations_run;

public:
  OBN_UBSD(
    const std::vector<double>& feat,
    const std::vector<double>& targ,
    int min_b = 3,
    int max_b = 5,
    double cutoff = 0.05,
    int max_prebins = 20,
    double conv_threshold = 1e-6,
    int max_iter = 1000,
    double laplace_smooth = 0.5
  ) : feature(feat), target(targ),
  min_bins(std::max(min_b, 2)),
  max_bins(std::max(max_b, min_bins)),
  bin_cutoff(cutoff),
  max_n_prebins(std::max(max_prebins, min_bins)),
  convergence_threshold(conv_threshold),
  max_iterations(max_iter),
  laplace_smoothing(laplace_smooth),
  total_iv(0.0),
  converged(false),
  iterations_run(0) {
    validate_inputs();
  }

  void fit() {
    drop_missing_values();

    std::vector<double> unique_values = get_unique_values();
    if (unique_values.size() <= 2) {
      handle_low_unique_values(unique_values);
      converged = true;
      iterations_run = 0;
      return;
    }

    create_initial_bins();
    assign_observations_to_bins();
    merge_small_bins();
    calculate_woe_iv();

    double prev_iv = get_total_iv();
    for (int iter = 0; iter < max_iterations; ++iter) {
      enforce_monotonicity();
      adjust_bin_count();
      calculate_woe_iv();

      const double current_iv = get_total_iv();
      iterations_run = iter + 1;
      if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      prev_iv = current_iv;
    }

    update_cutpoints();
  }

  Rcpp::List create_output() const {
    const size_t nb = bins.size();
    std::vector<std::string> bin_names;
    std::vector<double> woe_vals, iv_vals, event_rates;
    std::vector<int> c_vals, cpos_vals, cneg_vals;
    bin_names.reserve(nb);
    woe_vals.reserve(nb);
    iv_vals.reserve(nb);
    event_rates.reserve(nb);
    c_vals.reserve(nb);
    cpos_vals.reserve(nb);
    cneg_vals.reserve(nb);

    for (const auto& b : bins) {
      // Right-closed (lower; upper], as observations are assigned. These
      // labels used to read "[lower;upper)".
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(6);
      if (std::isinf(b.lower_bound)) {
        oss << "(-Inf;";
      } else {
        oss << "(" << b.lower_bound << ";";
      }
      if (std::isinf(b.upper_bound)) {
        oss << "+Inf]";
      } else {
        oss << b.upper_bound << "]";
      }
      bin_names.push_back(oss.str());
      woe_vals.push_back(b.woe);
      iv_vals.push_back(b.iv);
      c_vals.push_back(b.count);
      cpos_vals.push_back(b.count_pos);
      cneg_vals.push_back(b.count_neg);
      event_rates.push_back(b.event_rate());
    }

    Rcpp::NumericVector ids(static_cast<R_xlen_t>(nb));
    for (R_xlen_t i = 0; i < ids.size(); i++) {
      ids[i] = static_cast<double>(i + 1);
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woe_vals,
      Named("iv") = iv_vals,
      Named("count") = c_vals,
      Named("count_pos") = cpos_vals,
      Named("count_neg") = cneg_vals,
      Named("event_rate") = event_rates,
      Named("cutpoints") = cutpoints,
      Named("total_iv") = total_iv,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }

private:
  double get_total_iv() const {
    double sum = 0.0;
    for (const auto &b : bins) {
      sum += b.iv;
    }
    return sum;
  }

  // The numeric parameters and the length match are validated by the
  // exported wrapper.
  void validate_inputs() const {
    if (feature.empty()) {
      Rcpp::stop("Feature and target vectors cannot be empty.");
    }

    bool has_zero = false, has_one = false;
    for (double t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else Rcpp::stop("Target must contain only 0 and 1.");
      if (has_zero && has_one) break;
    }
    if (!has_zero || !has_one) {
      Rcpp::stop("Target must contain both classes (0 and 1).");
    }
  }

  /**
   * @brief Drop observations with a missing feature (NaN / NA) silently.
   * +/-Inf are kept as extreme values: they fall in the first / last bin and
   * are excluded from the mean, sd, min and max that place the edges.
   */
  void drop_missing_values() {
    size_t k = 0;
    for (size_t i = 0; i < feature.size(); i++) {
      if (!std::isnan(feature[i])) {
        feature[k] = feature[i];
        target[k] = target[i];
        k++;
      }
    }
    feature.resize(k);
    target.resize(k);
    if (feature.empty()) {
      Rcpp::stop("All feature values are missing (NA/NaN); nothing to bin.");
    }
  }

  /**
   * @brief Up to three distinct values, sorted: only "one", "two" or "more
   * than two" distinct values matter here, so one O(n) scan that stops at the
   * third distinct value replaces a full sort (or std::set) of the feature.
   */
  std::vector<double> get_unique_values() const {
    std::vector<double> u;
    for (double v : feature) {
      if (std::find(u.begin(), u.end(), v) == u.end()) {
        u.push_back(v);
        if (u.size() > 2) break;
      }
    }
    std::sort(u.begin(), u.end());
    return u;
  }

  /**
   * @brief One bin (one distinct value) or two bins split between the two
   * values. The cutpoint of the two-bin case used to be left out of the
   * returned `cutpoints`, which then described a single bin.
   */
  void handle_low_unique_values(const std::vector<double>& unique_vals) {
    bins.clear();
    cutpoints.clear();

    if (unique_vals.size() == 1) {
      bins.emplace_back(-std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::infinity());
      for (size_t i = 0; i < feature.size(); i++) {
        bins[0].count++;
        if (target[i] == 1) bins[0].count_pos++; else bins[0].count_neg++;
      }
    } else {
      const double cut = ubsd_safe_cut(unique_vals[0], unique_vals[1]);
      bins.emplace_back(-std::numeric_limits<double>::infinity(), cut);
      bins.emplace_back(cut, std::numeric_limits<double>::infinity());
      for (size_t i = 0; i < feature.size(); i++) {
        NumericalBin& b = bins[feature[i] <= cut ? 0 : 1];
        b.count++;
        if (target[i] == 1) b.count_pos++; else b.count_neg++;
      }
      cutpoints.push_back(cut);
    }
    calculate_woe_iv();
  }

  static double mean_of(const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
  }

  static double stddev_of(const std::vector<double>& v, double m) {
    double accum = 0.0;
    for (double val : v) {
      accum += (val - m) * (val - m);
    }
    return std::sqrt(accum / static_cast<double>(v.size() - 1));
  }

  /**
   * @brief Initial edges: mean +/- {0, 1, 2} sd together with equal-width
   * points, limited to max_n_prebins bins.
   *
   * Mean and sd are computed on the data rescaled by max|x| when the plain
   * sums overflow (features near the largest double); non-finite candidate
   * edges are discarded. Before, such features produced NaN edges, which break
   * the ordering of std::set (undefined behaviour) and yielded NaN cutpoints.
   */
  void create_initial_bins() {
    // Statistics of the finite values only (there is at least one: three or
    // more distinct values cannot all be +/-Inf).
    std::vector<double> finite;
    finite.reserve(feature.size());
    for (double v : feature) {
      if (std::isfinite(v)) finite.push_back(v);
    }
    const double min_val = *std::min_element(finite.begin(), finite.end());
    const double max_val = *std::max_element(finite.begin(), finite.end());

    double m = mean_of(finite);
    double sd = finite.size() > 1 ? stddev_of(finite, m) : 0.0;
    if (!std::isfinite(m) || !std::isfinite(sd)) {
      const double scale = std::max(std::fabs(min_val), std::fabs(max_val));
      std::vector<double> scaled(finite);
      for (double& v : scaled) v /= scale;
      const double ms = mean_of(scaled);
      m = ms * scale;
      sd = stddev_of(scaled, ms) * scale;
    }

    bins.clear();
    std::vector<double> edges;
    edges.push_back(-std::numeric_limits<double>::infinity());

    if (sd < EPSILON) {
      // Nearly constant feature (documented): equal-width bins between min and max
      if (max_val - min_val >= EPSILON) {
        const double step = (max_val - min_val) / min_bins;
        for (int i = 1; i < min_bins; i++) {
          edges.push_back(min_val + i * step);
        }
      }
      edges.push_back(std::numeric_limits<double>::infinity());
    } else {
      int n_pre = std::min(max_n_prebins, max_bins);
      n_pre = std::max(n_pre, min_bins);

      std::set<double> edge_set;
      auto add_edge = [&edge_set](double e) {
        if (std::isfinite(e)) edge_set.insert(e);
      };

      add_edge(m - 2.0 * sd);
      add_edge(m - 1.0 * sd);
      add_edge(m);
      add_edge(m + 1.0 * sd);
      add_edge(m + 2.0 * sd);

      const double range = max_val - min_val;
      if (std::isfinite(range)) {
        const double step = range / n_pre;
        for (int i = 1; i < n_pre; i++) {
          add_edge(min_val + i * step);
        }
      } else {
        const double step = max_val / n_pre - min_val / n_pre;
        for (int i = 1; i < n_pre; i++) {
          add_edge(min_val + i * step);
        }
      }

      edges.insert(edges.end(), edge_set.begin(), edge_set.end());
      edges.push_back(std::numeric_limits<double>::infinity());

      // Limit to max_n_prebins bins
      if (edges.size() > static_cast<size_t>(max_n_prebins + 1)) {
        std::vector<double> sampled_edges;
        sampled_edges.push_back(-std::numeric_limits<double>::infinity());
        const double sampling_step = static_cast<double>(edges.size() - 2) / (max_n_prebins - 1);
        for (int i = 1; i < max_n_prebins; i++) {
          const size_t idx = 1 + static_cast<size_t>(i * sampling_step);
          if (idx < edges.size() - 1) {
            sampled_edges.push_back(edges[idx]);
          }
        }
        sampled_edges.push_back(std::numeric_limits<double>::infinity());
        edges.swap(sampled_edges);
      }
    }

    for (size_t i = 0; i + 1 < edges.size(); i++) {
      bins.emplace_back(edges[i], edges[i + 1]);
    }
  }

  /**
   * @brief Count observations per bin. Bins are contiguous and right-closed,
   * so the bin of v is the first one whose upper bound is >= v (binary search
   * instead of a linear scan per observation).
   */
  void assign_observations_to_bins() {
    std::vector<double> uppers;
    uppers.reserve(bins.size());
    for (auto &b : bins) {
      b.count = 0;
      b.count_pos = 0;
      b.count_neg = 0;
      uppers.push_back(b.upper_bound);
    }
    for (size_t i = 0; i < feature.size(); i++) {
      const size_t idx = static_cast<size_t>(
        std::lower_bound(uppers.begin(), uppers.end(), feature[i]) - uppers.begin());
      NumericalBin& b = bins[idx];
      b.count++;
      if (target[i] == 1) b.count_pos++; else b.count_neg++;
    }
  }

  /**
   * @brief Merge bins whose share is below bin_cutoff.
   *
   * The direction (left or right neighbour) minimises IV_i + IV_neighbour as
   * documented; the IVs are now computed before and after every merge (they
   * used to be all zero here, so every rare bin was merged to the left).
   * Empty bins -- the sd-based edges can fall outside the data range -- are
   * merged even when that goes below min_bins, since min_bins cannot be met
   * with bins that hold no observation.
   */
  void merge_small_bins() {
    const double total_count = static_cast<double>(feature.size());
    calculate_woe_iv();

    while (bins.size() > 1) {
      size_t smallest_idx = 0;
      double smallest_prop = std::numeric_limits<double>::max();
      for (size_t i = 0; i < bins.size(); i++) {
        const double prop = static_cast<double>(bins[i].count) / total_count;
        if (prop < smallest_prop) {
          smallest_prop = prop;
          smallest_idx = i;
        }
      }

      if (!(smallest_prop < bin_cutoff)) break;
      if (static_cast<int>(bins.size()) <= min_bins && bins[smallest_idx].count > 0) break;

      if (smallest_idx == 0) {
        merge_bins(0);
      } else if (smallest_idx == bins.size() - 1) {
        merge_bins(bins.size() - 2);
      } else {
        const double iv_left = bins[smallest_idx - 1].iv + bins[smallest_idx].iv;
        const double iv_right = bins[smallest_idx].iv + bins[smallest_idx + 1].iv;
        merge_bins(iv_left <= iv_right ? smallest_idx - 1 : smallest_idx);
      }
      calculate_woe_iv();
    }
  }

  /**
   * @brief WoE and IV of every bin with Laplace smoothing.
   */
  void calculate_woe_iv() {
    double total_pos = 0.0;
    double total_neg = 0.0;
    for (const auto &b : bins) {
      total_pos += b.count_pos;
      total_neg += b.count_neg;
    }

    const double total_smoothed_pos = total_pos + static_cast<double>(bins.size()) * laplace_smoothing;
    const double total_smoothed_neg = total_neg + static_cast<double>(bins.size()) * laplace_smoothing;

    total_iv = 0.0;
    // Only one class left after dropping missing features: no evidence.
    if (total_pos == 0.0 || total_neg == 0.0) {
      for (auto &b : bins) {
        b.woe = 0.0;
        b.iv = 0.0;
      }
      return;
    }
    for (auto &b : bins) {
      const double p = (b.count_pos + laplace_smoothing) / total_smoothed_pos;
      const double q = (b.count_neg + laplace_smoothing) / total_smoothed_neg;

      if (p <= 0.0 && q <= 0.0) {
        b.woe = 0.0;    // empty bin without smoothing
      } else if (p <= 0.0) {
        b.woe = -20.0;  // Cap for stability
      } else if (q <= 0.0) {
        b.woe = 20.0;   // Cap for stability
      } else {
        b.woe = std::log(p / q);
      }
      b.iv = (p - q) * b.woe;
      total_iv += b.iv;
    }
  }

  void enforce_monotonicity() {
    if (bins.size() <= 2) return;

    const bool increasing = guess_trend();

    bool merged = true;
    while (merged && static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      merged = false;

      for (size_t i = 1; i < bins.size(); i++) {
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {

          double total_pos = 0.0, total_neg = 0.0;
          for (const auto &b : bins) {
            total_pos += b.count_pos;
            total_neg += b.count_neg;
          }

          const double k_after = static_cast<double>(bins.size() - 1);
          const double p = (bins[i - 1].count_pos + bins[i].count_pos + laplace_smoothing) /
                           (total_pos + k_after * laplace_smoothing);
          const double q = (bins[i - 1].count_neg + bins[i].count_neg + laplace_smoothing) /
                           (total_neg + k_after * laplace_smoothing);
          const double merged_woe = std::log(p / q);

          // Would merging i-1 and i create a new violation with bin i-2?
          bool new_violation = false;
          if (i > 1) {
            new_violation = (increasing && merged_woe < bins[i - 2].woe) ||
                            (!increasing && merged_woe > bins[i - 2].woe);
          }

          if (new_violation && i < bins.size() - 1) {
            merge_bins(i);        // merge i and i+1 instead
          } else {
            merge_bins(i - 1);    // merge i-1 and i
          }

          calculate_woe_iv();
          merged = true;
          break;
        }
      }

      iterations_run++;
    }
  }

  void adjust_bin_count() {
    while (static_cast<int>(bins.size()) > max_bins && iterations_run < max_iterations) {
      merge_bins(find_min_iv_merge());
      calculate_woe_iv();
      iterations_run++;
    }
  }

  /**
   * @brief Left index of the adjacent pair with the smallest IV sum.
   */
  size_t find_min_iv_merge() const {
    double min_iv_sum = std::numeric_limits<double>::max();
    size_t idx = 0;
    for (size_t i = 0; i + 1 < bins.size(); i++) {
      const double iv_sum = bins[i].iv + bins[i + 1].iv;
      if (iv_sum < min_iv_sum) {
        min_iv_sum = iv_sum;
        idx = i;
      }
    }
    return idx;
  }

  bool guess_trend() const {
    int inc = 0;
    int dec = 0;
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe > bins[i - 1].woe) inc++;
      else if (bins[i].woe < bins[i - 1].woe) dec++;
    }
    return inc >= dec;
  }

  /**
   * @brief Merge bins i and i+1 into bin i.
   */
  void merge_bins(size_t i) {
    const size_t j = i + 1;
    bins[i].upper_bound = bins[j].upper_bound;
    bins[i].count += bins[j].count;
    bins[i].count_pos += bins[j].count_pos;
    bins[i].count_neg += bins[j].count_neg;
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(j));
  }

  /**
   * @brief Cutpoints = lower bounds of bins 2..k (strictly increasing, finite).
   *
   * No tolerance-based de-duplication: the shared 1e-10 absolute tolerance
   * silently dropped genuine cutpoints of small-scale features, leaving fewer
   * cutpoints than bins.
   */
  void update_cutpoints() {
    cutpoints.clear();
    for (size_t i = 1; i < bins.size(); i++) {
      cutpoints.push_back(bins[i].lower_bound);
    }
  }
};


// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ubsd(
   Rcpp::NumericVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   double laplace_smoothing = 0.5
) {
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 if (min_bins < 2) {
   Rcpp::stop("min_bins must be at least 2.");
 }
 if (max_bins < min_bins) {
   Rcpp::stop("max_bins must be greater than or equal to min_bins.");
 }
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1.");
 }
 if (max_n_prebins < min_bins) {
   Rcpp::stop("max_n_prebins must be at least min_bins.");
 }
 if (convergence_threshold <= 0) {
   Rcpp::stop("convergence_threshold must be positive.");
 }
 if (max_iterations <= 0) {
   Rcpp::stop("max_iterations must be positive.");
 }
 if (laplace_smoothing < 0) {
   Rcpp::stop("laplace_smoothing must be non-negative.");
 }

 std::vector<double> f = as<std::vector<double>>(feature);
 std::vector<double> t = as<std::vector<double>>(target);

 try {
   OBN_UBSD model(
       f, t,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       convergence_threshold, max_iterations,
       laplace_smoothing
   );
   model.fit();
   return model.create_output();
 } catch (const std::exception &e) {
   Rcpp::stop(std::string("Error in optimal_binning_numerical_ubsd: ") + e.what());
 }
}
