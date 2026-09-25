// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <utility>
#include <sstream>
#include <iomanip>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

/**
 * @brief A finite cut c with a <= c < b separating two consecutive distinct
 * values a < b under the right-closed (lower, upper] convention.
 *
 * The plain midpoint (a + b) / 2 is used whenever it is valid, so ordinary data
 * is unaffected. It overflows to +/-Inf when a and b are both near the largest
 * finite double, and for two adjacent doubles it can round up to b itself; in
 * those cases a/2 + b/2, and then a, are used instead.
 */
inline double mob_safe_cut(double a, double b) {
  double m = (a + b) / 2.0;
  if (!std::isfinite(m)) m = a / 2.0 + b / 2.0;
  if (std::isfinite(m) && m >= a && m < b) return m;
  if (std::isfinite(a)) return a;
  // a == -Inf: the only finite value that keeps -Inf alone is the lowest double
  return std::numeric_limits<double>::lowest();
}

} // namespace

/**
 * @brief Monotonic Optimal Binning (MOB) for Numerical Features
 *
 * This class implements the Monotonic Optimal Binning algorithm for numerical features
 * in credit scoring and risk modeling applications. The algorithm ensures monotonicity
 * in the Weight of Evidence (WoE) values, which is often a desirable property for
 * interpretability and stability in production models.
 *
 * Key features:
 * 1. Creates initial pre-bins based on frequency distribution
 * 2. Merges rare bins to ensure statistical stability
 * 3. Enforces monotonicity in WoE values
 * 4. Optimizes bin boundaries to maximize information value
 * 5. Handles special cases (few unique values, extreme values)
 * 6. Applies Laplace smoothing for robust WoE calculation
 *
 * Missing values (NaN / NA) are removed silently before binning: they are
 * neither counted in any bin nor in the totals used for WoE, IV and
 * bin_cutoff. Infinite values are kept as extreme values: -Inf falls in the
 * first bin and +Inf in the last, and neither ever becomes a cutpoint.
 */
class OBN_MOB {
public:
  OBN_MOB(int min_bins_ = 3,
          int max_bins_ = 5,
          double bin_cutoff_ = 0.05,
          int max_n_prebins_ = 20,
          double convergence_threshold_ = 1e-6,
          int max_iterations_ = 1000,
          double laplace_smoothing_ = 0.5)
    : min_bins(min_bins_),
      max_bins(std::max(max_bins_, min_bins_)),
      bin_cutoff(bin_cutoff_),
      max_n_prebins(std::max(max_n_prebins_, min_bins_)),
      convergence_threshold(convergence_threshold_),
      max_iterations(max_iterations_),
      laplace_smoothing(laplace_smoothing_),
      converged(false),
      iterations(0),
      total_count(0),
      total_pos(0),
      total_neg(0) {
    validate_parameters();
  }

  /**
   * @brief Fit the MOB model to the provided data
   */
  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    validate_input_data(feature_, target_);

    // Sorted (value, target) pairs with NaN removed.
    std::vector<std::pair<double, int>> sorted_data = prepare_sorted_data(feature_, target_);
    if (sorted_data.empty()) {
      Rcpp::stop("All feature values are missing (NA/NaN); nothing to bin.");
    }

    // Totals over the observations that are actually binned.
    total_count = static_cast<int>(sorted_data.size());
    total_pos = 0;
    for (const auto& p : sorted_data) total_pos += p.second;
    total_neg = total_count - total_pos;

    // Distinct finite values and distinct values overall (including +/-Inf),
    // counted on the sorted data instead of through a std::set.
    int n_unique = 0;
    int actual_unique = 0;
    for (size_t i = 0; i < sorted_data.size(); i++) {
      const double v = sorted_data[i].first;
      const bool is_new = (i == 0) || (v != sorted_data[i - 1].first);
      if (is_new) {
        actual_unique++;
        if (std::isfinite(v)) n_unique++;
      }
    }

    adjust_bin_parameters(n_unique);

    // Very few distinct values: the binning is exact.
    if (actual_unique <= 2) {
      handle_few_unique_values(sorted_data, actual_unique);
      converged = true;
      iterations = 0;
      return;
    }

    create_prebins(sorted_data, n_unique);
    optimize_bins();
    calculate_woe_iv();
  }

  std::vector<NumericalBin> get_bin_metrics() const {
    return bins;
  }

  /**
   * @brief Cutpoints: the upper bounds of bins 1..k-1, strictly increasing and
   * finite by construction.
   */
  std::vector<double> get_cutpoints() const {
    std::vector<double> cp;
    if (bins.size() < 2) return cp;
    cp.reserve(bins.size() - 1);
    for (size_t i = 0; i + 1 < bins.size(); i++) {
      cp.push_back(bins[i].upper_bound);
    }
    return cp;
  }

  bool has_converged() const {
    return converged;
  }

  int get_iterations() const {
    return iterations;
  }

  double get_total_iv() const {
    double total_iv = 0.0;
    for (const auto& bin : bins) {
      total_iv += bin.iv;
    }
    return total_iv;
  }

private:
  // Algorithm parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  double laplace_smoothing;

  // State variables
  bool converged;
  int iterations;

  // Totals over the binned (non-missing) observations
  int total_count;
  int total_pos;
  int total_neg;

  std::vector<NumericalBin> bins;

  void validate_parameters() const {
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2.");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      Rcpp::stop("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0.0) {
      Rcpp::stop("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
    if (laplace_smoothing < 0.0) {
      Rcpp::stop("laplace_smoothing must be non-negative.");
    }
  }

  void validate_input_data(const std::vector<double>& feature_, const std::vector<int>& target_) const {
    if (feature_.size() != target_.size()) {
      Rcpp::stop("feature and target must have the same length.");
    }
    if (feature_.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    bool has_zero = false, has_one = false;
    for (int t : target_) {
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
   * @brief Sort (feature, target) pairs by feature, dropping NaN silently.
   */
  std::vector<std::pair<double, int>> prepare_sorted_data(const std::vector<double>& feature_,
                                                          const std::vector<int>& target_) const {
    std::vector<std::pair<double, int>> sorted_data;
    sorted_data.reserve(feature_.size());
    for (size_t i = 0; i < feature_.size(); i++) {
      const double v = feature_[i];
      if (!std::isnan(v)) sorted_data.emplace_back(v, target_[i]);
    }
    std::sort(sorted_data.begin(), sorted_data.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first < b.first;
              });
    return sorted_data;
  }

  void adjust_bin_parameters(int n_unique) {
    if (n_unique < min_bins) {
      min_bins = std::max(1, n_unique);  // only lowers min_bins: still <= max_bins
    }
    if (n_unique < max_bins) {
      max_bins = n_unique;
    }
  }

  /**
   * @brief One bin (one distinct value) or two bins (two distinct values).
   */
  void handle_few_unique_values(const std::vector<std::pair<double, int>>& sorted_data, int unique_count) {
    bins.clear();

    if (unique_count == 1) {
      NumericalBin bin;
      bin.lower_bound = -std::numeric_limits<double>::infinity();
      bin.upper_bound = std::numeric_limits<double>::infinity();
      bin.count = total_count;
      bin.count_pos = total_pos;
      bin.count_neg = total_neg;
      bins.push_back(bin);
    } else {
      const double val1 = sorted_data.front().first;
      const double val2 = sorted_data.back().first;

      NumericalBin bin1, bin2;
      bin1.lower_bound = -std::numeric_limits<double>::infinity();
      bin1.upper_bound = mob_safe_cut(val1, val2);
      bin2.lower_bound = bin1.upper_bound;
      bin2.upper_bound = std::numeric_limits<double>::infinity();

      for (const auto& p : sorted_data) {
        NumericalBin& b = (p.first == val1) ? bin1 : bin2;
        b.count++;
        if (p.second == 1) b.count_pos++; else b.count_neg++;
      }
      bins.push_back(bin1);
      bins.push_back(bin2);
    }
    calculate_woe_iv();
  }

  /**
   * @brief Equal-frequency pre-bins, never splitting a run of tied values.
   */
  void create_prebins(const std::vector<std::pair<double, int>>& sorted_data, int n_unique) {
    bins.clear();

    const size_t n = sorted_data.size();
    size_t n_prebins = std::min(static_cast<size_t>(max_n_prebins), static_cast<size_t>(n_unique));
    n_prebins = std::max(static_cast<size_t>(min_bins), n_prebins);

    size_t bin_size = n / n_prebins;
    if (bin_size < 1) bin_size = 1;

    const double neg_inf = -std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < n; ) {
      size_t end = std::min(i + bin_size, n);

      // Never split a run of tied values across two prebins. A pre-bin also
      // never ends on -Inf: its upper bound would be the non-finite cutpoint
      // -Inf, so the -Inf observations are pooled with the next value run.
      while (end < n && (sorted_data[end].first == sorted_data[end - 1].first ||
                         sorted_data[end - 1].first == neg_inf)) {
        ++end;
      }

      NumericalBin bin;
      bin.lower_bound = bins.empty() ? neg_inf : bins.back().upper_bound;
      bin.upper_bound = (end == n) ? std::numeric_limits<double>::infinity()
                                   : sorted_data[end - 1].first;
      bin.count = static_cast<int>(end - i);
      for (size_t j = i; j < end; j++) {
        if (sorted_data[j].second == 1) bin.count_pos++; else bin.count_neg++;
      }
      bins.push_back(bin);
      i = end;
    }
  }

  /**
   * @brief Rare-bin merging, monotonicity enforcement and max_bins reduction.
   */
  void optimize_bins() {
    iterations = 0;
    converged = true;

    calculate_woe_iv();  // the merge direction below needs the IVs
    merge_rare_bins();
    calculate_woe_iv();

    if (!is_monotonic_woe()) {
      enforce_monotonicity();
    }

    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      merge_bins(find_optimal_merge());
      iterations++;
    }

    // With Laplace smoothing the WoE of a merged bin is not necessarily between
    // the WoE of its two parts, so the max_bins reduction above can re-create
    // a violation. Enforce again (merging only lowers the bin count).
    if (!is_monotonic_woe()) {
      enforce_monotonicity();
    }

    // Reported through `converged` (no R warning).
    if (iterations >= max_iterations) {
      converged = false;
    }
  }

  void merge_rare_bins() {
    const double total = static_cast<double>(total_count);

    while (iterations < max_iterations && static_cast<int>(bins.size()) > min_bins) {
      double min_freq = std::numeric_limits<double>::max();
      size_t min_freq_idx = 0;
      for (size_t i = 0; i < bins.size(); i++) {
        const double freq = static_cast<double>(bins[i].count) / total;
        if (freq < min_freq) {
          min_freq = freq;
          min_freq_idx = i;
        }
      }

      if (!(min_freq < bin_cutoff)) break;

      if (min_freq_idx == 0) {
        merge_bins(0);
      } else if (min_freq_idx == bins.size() - 1) {
        merge_bins(bins.size() - 2);
      } else {
        const double iv_loss_left = bins[min_freq_idx - 1].iv + bins[min_freq_idx].iv;
        const double iv_loss_right = bins[min_freq_idx].iv + bins[min_freq_idx + 1].iv;
        merge_bins(iv_loss_left <= iv_loss_right ? min_freq_idx - 1 : min_freq_idx);
      }
      iterations++;
    }
  }

  /**
   * @brief WoE is non-decreasing or non-increasing.
   *
   * The direction is not inferred from the first pair alone: with a tie there
   * ([a, a, b] with b < a) that rule read a non-increasing sequence as
   * "increasing", reported it as a violation and merged bins needlessly.
   */
  bool is_monotonic_woe() const {
    bool non_decreasing = true;
    bool non_increasing = true;
    for (size_t i = 1; i < bins.size(); i++) {
      if (bins[i].woe < bins[i - 1].woe) non_decreasing = false;
      if (bins[i].woe > bins[i - 1].woe) non_increasing = false;
    }
    return non_decreasing || non_increasing;
  }

  void enforce_monotonicity() {
    // Direction by majority vote over adjacent pairs. Taking it from the first
    // two bins alone let sampling noise in the two lowest pre-bins decide, and
    // on a clearly increasing feature the "decreasing" guess merged everything
    // above them into one bin, stopping at min_bins with non-monotonic WoE.
    int n_inc = 0;
    int n_dec = 0;
    for (size_t k = 1; k < bins.size(); k++) {
      if (bins[k].woe > bins[k - 1].woe) n_inc++;
      else if (bins[k].woe < bins[k - 1].woe) n_dec++;
    }
    const bool should_increase = n_inc >= n_dec;

    while (!is_monotonic_woe() && iterations < max_iterations && static_cast<int>(bins.size()) > min_bins) {
      // First violation of the direction fixed on entry. The loop is entered
      // only for a sequence that is neither non-decreasing nor non-increasing,
      // so a violation of either direction exists; the bound is defensive.
      size_t i = 1;
      while (i < bins.size() &&
             ((should_increase && !(bins[i].woe < bins[i - 1].woe)) ||
              (!should_increase && !(bins[i].woe > bins[i - 1].woe)))) {
        ++i;
      }
      if (i >= bins.size()) break;

      // Would merging i-1 and i fix the violation with both neighbours?
      NumericalBin merged_bin = bins[i - 1];
      merged_bin.upper_bound = bins[i].upper_bound;
      merged_bin.count += bins[i].count;
      merged_bin.count_pos += bins[i].count_pos;
      merged_bin.count_neg += bins[i].count_neg;
      const double merged_woe = calculate_bin_woe(merged_bin);

      bool merge_fixes = true;
      if (i > 1) {
        if ((should_increase && merged_woe < bins[i - 2].woe) ||
            (!should_increase && merged_woe > bins[i - 2].woe)) {
          merge_fixes = false;
        }
      }
      if (i < bins.size() - 1) {
        if ((should_increase && bins[i + 1].woe < merged_woe) ||
            (!should_increase && bins[i + 1].woe > merged_woe)) {
          merge_fixes = false;
        }
      }

      if (!merge_fixes && i < bins.size() - 1) {
        merge_bins(i);        // merge i and i+1 instead
      } else {
        merge_bins(i - 1);    // merge i-1 and i
      }
      iterations++;
    }
  }

  /**
   * @brief Left index of the adjacent pair whose merge loses the least IV.
   */
  size_t find_optimal_merge() const {
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_idx = 0;

    for (size_t i = 0; i + 1 < bins.size(); i++) {
      const double iv_before = bins[i].iv + bins[i + 1].iv;

      NumericalBin merged;
      merged.lower_bound = bins[i].lower_bound;
      merged.upper_bound = bins[i + 1].upper_bound;
      merged.count = bins[i].count + bins[i + 1].count;
      merged.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
      merged.count_neg = bins[i].count_neg + bins[i + 1].count_neg;

      const double woe = calculate_bin_woe(merged);
      const double iv = calculate_bin_iv(merged, woe);
      const double iv_loss = iv_before - iv;
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_idx = i;
      }
    }
    return merge_idx;
  }

  /**
   * @brief Merge bins i and i+1 into bin i and refresh WoE / IV.
   */
  void merge_bins(size_t i) {
    const size_t j = i + 1;
    bins[i].upper_bound = bins[j].upper_bound;
    bins[i].count += bins[j].count;
    bins[i].count_pos += bins[j].count_pos;
    bins[i].count_neg += bins[j].count_neg;
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(j));
    calculate_woe_iv();
  }

  double calculate_bin_woe(const NumericalBin& bin) const {
    const double smoothed_pos = bin.count_pos + laplace_smoothing;
    const double smoothed_neg = bin.count_neg + laplace_smoothing;
    const double total_smoothed_pos = total_pos + static_cast<double>(bins.size()) * laplace_smoothing;
    const double total_smoothed_neg = total_neg + static_cast<double>(bins.size()) * laplace_smoothing;
    const double dist_pos = smoothed_pos / total_smoothed_pos;
    const double dist_neg = smoothed_neg / total_smoothed_neg;

    // A merged bin is never empty, so dist_pos and dist_neg are not both 0.
    if (dist_pos <= 0.0) {
      return -20.0;
    } else if (dist_neg <= 0.0) {
      return 20.0;
    }
    return std::log(dist_pos / dist_neg);
  }

  double calculate_bin_iv(const NumericalBin& bin, double woe) const {
    const double smoothed_pos = bin.count_pos + laplace_smoothing;
    const double smoothed_neg = bin.count_neg + laplace_smoothing;
    const double total_smoothed_pos = total_pos + static_cast<double>(bins.size()) * laplace_smoothing;
    const double total_smoothed_neg = total_neg + static_cast<double>(bins.size()) * laplace_smoothing;
    const double dist_pos = smoothed_pos / total_smoothed_pos;
    const double dist_neg = smoothed_neg / total_smoothed_neg;
    return (dist_pos - dist_neg) * woe;
  }

  /**
   * @brief WoE and IV of every bin, with Laplace smoothing.
   *
   * The totals are summed over the bins, i.e. over the binned observations.
   * Both classes are always present (checked on input, and totals exclude
   * only missing features), except when every observation of one class has a
   * missing feature; that degenerate case yields WoE = IV = 0 instead of NaN.
   */
  void calculate_woe_iv() {
    int pos_total = 0;
    int neg_total = 0;
    for (const auto& bin : bins) {
      pos_total += bin.count_pos;
      neg_total += bin.count_neg;
    }

    if (pos_total == 0 || neg_total == 0) {
      for (auto& bin : bins) {
        bin.woe = 0.0;
        bin.iv = 0.0;
      }
      return;
    }

    const double total_smoothed_pos = pos_total + static_cast<double>(bins.size()) * laplace_smoothing;
    const double total_smoothed_neg = neg_total + static_cast<double>(bins.size()) * laplace_smoothing;

    for (auto& bin : bins) {
      const double dist_pos = (bin.count_pos + laplace_smoothing) / total_smoothed_pos;
      const double dist_neg = (bin.count_neg + laplace_smoothing) / total_smoothed_neg;

      if (dist_pos <= 0.0) {
        bin.woe = -20.0;  // Cap for stability (dist_neg > 0: the bin is not empty)
      } else if (dist_neg <= 0.0) {
        bin.woe = 20.0;   // Cap for stability
      } else {
        bin.woe = std::log(dist_pos / dist_neg);
      }
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
};


// [[Rcpp::export]]
List optimal_binning_numerical_mob(IntegerVector target,
                                  NumericVector feature,
                                  int min_bins = 3,
                                  int max_bins = 5,
                                  double bin_cutoff = 0.05,
                                  int max_n_prebins = 20,
                                  double convergence_threshold = 1e-6,
                                  int max_iterations = 1000,
                                  double laplace_smoothing = 0.5) {
 std::vector<double> f = as<std::vector<double>>(feature);
 std::vector<int> t = as<std::vector<int>>(target);

 OBN_MOB mob(min_bins, max_bins, bin_cutoff, max_n_prebins,
             convergence_threshold, max_iterations, laplace_smoothing);
 mob.fit(f, t);

 const std::vector<NumericalBin> bins = mob.get_bin_metrics();
 const size_t n_bins = bins.size();

 std::vector<std::string> bin_labels;
 std::vector<double> woe_values;
 std::vector<double> iv_values;
 std::vector<int> counts;
 std::vector<int> counts_pos;
 std::vector<int> counts_neg;
 std::vector<double> event_rates;
 bin_labels.reserve(n_bins);
 woe_values.reserve(n_bins);
 iv_values.reserve(n_bins);
 counts.reserve(n_bins);
 counts_pos.reserve(n_bins);
 counts_neg.reserve(n_bins);
 event_rates.reserve(n_bins);

 for (const auto& b : bins) {
   // Right-closed (lower; upper], matching the assignment rule.
   std::ostringstream oss;
   oss << std::fixed << std::setprecision(6);
   if (b.lower_bound == -std::numeric_limits<double>::infinity()) {
     oss << "(-Inf";
   } else {
     oss << "(" << b.lower_bound;
   }
   oss << ";";
   if (b.upper_bound == std::numeric_limits<double>::infinity()) {
     oss << "+Inf]";
   } else {
     oss << b.upper_bound << "]";
   }

   bin_labels.push_back(oss.str());
   woe_values.push_back(b.woe);
   iv_values.push_back(b.iv);
   counts.push_back(b.count);
   counts_pos.push_back(b.count_pos);
   counts_neg.push_back(b.count_neg);
   event_rates.push_back(static_cast<double>(b.count_pos) / std::max(1, b.count));
 }

 Rcpp::NumericVector ids(n_bins);
 for (size_t i = 0; i < n_bins; i++) {
   ids[static_cast<R_xlen_t>(i)] = static_cast<double>(i + 1);
 }

 return Rcpp::List::create(
   Named("id") = ids,
   Named("bin") = bin_labels,
   Named("woe") = woe_values,
   Named("iv") = iv_values,
   Named("count") = counts,
   Named("count_pos") = counts_pos,
   Named("count_neg") = counts_neg,
   Named("event_rate") = event_rates,
   Named("cutpoints") = mob.get_cutpoints(),
   Named("total_iv") = mob.get_total_iv(),
   Named("converged") = mob.has_converged(),
   Named("iterations") = mob.get_iterations()
 );
}
