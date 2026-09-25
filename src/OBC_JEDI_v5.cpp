// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Global constants for better precision and readability
static constexpr double EPS = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();

// Main class. All merge decisions compare the total IV of the binning that a
// candidate merge would produce, summed bin by bin in bin order exactly as
// calculate_iv() does; the fast paths below reproduce that value bit for bit
// (see find_min_loss_pair()).
class OBC_JEDI {
private:
  std::vector<std::string> feature_;
  std::vector<int> target_;
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;

  std::vector<CategoricalBin> bins_;
  int total_pos_;
  int total_neg_;
  bool converged_;
  int iterations_run_;

  // Per-category counts, built once in the constructor (it also yields the
  // number of categories, which used to cost two extra hash passes over the
  // data). Same reserve and insertion sequence as before, so the iteration
  // order -- which feeds the unstable sorts below -- is unchanged.
  std::unordered_map<std::string, CategoricalBin> bin_map_;
  // Categories in order of first appearance (used only by the <= 2 path).
  std::vector<std::string> first_seen_;
  int ncat_;

  // Enhanced input validation with comprehensive checks
  void validate_inputs() {
    if (feature_.size() != target_.size()) {
      throw std::invalid_argument(
          "Feature and target vectors must have the same length");
    }
    if (feature_.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty");
    }
    // A non-positive min_bins was cast to size_t in the merge loops (a huge
    // bound, so nothing was ever merged and max_bins was ignored).
    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1");
    }

    // Check for empty strings in feature
    if (std::any_of(feature_.begin(), feature_.end(),
                    [](const std::string &s) { return s.empty(); })) {
      throw std::invalid_argument("Feature cannot contain empty strings. "
                                  "Consider preprocessing your data.");
    }

    // Every value is checked: stopping as soon as both 0 and 1 had been seen
    // let a later 2 (or -1) through, and it was then counted as
    // count_pos += 2, count_neg += -1, i.e. negative counts.
    bool has_zero = false;
    bool has_one = false;
    for (int t : target_) {
      if (t == 0)
        has_zero = true;
      else if (t == 1)
        has_one = true;
      else
        throw std::invalid_argument("Target must be binary (0/1)");
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
  }

  // Contribution of one bin to calculate_iv(): identical expression, so the
  // value is bit-identical to what calculate_iv() adds for that bin.
  inline double iv_term(int count_pos, int count_neg) const {
    double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                       static_cast<double>(total_pos_) /
                       (total_pos_ + total_neg_);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

    double pos_rate = static_cast<double>(count_pos + prior_pos) /
                      static_cast<double>(total_pos_ + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(count_neg + prior_neg) /
                      static_cast<double>(total_neg_ + BAYESIAN_PRIOR_STRENGTH);

    if (pos_rate > EPS && neg_rate > EPS) {
      double woe = std::log(pos_rate / neg_rate);
      double bin_iv = (pos_rate - neg_rate) * woe;
      if (std::isfinite(bin_iv)) {
        return bin_iv;
      }
    }
    return 0.0;
  }

  // Total IV with Bayesian smoothing, summed in bin order.
  double calculate_iv(const std::vector<CategoricalBin> &current_bins) const {
    double iv = 0.0;
    for (const auto &bin : current_bins) {
      iv += iv_term(bin.count_pos, bin.count_neg);
    }
    return iv;
  }

  // Enhanced WoE and IV calculation for all bins
  void compute_woe_iv(std::vector<CategoricalBin> &current_bins) {
    for (auto &bin : current_bins) {
      bin.calculate_metrics(total_pos_, total_neg_);
    }
  }

  // Enhanced monotonicity check with adaptive threshold
  bool is_monotonic(const std::vector<CategoricalBin> &current_bins) const {
    if (current_bins.size() <= 2)
      return true;

    // Calculate average WoE gap for context-aware check
    double total_gap = 0.0;
    for (size_t i = 1; i < current_bins.size(); ++i) {
      total_gap += std::abs(current_bins[i].woe - current_bins[i - 1].woe);
    }

    double avg_gap =
        total_gap / static_cast<double>(current_bins.size() - 1);

    // Adaptive threshold based on average gap
    double monotonicity_threshold = std::min(EPS, avg_gap * 0.01);

    // Check the direction in the first two bins
    bool should_increase =
        current_bins[1].woe >= current_bins[0].woe - monotonicity_threshold;

    for (size_t i = 2; i < current_bins.size(); ++i) {
      if (should_increase && current_bins[i].woe < current_bins[i - 1].woe -
                                                       monotonicity_threshold) {
        return false;
      }
      if (!should_increase &&
          current_bins[i].woe >
              current_bins[i - 1].woe + monotonicity_threshold) {
        return false;
      }
    }
    return true;
  }

  // Optimized category name joining with pre-allocation
  static std::string join_categories(const std::vector<std::string> &cats,
                                     const std::string &sep) {
    if (cats.empty())
      return "";
    if (cats.size() == 1)
      return cats[0];

    size_t total_size = 0;
    for (const auto &cat : cats) {
      total_size += cat.size();
    }
    total_size += sep.size() * (cats.size() - 1);

    std::string result;
    result.reserve(total_size);
    result = cats[0];
    for (size_t i = 1; i < cats.size(); ++i) {
      result += sep;
      result += cats[i];
    }
    return result;
  }

  // Count every category in one pass (single hash lookup per row).
  void count_categories() {
    size_t est_cats = std::min(feature_.size() / 4, static_cast<size_t>(1024));
    bin_map_.reserve(est_cats);
    first_seen_.clear();

    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string &cat = feature_[i];
      const int val = target_[i];

      auto it = bin_map_.find(cat);
      if (it == bin_map_.end()) {
        it = bin_map_.emplace(cat, CategoricalBin()).first;
        it->second.categories.push_back(cat);
        if (first_seen_.size() < 3) {
          first_seen_.push_back(cat);
        }
      }
      CategoricalBin &bin = it->second;
      bin.count++;
      bin.count_pos += val;
      bin.count_neg += (1 - val);
    }
    ncat_ = static_cast<int>(bin_map_.size());
  }

  // Enhanced bin initialization with improved statistical handling
  void initial_binning() {
    total_pos_ = 0;
    total_neg_ = 0;
    for (int val : target_) {
      total_pos_ += val;
      total_neg_ += (1 - val);
    }

    // Check for extremely imbalanced datasets
    if (total_pos_ < 5 || total_neg_ < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results "
                    "may be unstable.");
    }

    // Transfer to final vector
    bins_.clear();
    bins_.reserve(bin_map_.size());
    for (auto &kv : bin_map_) {
      bins_.push_back(std::move(kv.second));
    }
    bin_map_.clear();
  }

  // Enhanced rare category merging with improved handling
  void merge_low_freq() {
    int total_count = 0;
    for (auto &b : bins_) {
      total_count += b.count;
    }
    double cutoff_count = total_count * bin_cutoff_;

    // Sort by frequency (rarest first)
    std::sort(bins_.begin(), bins_.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.count < b.count;
              });

    // Up to min_bins rare categories stay separate (as they always have); the
    // rest are pooled. The rare categories occupy the front of the ascending
    // order, so the ones kept are now the last -- most frequent -- of them.
    // The previous test (`new_bins.size() < min_bins_` while walking that
    // order) kept the min_bins *rarest* categories -- typically singletons --
    // as bins of their own and pooled the larger rare ones, producing
    // 1-observation bins in high-cardinality features.
    size_t n_rare = 0;
    for (const auto &b : bins_) {
      if (b.count < cutoff_count)
        ++n_rare;
    }
    const size_t keep_rare =
        std::min(n_rare, static_cast<size_t>(std::max(min_bins_, 0)));

    std::vector<CategoricalBin> new_bins;
    new_bins.reserve(bins_.size());

    CategoricalBin others;

    for (size_t pos = 0; pos < bins_.size(); ++pos) {
      auto &b = bins_[pos];
      if (b.count >= cutoff_count || pos >= n_rare - keep_rare) {
        new_bins.push_back(std::move(b));
      } else {
        others.categories.insert(others.categories.end(),
                                 std::make_move_iterator(b.categories.begin()),
                                 std::make_move_iterator(b.categories.end()));
        others.count += b.count;
        others.count_pos += b.count_pos;
        others.count_neg += b.count_neg;
        others.update_count();
      }
    }

    if (others.count > 0) {
      new_bins.push_back(std::move(others));
    }

    bins_ = std::move(new_bins);
  }

  // Enhanced monotonic ordering with stability improvements
  void ensure_monotonic_order() {
    compute_woe_iv(bins_);
    std::sort(bins_.begin(), bins_.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.woe < b.woe;
              });
  }

  // Merge bin idx+1 into bin idx. Only the merged bin's WoE/IV change (each
  // bin's metrics depend on its own counts and the fixed totals), so it is the
  // only one recomputed.
  void merge_bins(size_t idx) {
    CategoricalBin &dst = bins_[idx];
    CategoricalBin &src = bins_[idx + 1];
    dst.categories.insert(dst.categories.end(),
                          std::make_move_iterator(src.categories.begin()),
                          std::make_move_iterator(src.categories.end()));
    dst.count += src.count;
    dst.count_pos += src.count_pos;
    dst.count_neg += src.count_neg;
    dst.update_count();
    dst.calculate_metrics(total_pos_, total_neg_);
    bins_.erase(bins_.begin() + static_cast<std::ptrdiff_t>(idx) + 1);
  }

  // Total IV of the binning obtained by merging bins i and i+1, summed in the
  // same order as calculate_iv() on that binning: prefix[i] is the running
  // sum of terms[0..i-1], then the merged bin, then terms[i+2..].
  static double iv_after_merge(const std::vector<double> &prefix,
                               const std::vector<double> &terms, size_t i,
                               double merged_term) {
    double acc = prefix[i];
    acc += merged_term;
    for (size_t j = i + 2; j < terms.size(); ++j) {
      acc += terms[j];
    }
    return acc;
  }

  // Adjacent pair whose merge loses the least IV: the first i minimising
  // calculate_iv(bins) - calculate_iv(bins with i, i+1 merged).
  //
  // The previous code materialised a full copy of the bin vector for every
  // candidate (O(B) string-vector copies per pair, O(B^3) per reduction).
  // Evaluating that exact difference for every pair is still O(B^2) per
  // merge, so the local form terms[i] + terms[i+1] - merged, which equals it
  // up to rounding, screens the candidates first and only those within a
  // rigorous rounding bound of the best are evaluated exactly. The exact
  // values decide, with the same strict "<" in index order, so the chosen
  // pair is the one the full evaluation would choose.
  size_t find_min_loss_pair(const std::vector<double> &terms) const {
    const size_t nb = bins_.size();
    std::vector<double> merged(nb - 1);
    std::vector<double> local(nb - 1);
    double max_merged = 0.0;
    double min_local = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < nb; ++i) {
      merged[i] = iv_term(bins_[i].count_pos + bins_[i + 1].count_pos,
                          bins_[i].count_neg + bins_[i + 1].count_neg);
      local[i] = (terms[i] + terms[i + 1]) - merged[i];
      max_merged = std::max(max_merged, std::fabs(merged[i]));
      min_local = std::min(min_local, local[i]);
    }

    std::vector<double> prefix(nb + 1);
    prefix[0] = 0.0;
    double abs_sum = 0.0;
    for (size_t j = 0; j < nb; ++j) {
      prefix[j + 1] = prefix[j] + terms[j];
      abs_sum += std::fabs(terms[j]);
    }
    const double original_iv = prefix[nb];

    // |exact - local| is bounded by the rounding of two length-nb sums plus
    // a few operations; 4x that bound on each side keeps every pair that
    // could be the exact minimiser.
    const double bound = 4.0 * static_cast<double>(2 * nb + 8) * DBL_EPSILON *
                         (abs_sum + max_merged);
    const double cut = min_local + 2.0 * bound;

    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;
    for (size_t i = 0; i + 1 < nb; ++i) {
      if (!(local[i] <= cut))
        continue;
      const double iv_loss =
          original_iv - iv_after_merge(prefix, terms, i, merged[i]);
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }
    return merge_index;
  }

  // Greedy minimum-IV-loss merging of WoE-adjacent bins while the bin count
  // is above `target` (and above `floor_bins`).
  void reduce_bins(size_t target, size_t floor_bins) {
    if (bins_.size() <= target || bins_.size() <= floor_bins ||
        bins_.size() < 2)
      return;
    std::vector<double> terms(bins_.size());
    for (size_t j = 0; j < bins_.size(); ++j) {
      terms[j] = iv_term(bins_[j].count_pos, bins_[j].count_neg);
    }
    while (bins_.size() > target && bins_.size() > floor_bins &&
           bins_.size() >= 2) {
      const size_t idx = find_min_loss_pair(terms);
      merge_bins(idx);
      terms.erase(terms.begin() + static_cast<std::ptrdiff_t>(idx) + 1);
      terms[idx] = iv_term(bins_[idx].count_pos, bins_[idx].count_neg);
      Rcpp::checkUserInterrupt();
    }
  }

  // Enhanced pre-bin merging with improved IV consideration
  void merge_adjacent_bins_for_prebins() {
    reduce_bins(static_cast<size_t>(std::max(max_n_prebins_, 0)),
                static_cast<size_t>(std::max(min_bins_, 0)));
  }

  // Enhanced monotonicity improvement with smarter bin selection
  void improve_monotonicity_step() {
    std::vector<std::pair<size_t, double>> violations;

    for (size_t i = 1; i + 1 < bins_.size(); ++i) {
      bool is_peak = bins_[i].woe > bins_[i - 1].woe + EPS &&
                     bins_[i].woe > bins_[i + 1].woe + EPS;
      bool is_valley = bins_[i].woe < bins_[i - 1].woe - EPS &&
                       bins_[i].woe < bins_[i + 1].woe - EPS;

      if (is_peak || is_valley) {
        double severity = std::max(std::abs(bins_[i].woe - bins_[i - 1].woe),
                                   std::abs(bins_[i].woe - bins_[i + 1].woe));
        violations.push_back({i, severity});
      }
    }

    if (violations.empty())
      return;

    // Sort by severity (largest first)
    std::sort(violations.begin(), violations.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Fix the worst violation
    size_t i = violations[0].first;

    std::vector<double> terms(bins_.size());
    std::vector<double> prefix(bins_.size() + 1);
    prefix[0] = 0.0;
    for (size_t j = 0; j < bins_.size(); ++j) {
      terms[j] = iv_term(bins_[j].count_pos, bins_[j].count_neg);
      prefix[j + 1] = prefix[j] + terms[j];
    }
    double orig_iv = prefix[bins_.size()];

    double iv_merge_1 = iv_after_merge(
        prefix, terms, i - 1,
        iv_term(bins_[i - 1].count_pos + bins_[i].count_pos,
                bins_[i - 1].count_neg + bins_[i].count_neg));
    double iv_merge_2 = iv_after_merge(
        prefix, terms, i,
        iv_term(bins_[i].count_pos + bins_[i + 1].count_pos,
                bins_[i].count_neg + bins_[i + 1].count_neg));

    double loss1 = orig_iv - iv_merge_1;
    double loss2 = orig_iv - iv_merge_2;

    if (loss1 < loss2) {
      merge_bins(i - 1);
    } else {
      merge_bins(i);
    }
  }

  // Enhanced main optimization algorithm with better convergence properties
  void optimize() {
    double prev_iv = calculate_iv(bins_);
    converged_ = false;
    iterations_run_ = 0;

    // Track best solution seen so far
    double best_iv = prev_iv;
    std::vector<CategoricalBin> best_bins = bins_;

    while (iterations_run_ < max_iterations_) {
      if (is_monotonic(bins_) &&
          bins_.size() <= static_cast<size_t>(max_bins_) &&
          bins_.size() >= static_cast<size_t>(min_bins_)) {

        double current_iv = calculate_iv(bins_);
        if (current_iv > best_iv) {
          best_iv = current_iv;
          best_bins = bins_;
        }

        converged_ = true;
        break;
      }

      if (bins_.size() > static_cast<size_t>(min_bins_)) {
        if (bins_.size() > static_cast<size_t>(max_bins_)) {
          merge_adjacent_bins_for_prebins();
        } else {
          improve_monotonicity_step();
        }
      } else {
        // Can't achieve min_bins, stop
        break;
      }

      double current_iv = calculate_iv(bins_);

      // Track best solution even if not converged
      if (current_iv > best_iv &&
          bins_.size() <= static_cast<size_t>(max_bins_) &&
          bins_.size() >= static_cast<size_t>(min_bins_)) {
        best_iv = current_iv;
        best_bins = bins_;
      }

      if (std::abs(current_iv - prev_iv) < convergence_threshold_) {
        converged_ = true;
        break;
      }
      prev_iv = current_iv;
      iterations_run_++;
    }

    // Restore best solution if we've seen a valid one
    if (best_iv > NEG_INFINITY && !best_bins.empty()) {
      bins_ = std::move(best_bins);
    }

    // Adjust bins if still above max_bins_
    reduce_bins(static_cast<size_t>(std::max(max_bins_, 1)), 1);

    ensure_monotonic_order();
    compute_woe_iv(bins_);
  }

public:
  OBC_JEDI(std::vector<std::string> feature, std::vector<int> target,
           int min_bins, int max_bins, double bin_cutoff, int max_n_prebins,
           std::string bin_separator, double convergence_threshold,
           int max_iterations)
      : feature_(std::move(feature)), target_(std::move(target)),
        min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
        max_n_prebins_(max_n_prebins),
        bin_separator_(std::move(bin_separator)),
        convergence_threshold_(convergence_threshold),
        max_iterations_(max_iterations), bins_(), total_pos_(0),
        total_neg_(0), converged_(false), iterations_run_(0), ncat_(0) {
    validate_inputs();

    // Adjust parameters based on unique category count
    count_categories();
    const int ncat = ncat_;

    if (ncat < min_bins_) {
      min_bins_ = std::max(1, ncat);
    }
    if (max_bins_ < min_bins_) {
      max_bins_ = min_bins_;
    }
    if (max_n_prebins_ < min_bins_) {
      max_n_prebins_ = min_bins_;
    }
  }

  void fit() {
    if (ncat_ <= 2) {
      // Trivial case: <= 2 categories, one bin each. The bins come out in the
      // iteration order of an un-reserved map filled in order of first
      // appearance, as they always have.
      std::unordered_map<std::string, CategoricalBin> small_map;
      for (const auto &cat : first_seen_) {
        small_map[cat].categories.push_back(cat);
      }
      int total_pos = 0, total_neg = 0;
      for (size_t i = 0; i < feature_.size(); ++i) {
        auto &bin = small_map[feature_[i]];
        bin.count++;
        bin.count_pos += target_[i];
        bin.count_neg += (1 - target_[i]);
        total_pos += target_[i];
        total_neg += (1 - target_[i]);
      }

      bins_.clear();
      bins_.reserve(small_map.size());
      for (auto &kv : small_map) {
        bins_.push_back(std::move(kv.second));
      }

      total_pos_ = total_pos;
      total_neg_ = total_neg;
      compute_woe_iv(bins_);
      converged_ = true;
      iterations_run_ = 0;
      return;
    }

    // Normal flow for many categories
    initial_binning();
    merge_low_freq();
    compute_woe_iv(bins_);
    ensure_monotonic_order();

    if (static_cast<int>(bins_.size()) > max_n_prebins_) {
      merge_adjacent_bins_for_prebins();
    }

    optimize();
  }

  Rcpp::List get_results() const {
    const size_t n_bins = bins_.size();

    CharacterVector bin_names(n_bins);
    NumericVector woes(n_bins);
    NumericVector ivs(n_bins);
    IntegerVector counts(n_bins);
    IntegerVector counts_pos(n_bins);
    IntegerVector counts_neg(n_bins);
    NumericVector ids(n_bins);

    double total_iv = 0.0;

    for (size_t i = 0; i < n_bins; ++i) {
      bin_names[i] = join_categories(bins_[i].categories, bin_separator_);
      woes[i] = bins_[i].woe;
      ivs[i] = bins_[i].iv;
      counts[i] = bins_[i].count;
      counts_pos[i] = bins_[i].count_pos;
      counts_neg[i] = bins_[i].count_neg;
      ids[i] = static_cast<double>(i + 1);

      total_iv += bins_[i].iv;
    }

    return Rcpp::List::create(
        Named("id") = ids, Named("bin") = bin_names, Named("woe") = woes,
        Named("iv") = ivs, Named("count") = counts,
        Named("count_pos") = counts_pos, Named("count_neg") = counts_neg,
        Named("total_iv") = total_iv, Named("converged") = converged_,
        Named("iterations") = iterations_run_);
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi(
    Rcpp::IntegerVector target, Rcpp::StringVector feature, int min_bins = 3,
    int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
    std::string bin_separator = "%;%", double convergence_threshold = 1e-6,
    int max_iterations = 1000) {
  // Preliminary validations
  if (feature.size() == 0 || target.size() == 0) {
    stop("Feature and target vectors cannot be empty");
  }

  if (feature.size() != target.size()) {
    stop("Feature and target vectors must have the same length");
  }

  std::vector<std::string> feature_vec;
  std::vector<int> target_vec;

  feature_vec.reserve(static_cast<size_t>(feature.size()));
  target_vec.reserve(static_cast<size_t>(target.size()));

  int na_feature_count = 0;

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    // NA handling in feature
    SEXP s = STRING_ELT(feature, i);
    if (s == NA_STRING) {
      feature_vec.emplace_back("NA");
      na_feature_count++;
    } else {
      feature_vec.emplace_back(CHAR(s));
    }

    // NA handling in target
    if (IntegerVector::is_na(target[i])) {
      stop("Target cannot contain NA values at position %d",
           static_cast<long long>(i) + 1);
    } else {
      target_vec.push_back(target[i]);
    }
  }

  // Warn about NA values in feature
  if (na_feature_count > 0) {
    Rcpp::warning(
        "%d missing values found in feature and converted to \"NA\" category.",
        na_feature_count);
  }

  try {
    OBC_JEDI jedi(std::move(feature_vec), std::move(target_vec), min_bins,
                  max_bins, bin_cutoff, max_n_prebins, bin_separator,
                  convergence_threshold, max_iterations);
    jedi.fit();
    return jedi.get_results();
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimal_binning_categorical_jedi: %s", e.what());
  }
}
