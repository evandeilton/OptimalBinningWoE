// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Rcpp;

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Global constants for better precision and readability
static constexpr double EPS = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
// Constant removed (uses shared definition)

// Optimized structure for categorical bins with enhanced statistics
// Local CategoricalBin definition removed

// Enhanced cache for frequent IV calculations
namespace {
class IVCache {
private:
  std::vector<std::vector<double>> cache;
  bool enabled;

public:
  explicit IVCache(size_t max_size, bool use_cache = true)
      : enabled(use_cache) {
    if (enabled) {
      cache.resize(max_size);
      for (auto &row : cache) {
        row.resize(max_size, -1.0);
      }
    }
  }

  inline double get(size_t i, size_t j) {
    if (!enabled || i >= cache.size() || j >= cache.size()) {
      return -1.0;
    }
    return cache[i][j];
  }

  inline void set(size_t i, size_t j, double value) {
    if (!enabled || i >= cache.size() || j >= cache.size()) {
      return;
    }
    cache[i][j] = value;
  }

  inline void invalidate_bin(size_t idx) {
    if (!enabled || idx >= cache.size()) {
      return;
    }
    for (size_t i = 0; i < cache.size(); ++i) {
      cache[idx][i] = -1.0;
      cache[i][idx] = -1.0;
    }
  }

  inline void resize(size_t new_size) {
    if (!enabled)
      return;
    cache.resize(new_size);
    for (auto &row : cache) {
      row.resize(new_size, -1.0);
    }
  }
};
} // namespace

// Enhanced main class with improved optimization strategies
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
  std::unique_ptr<IVCache> iv_cache_;
  int total_pos_;
  int total_neg_;
  bool converged_;
  int iterations_run_;

  // Enhanced input validation with comprehensive checks
  void validate_inputs() {
    if (feature_.size() != target_.size()) {
      throw std::invalid_argument(
          "Feature and target vectors must have the same length");
    }
    if (feature_.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty");
    }

    // Check for empty strings in feature
    if (std::any_of(feature_.begin(), feature_.end(),
                    [](const std::string &s) { return s.empty(); })) {
      throw std::invalid_argument("Feature cannot contain empty strings. "
                                  "Consider preprocessing your data.");
    }

    // Fast binary value check
    bool has_zero = false;
    bool has_one = false;

    for (int t : target_) {
      if (t == 0)
        has_zero = true;
      else if (t == 1)
        has_one = true;
      else
        throw std::invalid_argument("Target must be binary (0/1)");

      // Optimization: Stop once we've confirmed both values exist
      if (has_zero && has_one)
        break;
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
  }

  // Enhanced WoE calculation with Bayesian smoothing
  inline double calculate_woe(int pos, int neg) const {
    // Calculate Bayesian prior based on overall prevalence
    double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                       static_cast<double>(total_pos_) /
                       (total_pos_ + total_neg_);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

    // Apply Bayesian smoothing to rates
    double pos_rate = static_cast<double>(pos + prior_pos) /
                      static_cast<double>(total_pos_ + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(neg + prior_neg) /
                      static_cast<double>(total_neg_ + BAYESIAN_PRIOR_STRENGTH);

    return std::log(pos_rate / neg_rate);
  }

  // Enhanced total IV calculation with Bayesian smoothing
  double calculate_iv(const std::vector<CategoricalBin> &current_bins) const {
    double iv = 0.0;
    for (const auto &bin : current_bins) {
      // Calculate Bayesian prior based on overall prevalence
      double prior_pos = BAYESIAN_PRIOR_STRENGTH *
                         static_cast<double>(total_pos_) /
                         (total_pos_ + total_neg_);
      double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

      // Apply Bayesian smoothing to rates
      double pos_rate =
          static_cast<double>(bin.count_pos + prior_pos) /
          static_cast<double>(total_pos_ + BAYESIAN_PRIOR_STRENGTH);
      double neg_rate =
          static_cast<double>(bin.count_neg + prior_neg) /
          static_cast<double>(total_neg_ + BAYESIAN_PRIOR_STRENGTH);

      if (pos_rate > EPS && neg_rate > EPS) {
        double woe = std::log(pos_rate / neg_rate);
        double bin_iv = (pos_rate - neg_rate) * woe;

        if (std::isfinite(bin_iv)) {
          iv += bin_iv;
        }
      }
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

    double avg_gap = total_gap / (current_bins.size() - 1);

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

    // Estimate total size for pre-allocation
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

  // Enhanced bin initialization with improved statistical handling
  void initial_binning() {
    // Estimate unique category count to avoid reallocations
    size_t est_cats = std::min(feature_.size() / 4, static_cast<size_t>(1024));
    std::unordered_map<std::string, CategoricalBin> bin_map;
    bin_map.reserve(est_cats);

    total_pos_ = 0;
    total_neg_ = 0;

    // Count in a single pass
    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string &cat = feature_[i];
      int val = target_[i];

      auto it = bin_map.find(cat);
      if (it == bin_map.end()) {
        auto &bin = bin_map[cat];
        bin.categories.push_back(cat);
        bin.count++;
        bin.count_pos += val;
        bin.count_neg += (1 - val);
      } else {
        auto &bin = it->second;
        bin.count++;
        bin.count_pos += val;
        bin.count_neg += (1 - val);
        // update_event_rate removed
      }

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
    bins_.reserve(bin_map.size());
    for (auto &kv : bin_map) {
      bins_.push_back(std::move(kv.second));
    }

    // Initialize IV cache if needed
    iv_cache_ = std::make_unique<IVCache>(bins_.size(), bins_.size() > 10);
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

    // Efficiently merge rare categories
    std::vector<CategoricalBin> new_bins;
    new_bins.reserve(bins_.size());

    CategoricalBin others;

    for (auto &b : bins_) {
      if (b.count >= cutoff_count ||
          static_cast<int>(new_bins.size()) < min_bins_) {
        new_bins.push_back(std::move(b));
      } else {
        others.merge_with(b);
      }
    }

    if (others.count > 0) {
      // Add "Others" category only if there were no categories
      if (others.categories.empty()) {
        others.categories.push_back("Others");
      }
      new_bins.push_back(std::move(others));
    }

    bins_ = std::move(new_bins);
    iv_cache_->resize(bins_.size());
  }

  // Enhanced monotonic ordering with stability improvements
  void ensure_monotonic_order() {
    // Calculate metrics before sorting
    compute_woe_iv(bins_);

    // Sort by WoE
    std::sort(bins_.begin(), bins_.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.woe < b.woe;
              });
  }

  // Enhanced bin merging in temporary vector with Bayesian metrics
  void merge_bins_in_temp(std::vector<CategoricalBin> &temp, size_t idx1,
                          size_t idx2) {
    if (idx1 >= temp.size() || idx2 >= temp.size() || idx1 == idx2)
      return;
    if (idx2 < idx1)
      std::swap(idx1, idx2);

    temp[idx1].merge_with(temp[idx2]);
    temp.erase(temp.begin() + idx2);
    compute_woe_iv(temp);
  }

  // Enhanced bin merging in main vector with cache updates
  void merge_bins(size_t idx1, size_t idx2) {
    if (idx1 >= bins_.size() || idx2 >= bins_.size() || idx1 == idx2)
      return;
    if (idx2 < idx1)
      std::swap(idx1, idx2);

    bins_[idx1].merge_with(bins_[idx2]);
    bins_.erase(bins_.begin() + idx2);
    compute_woe_iv(bins_);

    // Update IV cache
    iv_cache_->invalidate_bin(idx1);
    iv_cache_->resize(bins_.size());
  }

  // Enhanced pre-bin merging with improved IV consideration
  void merge_adjacent_bins_for_prebins() {
    while (bins_.size() > static_cast<size_t>(max_n_prebins_) &&
           bins_.size() > static_cast<size_t>(min_bins_)) {

      double original_iv = calculate_iv(bins_);
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t merge_index = 0;

      // Find pair with minimum IV loss
      for (size_t i = 0; i < bins_.size() - 1; ++i) {
        // Check cache first
        double cached_iv = iv_cache_->get(i, i + 1);
        double iv_loss;

        if (cached_iv >= 0.0) {
          iv_loss = original_iv - cached_iv;
        } else {
          // Calculate if not in cache
          std::vector<CategoricalBin> temp = bins_;
          merge_bins_in_temp(temp, i, i + 1);
          double new_iv = calculate_iv(temp);
          iv_cache_->set(i, i + 1, new_iv);
          iv_loss = original_iv - new_iv;
        }

        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_index = i;
        }
      }

      merge_bins(merge_index, merge_index + 1);
    }
  }

  // Enhanced monotonicity improvement with smarter bin selection
  void improve_monotonicity_step() {
    // Calculate monotonicity violations and their severity
    std::vector<std::pair<size_t, double>> violations;

    for (size_t i = 1; i + 1 < bins_.size(); ++i) {
      // Check for non-monotonic patterns (peaks or valleys)
      bool is_peak = bins_[i].woe > bins_[i - 1].woe + EPS &&
                     bins_[i].woe > bins_[i + 1].woe + EPS;
      bool is_valley = bins_[i].woe < bins_[i - 1].woe - EPS &&
                       bins_[i].woe < bins_[i + 1].woe - EPS;

      if (is_peak || is_valley) {
        // Calculate violation severity
        double severity = std::max(std::abs(bins_[i].woe - bins_[i - 1].woe),
                                   std::abs(bins_[i].woe - bins_[i + 1].woe));
        violations.push_back({i, severity});
      }
    }

    // If no violations, return
    if (violations.empty())
      return;

    // Sort by severity (largest first)
    std::sort(violations.begin(), violations.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });

    // Fix the worst violation
    size_t i = violations[0].first;

    double orig_iv = calculate_iv(bins_);

    // Check cache for merge i-1,i
    double iv_merge_1;
    double cached_iv_1 = iv_cache_->get(i - 1, i);

    if (cached_iv_1 >= 0.0) {
      iv_merge_1 = cached_iv_1;
    } else {
      std::vector<CategoricalBin> temp1 = bins_;
      merge_bins_in_temp(temp1, i - 1, i);
      iv_merge_1 = calculate_iv(temp1);
      iv_cache_->set(i - 1, i, iv_merge_1);
    }

    // Check cache for merge i,i+1
    double iv_merge_2;
    double cached_iv_2 = iv_cache_->get(i, i + 1);

    if (cached_iv_2 >= 0.0) {
      iv_merge_2 = cached_iv_2;
    } else {
      std::vector<CategoricalBin> temp2 = bins_;
      merge_bins_in_temp(temp2, i, i + 1);
      iv_merge_2 = calculate_iv(temp2);
      iv_cache_->set(i, i + 1, iv_merge_2);
    }

    // Calculate information loss for each merge option
    double loss1 = orig_iv - iv_merge_1;
    double loss2 = orig_iv - iv_merge_2;

    // Choose merge with smallest information loss
    if (loss1 < loss2) {
      merge_bins(i - 1, i);
    } else {
      merge_bins(i, i + 1);
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

        // Found a valid solution, check if it's the best
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
    while (static_cast<int>(bins_.size()) > max_bins_) {
      if (bins_.size() <= 1)
        break;

      double orig_iv = calculate_iv(bins_);
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t merge_index = 0;

      // Find pair with minimum IV loss to merge
      for (size_t i = 0; i < bins_.size() - 1; ++i) {
        double cached_iv = iv_cache_->get(i, i + 1);
        double iv_loss;

        if (cached_iv >= 0.0) {
          iv_loss = orig_iv - cached_iv;
        } else {
          std::vector<CategoricalBin> temp = bins_;
          merge_bins_in_temp(temp, i, i + 1);
          double new_iv = calculate_iv(temp);
          iv_cache_->set(i, i + 1, new_iv);
          iv_loss = orig_iv - new_iv;
        }

        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          merge_index = i;
        }
      }

      merge_bins(merge_index, merge_index + 1);
    }

    ensure_monotonic_order();
    compute_woe_iv(bins_);
  }

public:
  OBC_JEDI(const std::vector<std::string> &feature,
           const std::vector<int> &target, int min_bins, int max_bins,
           double bin_cutoff, int max_n_prebins, std::string bin_separator,
           double convergence_threshold, int max_iterations)
      : feature_(feature), target_(target), min_bins_(min_bins),
        max_bins_(max_bins), bin_cutoff_(bin_cutoff),
        max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
        convergence_threshold_(convergence_threshold),
        max_iterations_(max_iterations), bins_(), total_pos_(0), total_neg_(0),
        converged_(false), iterations_run_(0) {
    validate_inputs();

    // Adjust parameters based on unique category count
    std::unordered_set<std::string> unique_cats(feature_.begin(),
                                                feature_.end());
    int ncat = static_cast<int>(unique_cats.size());

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
    // Special case: few categories
    std::unordered_set<std::string> unique_cats(feature_.begin(),
                                                feature_.end());
    int ncat = static_cast<int>(unique_cats.size());

    if (ncat <= 2) {
      // Trivial case: <=2 categories
      int total_pos = 0, total_neg = 0;
      std::unordered_map<std::string, CategoricalBin> bin_map;

      for (size_t i = 0; i < feature_.size(); ++i) {
        auto &bin = bin_map[feature_[i]];

        if (bin.categories.empty()) {
          bin.categories.push_back(feature_[i]);
        }

        bin.count++;
        bin.count_pos += target_[i];
        bin.count_neg += (1 - target_[i]);
        // update_event_rate removed

        total_pos += target_[i];
        total_neg += (1 - target_[i]);
      }

      bins_.clear();
      bins_.reserve(bin_map.size());

      for (auto &kv : bin_map) {
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
    // Efficient result preparation
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
      ids[i] = i + 1;

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

  // Enhanced R to C++ conversion with better NA handling
  std::vector<std::string> feature_vec;
  std::vector<int> target_vec;

  feature_vec.reserve(feature.size());
  target_vec.reserve(target.size());

  int na_feature_count = 0;
  int na_target_count = 0;

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    // NA handling in feature
    if (feature[i] == NA_STRING) {
      feature_vec.push_back("NA");
      na_feature_count++;
    } else {
      feature_vec.push_back(as<std::string>(feature[i]));
    }

    // NA handling in target
    if (IntegerVector::is_na(target[i])) {
      na_target_count++;
      stop("Target cannot contain NA values at position %d", i + 1);
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
    OBC_JEDI jedi(feature_vec, target_vec, min_bins, max_bins, bin_cutoff,
                  max_n_prebins, bin_separator, convergence_threshold,
                  max_iterations);
    jedi.fit();
    return jedi.get_results();
  } catch (const std::exception &e) {
    Rcpp::stop("Error in optimal_binning_categorical_jedi: %s", e.what());
  }
}