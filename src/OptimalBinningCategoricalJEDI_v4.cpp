// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <memory>

using namespace Rcpp;

// Global constants for better precision and readability
static constexpr double EPS = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
static constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

// Optimized structure for categorical bins with enhanced statistics
struct CatBinInfo {
  std::vector<std::string> categories;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  double event_rate; // Cache for event rate
  
  CatBinInfo() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0), event_rate(0.0) {
    categories.reserve(8); // Pre-allocate to avoid frequent reallocations
  }
  
  // Optimized method to update statistics in a single pass
  inline void add_category(const std::string& cat, int val) {
    categories.push_back(cat);
    count++;
    count_pos += val;
    count_neg += (1 - val);
    update_event_rate();
  }
  
  // Optimized method to merge bins
  inline void merge_with(const CatBinInfo& other) {
    // Pre-allocate to avoid multiple reallocations
    categories.reserve(categories.size() + other.categories.size());
    categories.insert(categories.end(), other.categories.begin(), other.categories.end());
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    update_event_rate();
  }
  
  // Method to update event rate
  inline void update_event_rate() {
    event_rate = count > 0 ? static_cast<double>(count_pos) / count : 0.0;
  }
  
  // Enhanced method to calculate WoE and IV with Bayesian smoothing
  inline void compute_metrics(int total_pos, int total_neg) {
    // Calculate Bayesian prior based on overall prevalence
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
      (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    // Apply Bayesian smoothing to rates
    double pos_rate = static_cast<double>(count_pos + prior_pos) / 
      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(count_neg + prior_neg) / 
      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    // Calculate WoE and IV with improved numerical stability
    woe = std::log(pos_rate / neg_rate);
    iv = (pos_rate - neg_rate) * woe;
    
    // Handle non-finite values
    if (!std::isfinite(woe)) woe = 0.0;
    if (!std::isfinite(iv)) iv = 0.0;
  }
};

// Enhanced cache for frequent IV calculations
class IVCache {
private:
  std::vector<std::vector<double>> cache;
  bool enabled;
  
public:
  explicit IVCache(size_t max_size, bool use_cache = true) : enabled(use_cache) {
    if (enabled) {
      cache.resize(max_size);
      for (auto& row : cache) {
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
    if (!enabled) return;
    cache.resize(new_size);
    for (auto& row : cache) {
      row.resize(new_size, -1.0);
    }
  }
};

// Enhanced main class with improved optimization strategies
class OptimalBinningCategoricalJEDI {
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
  
  std::vector<CatBinInfo> bins_;
  std::unique_ptr<IVCache> iv_cache_;
  int total_pos_;
  int total_neg_;
  bool converged_;
  int iterations_run_;
  
  // Enhanced input validation with comprehensive checks
  void validate_inputs() {
    if (feature_.size() != target_.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }
    if (feature_.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty");
    }
    
    // Check for empty strings in feature
    if (std::any_of(feature_.begin(), feature_.end(), [](const std::string& s) { 
      return s.empty(); 
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }
    
    // Fast binary value check
    bool has_zero = false;
    bool has_one = false;
    
    for (int t : target_) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must be binary (0/1)");
      
      // Optimization: Stop once we've confirmed both values exist
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
  }
  
  // Enhanced WoE calculation with Bayesian smoothing
  inline double calculate_woe(int pos, int neg) const {
    // Calculate Bayesian prior based on overall prevalence
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos_) / 
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
  double calculate_iv(const std::vector<CatBinInfo>& current_bins) const {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      // Calculate Bayesian prior based on overall prevalence
      double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos_) / 
        (total_pos_ + total_neg_);
      double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
      
      // Apply Bayesian smoothing to rates
      double pos_rate = static_cast<double>(bin.count_pos + prior_pos) / 
        static_cast<double>(total_pos_ + BAYESIAN_PRIOR_STRENGTH);
      double neg_rate = static_cast<double>(bin.count_neg + prior_neg) / 
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
  void compute_woe_iv(std::vector<CatBinInfo>& current_bins) {
    for (auto& bin : current_bins) {
      bin.compute_metrics(total_pos_, total_neg_);
    }
  }
  
  // Enhanced monotonicity check with adaptive threshold
  bool is_monotonic(const std::vector<CatBinInfo>& current_bins) const {
    if (current_bins.size() <= 2) return true;
    
    // Calculate average WoE gap for context-aware check
    double total_gap = 0.0;
    for (size_t i = 1; i < current_bins.size(); ++i) {
      total_gap += std::abs(current_bins[i].woe - current_bins[i-1].woe);
    }
    
    double avg_gap = total_gap / (current_bins.size() - 1);
    
    // Adaptive threshold based on average gap
    double monotonicity_threshold = std::min(EPS, avg_gap * 0.01);
    
    // Check the direction in the first two bins
    bool should_increase = current_bins[1].woe >= current_bins[0].woe - monotonicity_threshold;
    
    for (size_t i = 2; i < current_bins.size(); ++i) {
      if (should_increase && current_bins[i].woe < current_bins[i-1].woe - monotonicity_threshold) {
        return false;
      }
      if (!should_increase && current_bins[i].woe > current_bins[i-1].woe + monotonicity_threshold) {
        return false;
      }
    }
    return true;
  }
  
  // Optimized category name joining with pre-allocation
  static std::string join_categories(const std::vector<std::string>& cats, const std::string& sep) {
    if (cats.empty()) return "";
    if (cats.size() == 1) return cats[0];
    
    // Estimate total size for pre-allocation
    size_t total_size = 0;
    for (const auto& cat : cats) {
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
    std::unordered_map<std::string, CatBinInfo> bin_map;
    bin_map.reserve(est_cats);
    
    total_pos_ = 0;
    total_neg_ = 0;
    
    // Count in a single pass
    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string& cat = feature_[i];
      int val = target_[i];
      
      auto it = bin_map.find(cat);
      if (it == bin_map.end()) {
        auto& bin = bin_map[cat];
        bin.add_category(cat, val);
      } else {
        auto& bin = it->second;
        bin.count++;
        bin.count_pos += val;
        bin.count_neg += (1 - val);
        bin.update_event_rate();
      }
      
      total_pos_ += val;
      total_neg_ += (1 - val);
    }
    
    // Check for extremely imbalanced datasets
    if (total_pos_ < 5 || total_neg_ < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    // Transfer to final vector
    bins_.clear();
    bins_.reserve(bin_map.size());
    for (auto& kv : bin_map) {
      bins_.push_back(std::move(kv.second));
    }
    
    // Initialize IV cache if needed
    iv_cache_ = std::make_unique<IVCache>(bins_.size(), bins_.size() > 10);
  }
  
  // Enhanced rare category merging with improved handling
  void merge_low_freq() {
    int total_count = 0;
    for (auto& b : bins_) {
      total_count += b.count;
    }
    double cutoff_count = total_count * bin_cutoff_;
    
    // Sort by frequency (rarest first)
    std::sort(bins_.begin(), bins_.end(), [](const CatBinInfo& a, const CatBinInfo& b) {
      return a.count < b.count;
    });
    
    // Efficiently merge rare categories
    std::vector<CatBinInfo> new_bins;
    new_bins.reserve(bins_.size());
    
    CatBinInfo others;
    
    for (auto& b : bins_) {
      if (b.count >= cutoff_count || static_cast<int>(new_bins.size()) < min_bins_) {
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
    std::sort(bins_.begin(), bins_.end(), [](const CatBinInfo& a, const CatBinInfo& b) {
      return a.woe < b.woe;
    });
  }
  
  // Enhanced bin merging in temporary vector with Bayesian metrics
  void merge_bins_in_temp(std::vector<CatBinInfo>& temp, size_t idx1, size_t idx2) {
    if (idx1 >= temp.size() || idx2 >= temp.size() || idx1 == idx2) return;
    if (idx2 < idx1) std::swap(idx1, idx2);
    
    temp[idx1].merge_with(temp[idx2]);
    temp.erase(temp.begin() + idx2);
    compute_woe_iv(temp);
  }
  
  // Enhanced bin merging in main vector with cache updates
  void merge_bins(size_t idx1, size_t idx2) {
    if (idx1 >= bins_.size() || idx2 >= bins_.size() || idx1 == idx2) return;
    if (idx2 < idx1) std::swap(idx1, idx2);
    
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
          std::vector<CatBinInfo> temp = bins_;
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
      bool is_peak = bins_[i].woe > bins_[i-1].woe + EPS && bins_[i].woe > bins_[i+1].woe + EPS;
      bool is_valley = bins_[i].woe < bins_[i-1].woe - EPS && bins_[i].woe < bins_[i+1].woe - EPS;
      
      if (is_peak || is_valley) {
        // Calculate violation severity
        double severity = std::max(
          std::abs(bins_[i].woe - bins_[i-1].woe),
          std::abs(bins_[i].woe - bins_[i+1].woe)
        );
        violations.push_back({i, severity});
      }
    }
    
    // If no violations, return
    if (violations.empty()) return;
    
    // Sort by severity (largest first)
    std::sort(violations.begin(), violations.end(), 
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Fix the worst violation
    size_t i = violations[0].first;
    
    double orig_iv = calculate_iv(bins_);
    
    // Check cache for merge i-1,i
    double iv_merge_1;
    double cached_iv_1 = iv_cache_->get(i-1, i);
    
    if (cached_iv_1 >= 0.0) {
      iv_merge_1 = cached_iv_1;
    } else {
      std::vector<CatBinInfo> temp1 = bins_;
      merge_bins_in_temp(temp1, i-1, i);
      iv_merge_1 = calculate_iv(temp1);
      iv_cache_->set(i-1, i, iv_merge_1);
    }
    
    // Check cache for merge i,i+1
    double iv_merge_2;
    double cached_iv_2 = iv_cache_->get(i, i+1);
    
    if (cached_iv_2 >= 0.0) {
      iv_merge_2 = cached_iv_2;
    } else {
      std::vector<CatBinInfo> temp2 = bins_;
      merge_bins_in_temp(temp2, i, i+1);
      iv_merge_2 = calculate_iv(temp2);
      iv_cache_->set(i, i+1, iv_merge_2);
    }
    
    // Calculate information loss for each merge option
    double loss1 = orig_iv - iv_merge_1;
    double loss2 = orig_iv - iv_merge_2;
    
    // Choose merge with smallest information loss
    if (loss1 < loss2) {
      merge_bins(i-1, i);
    } else {
      merge_bins(i, i+1);
    }
  }
  
  // Enhanced main optimization algorithm with better convergence properties
  void optimize() {
    double prev_iv = calculate_iv(bins_);
    converged_ = false;
    iterations_run_ = 0;
    
    // Track best solution seen so far
    double best_iv = prev_iv;
    std::vector<CatBinInfo> best_bins = bins_;
    
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
      if (bins_.size() <= 1) break;
      
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
          std::vector<CatBinInfo> temp = bins_;
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
  OptimalBinningCategoricalJEDI(
    const std::vector<std::string>& feature,
    const std::vector<int>& target,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins,
    std::string bin_separator,
    double convergence_threshold,
    int max_iterations
  ) : feature_(feature), target_(target),
  min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
  max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
  convergence_threshold_(convergence_threshold), max_iterations_(max_iterations),
  bins_(), total_pos_(0), total_neg_(0), converged_(false), iterations_run_(0)
  {
    validate_inputs();
    
    // Adjust parameters based on unique category count
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
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
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    if (ncat <= 2) {
      // Trivial case: <=2 categories
      int total_pos = 0, total_neg = 0;
      std::unordered_map<std::string, CatBinInfo> bin_map;
      
      for (size_t i = 0; i < feature_.size(); ++i) {
        auto& bin = bin_map[feature_[i]];
        
        if (bin.categories.empty()) {
          bin.categories.push_back(feature_[i]);
        }
        
        bin.count++;
        bin.count_pos += target_[i];
        bin.count_neg += (1 - target_[i]);
        bin.update_event_rate();
        
        total_pos += target_[i];
        total_neg += (1 - target_[i]);
      }
      
      bins_.clear();
      bins_.reserve(bin_map.size());
      
      for (auto& kv : bin_map) {
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
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg,
      Named("total_iv") = total_iv,
      Named("converged") = converged_,
      Named("iterations") = iterations_run_
    );
  }
};


//' @title Optimal Categorical Binning JEDI (Joint Entropy-Driven Information Maximization)
//'
//' @description
//' A robust categorical binning algorithm that optimizes Information Value (IV) while maintaining
//' monotonic Weight of Evidence (WoE) relationships. This implementation employs Bayesian smoothing,
//' adaptive monotonicity enforcement, and sophisticated information-theoretic optimization to create
//' statistically stable and interpretable bins.
//'
//' @details
//' The algorithm employs a multi-phase optimization approach based on information theory principles:
//'
//' \strong{Mathematical Framework:}
//' 
//' For a bin i, the Weight of Evidence (WoE) is calculated with Bayesian smoothing as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{p_i^*}{n_i^*}\right)}
//' 
//' where:
//' \itemize{
//'   \item \eqn{p_i^* = \frac{n_i^+ + \alpha \cdot \pi}{N^+ + \alpha}} is the smoothed proportion of positive cases
//'   \item \eqn{n_i^* = \frac{n_i^- + \alpha \cdot (1-\pi)}{N^- + \alpha}} is the smoothed proportion of negative cases
//'   \item \eqn{\pi = \frac{N^+}{N^+ + N^-}} is the overall positive rate
//'   \item \eqn{\alpha} is the prior strength parameter (default: 0.5)
//'   \item \eqn{n_i^+} is the count of positive cases in bin i
//'   \item \eqn{n_i^-} is the count of negative cases in bin i
//'   \item \eqn{N^+} is the total number of positive cases
//'   \item \eqn{N^-} is the total number of negative cases
//' }
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV_i = (p_i^* - n_i^*) \times WoE_i}
//'
//' And the total IV is:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//'
//' \strong{Algorithm Phases:}
//' \enumerate{
//'   \item \strong{Initial Binning:} Creates individual bins for unique categories with comprehensive statistics
//'   \item \strong{Low-Frequency Treatment:} Combines rare categories (< bin_cutoff) to ensure statistical stability
//'   \item \strong{Optimization:} Iteratively merges bins using adaptive IV loss minimization while ensuring WoE monotonicity
//'   \item \strong{Final Adjustment:} Ensures bin count constraints (min_bins <= bins <= max_bins) when feasible
//' }
//'
//' \strong{Key Features:}
//' \itemize{
//'   \item Bayesian smoothing for robust WoE estimation with small samples
//'   \item Adaptive monotonicity enforcement with violation severity prioritization
//'   \item Information-theoretic merging strategy that minimizes information loss
//'   \item Handling of edge cases including imbalanced datasets and sparse categories
//'   \item Best-solution tracking to ensure optimal results even with early convergence
//' }
//'
//' @param target Integer binary vector (0 or 1) representing the response variable
//' @param feature Character vector of categorical predictor values
//' @param min_bins Minimum number of output bins (default: 3). Adjusted if unique categories < min_bins
//' @param max_bins Maximum number of output bins (default: 5). Must be >= min_bins
//' @param bin_cutoff Minimum relative frequency threshold for individual bins (default: 0.05)
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20)
//' @param bin_separator Delimiter for names of combined categories (default: "%;%")
//' @param convergence_threshold IV difference threshold for convergence (default: 1e-6)
//' @param max_iterations Maximum number of optimization iterations (default: 1000)
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Numeric vector with bin identifiers
//'   \item bin: Character vector with bin names (concatenated categories)
//'   \item woe: Numeric vector with Weight of Evidence values
//'   \item iv: Numeric vector with Information Value per bin
//'   \item count: Integer vector with observation counts per bin
//'   \item count_pos: Integer vector with positive class counts per bin
//'   \item count_neg: Integer vector with negative class counts per bin
//'   \item total_iv: Total Information Value of the binning
//'   \item converged: Logical indicating whether the algorithm converged
//'   \item iterations: Integer count of optimization iterations performed
//' }
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone Optimal Binning Algorithm for Credit Risk Modeling. Working Paper.
//'   \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and its Applications. SIAM.
//'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//'   \item García-Magariño, I., Medrano, C., Lombas, A. S., & Barrasa, A. (2019). A hybrid approach with agent-based simulation and clustering for sociograms. Information Sciences, 499, 47-61.
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
//' }
//'
//' @examples
//' \dontrun{
//' # Basic usage
//' result <- optimal_binning_categorical_jedi(
//'   target = c(1,0,1,1,0),
//'   feature = c("A","B","A","C","B"),
//'   min_bins = 2,
//'   max_bins = 3
//' )
//'
//' # Rare category handling
//' result <- optimal_binning_categorical_jedi(
//'   target = target_vector,
//'   feature = feature_vector,
//'   bin_cutoff = 0.03,  # More aggressive rare category treatment
//'   max_n_prebins = 15  # Limit on initial bins
//' )
//'
//' # Working with more complex settings
//' result <- optimal_binning_categorical_jedi(
//'   target = target_vector,
//'   feature = feature_vector,
//'   min_bins = 3,
//'   max_bins = 10,
//'   bin_cutoff = 0.01,
//'   convergence_threshold = 1e-8,  # Stricter convergence
//'   max_iterations = 2000  # More iterations for complex problems
//' )
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi(
   Rcpp::IntegerVector target,
   Rcpp::StringVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
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
     stop("Target cannot contain NA values at position %d", i+1);
   } else {
     target_vec.push_back(target[i]);
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
 try {
   OptimalBinningCategoricalJEDI jedi(
       feature_vec, target_vec, min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations
   );
   jedi.fit();
   return jedi.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal_binning_categorical_jedi: %s", e.what());
 }
}

