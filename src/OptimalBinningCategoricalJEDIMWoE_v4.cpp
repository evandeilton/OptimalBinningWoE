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
#include <numeric>

using namespace Rcpp;

// Global constants for better readability and consistency
static constexpr double EPSILON = 1e-10;
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
static constexpr const char* MISSING_VALUE = "__MISSING__";  // Special category for missing values
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Join vector of strings ensuring uniqueness
inline std::string join_categories(const std::vector<std::string>& categories, 
                                   const std::string& separator) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Create a set for uniqueness check
  std::unordered_set<std::string> unique_cats;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_cats.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  // Estimate result size for pre-allocation
  size_t total_length = 0;
  for (const auto& cat : unique_vec) {
    total_length += cat.length();
  }
  total_length += separator.length() * (unique_vec.size() - 1);
  
  // Build result string
  std::string result;
  result.reserve(total_length);
  
  result = unique_vec[0];
  for (size_t i = 1; i < unique_vec.size(); ++i) {
    result += separator;
    result += unique_vec[i];
  }
  
  return result;
}

// Calculate Multinomial Weight of Evidence with Laplace smoothing
inline double calculate_mwoe(int class_count, int total_class_count,
                             const std::vector<int>& other_counts,
                             const std::vector<int>& total_other_counts,
                             double alpha = LAPLACE_ALPHA) {
  // Apply Laplace smoothing for current class
  double class_rate = (class_count + alpha) / (total_class_count + alpha * 2);
  
  // Calculate combined rate for all other classes with smoothing
  int other_total = std::accumulate(other_counts.begin(), other_counts.end(), 0);
  int total_other = std::accumulate(total_other_counts.begin(), total_other_counts.end(), 0);
  
  double other_rate = (other_total + alpha) / (total_other + alpha * 2);
  
  // Calculate M-WoE
  return safe_log(class_rate / other_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(int class_count, int total_class_count,
                           const std::vector<int>& other_counts,
                           const std::vector<int>& total_other_counts,
                           double alpha = LAPLACE_ALPHA) {
  // Apply Laplace smoothing
  double class_rate = (class_count + alpha) / (total_class_count + alpha * 2);
  
  // Calculate combined rate for all other classes with smoothing
  int other_total = std::accumulate(other_counts.begin(), other_counts.end(), 0);
  int total_other = std::accumulate(total_other_counts.begin(), total_other_counts.end(), 0);
  
  double other_rate = (other_total + alpha) / (total_other + alpha * 2);
  
  // Calculate M-WoE and IV
  double woe = safe_log(class_rate / other_rate);
  return (class_rate - other_rate) * woe;
}

// Calculate Jensen-Shannon divergence between bins for multiclass case
inline double calculate_divergence(const std::vector<int>& bin1_counts, 
                                   const std::vector<int>& bin2_counts,
                                   const std::vector<int>& total_counts) {
  // Preallocate vectors
  std::vector<double> p1(bin1_counts.size()), p2(bin2_counts.size()), m(bin1_counts.size());
  
  // Calculate bin total counts
  int bin1_total = std::accumulate(bin1_counts.begin(), bin1_counts.end(), 0);
  int bin2_total = std::accumulate(bin2_counts.begin(), bin2_counts.end(), 0);
  
  // Calculate smoothed proportions for each bin
  for (size_t i = 0; i < bin1_counts.size(); ++i) {
    p1[i] = (bin1_counts[i] + LAPLACE_ALPHA) / (bin1_total + LAPLACE_ALPHA * bin1_counts.size());
    p2[i] = (bin2_counts[i] + LAPLACE_ALPHA) / (bin2_total + LAPLACE_ALPHA * bin2_counts.size());
    m[i] = (p1[i] + p2[i]) / 2.0;
  }
  
  // Calculate Jensen-Shannon divergence (symmetric KL divergence)
  double div = 0.0;
  for (size_t i = 0; i < bin1_counts.size(); ++i) {
    if (p1[i] > EPSILON) {
      div += 0.5 * p1[i] * safe_log(p1[i] / m[i]);
    }
    if (p2[i] > EPSILON) {
      div += 0.5 * p2[i] * safe_log(p2[i] / m[i]);
    }
  }
  
  return div;
}
}

// Enhanced cache for M-WoE and IV values
class MWoECache {
private:
  std::vector<std::vector<std::vector<double>>> bin_pair_iv_cache;
  std::vector<std::vector<double>> bin_pair_divergence_cache;
  size_t n_classes;
  bool enabled;
  
public:
  MWoECache(size_t max_bins, size_t num_classes, bool use_cache = true) 
    : n_classes(num_classes), enabled(use_cache && max_bins > 0) {
    if (enabled) {
      // IV cache: three dimensions: bin1 x bin2 x class
      bin_pair_iv_cache.resize(max_bins);
      for (auto& matrix : bin_pair_iv_cache) {
        matrix.resize(max_bins);
        for (auto& row : matrix) {
          row.resize(n_classes, -1.0);
        }
      }
      
      // Divergence cache: two dimensions: bin1 x bin2
      bin_pair_divergence_cache.resize(max_bins);
      for (auto& row : bin_pair_divergence_cache) {
        row.resize(max_bins, -1.0);
      }
    }
  }
  
  // Retrieve cached class IV
  inline double get_class_iv(size_t bin1, size_t bin2, size_t class_idx) {
    if (!enabled || bin1 >= bin_pair_iv_cache.size() || bin2 >= bin_pair_iv_cache[bin1].size()) {
      return -1.0;
    }
    return bin_pair_iv_cache[bin1][bin2][class_idx];
  }
  
  // Store class IV in cache
  inline void set_class_iv(size_t bin1, size_t bin2, size_t class_idx, double value) {
    if (!enabled || bin1 >= bin_pair_iv_cache.size() || bin2 >= bin_pair_iv_cache[bin1].size()) {
      return;
    }
    bin_pair_iv_cache[bin1][bin2][class_idx] = value;
  }
  
  // Retrieve cached divergence
  inline double get_divergence(size_t bin1, size_t bin2) {
    if (!enabled || bin1 >= bin_pair_divergence_cache.size() || bin2 >= bin_pair_divergence_cache[bin1].size()) {
      return -1.0;
    }
    return bin_pair_divergence_cache[bin1][bin2];
  }
  
  // Store divergence in cache
  inline void set_divergence(size_t bin1, size_t bin2, double value) {
    if (!enabled || bin1 >= bin_pair_divergence_cache.size() || bin2 >= bin_pair_divergence_cache[bin1].size()) {
      return;
    }
    bin_pair_divergence_cache[bin1][bin2] = value;
  }
  
  // Invalidate cache entries for a specific bin
  inline void invalidate_bin(size_t bin_idx) {
    if (!enabled || bin_idx >= bin_pair_iv_cache.size()) {
      return;
    }
    
    // Clear IV cache
    for (size_t i = 0; i < bin_pair_iv_cache.size(); ++i) {
      if (i < bin_pair_iv_cache[bin_idx].size()) {
        std::fill(bin_pair_iv_cache[bin_idx][i].begin(), bin_pair_iv_cache[bin_idx][i].end(), -1.0);
      }
      if (bin_idx < bin_pair_iv_cache[i].size()) {
        std::fill(bin_pair_iv_cache[i][bin_idx].begin(), bin_pair_iv_cache[i][bin_idx].end(), -1.0);
      }
    }
    
    // Clear divergence cache
    for (size_t i = 0; i < bin_pair_divergence_cache.size(); ++i) {
      if (i < bin_pair_divergence_cache.size()) {
        bin_pair_divergence_cache[bin_idx][i] = -1.0;
      }
      if (bin_idx < bin_pair_divergence_cache[i].size()) {
        bin_pair_divergence_cache[i][bin_idx] = -1.0;
      }
    }
  }
  
  // Resize cache for new number of bins
  inline void resize(size_t new_size) {
    if (!enabled) return;
    
    // Resize IV cache
    bin_pair_iv_cache.resize(new_size);
    for (auto& matrix : bin_pair_iv_cache) {
      matrix.resize(new_size);
      for (auto& row : matrix) {
        row.resize(n_classes, -1.0);
      }
    }
    
    // Resize divergence cache
    bin_pair_divergence_cache.resize(new_size);
    for (auto& row : bin_pair_divergence_cache) {
      row.resize(new_size, -1.0);
    }
  }
};

// Enhanced structure for multinomial bin information
struct MultiCatBinInfo {
  std::unordered_set<std::string> category_set;  // For uniqueness check
  std::vector<std::string> categories;           // For ordered storage
  int total_count;
  std::vector<int> class_counts;
  std::vector<double> woes;
  std::vector<double> ivs;
  std::vector<double> class_rates;  // Cache for class rates
  
  // Default constructor
  MultiCatBinInfo() : total_count(0) {}
  
  // Constructor with number of classes
  MultiCatBinInfo(size_t n_classes) 
    : total_count(0), 
      class_counts(n_classes, 0),
      woes(n_classes, 0.0),
      ivs(n_classes, 0.0),
      class_rates(n_classes, 0.0) {
        categories.reserve(8);  // Pre-allocate for typical categories
  }
  
  // Add a category ensuring uniqueness
  inline void add_category(const std::string& cat) {
    if (category_set.insert(cat).second) {  // Only add if not already present
      categories.push_back(cat);
    }
  }
  
  // Add a category with its class
  inline void add_instance(const std::string& cat, int class_idx, size_t n_classes) {
    if (categories.empty()) {
      // If first, initialize vectors
      class_counts.resize(n_classes, 0);
      woes.resize(n_classes, 0.0);
      ivs.resize(n_classes, 0.0);
      class_rates.resize(n_classes, 0.0);
    }
    
    // Add category ensuring uniqueness
    add_category(cat);
    
    // Update counts
    total_count++;
    class_counts[class_idx]++;
    update_class_rates();
  }
  
  // Merge with another bin ensuring uniqueness of categories
  inline void merge_with(const MultiCatBinInfo& other) {
    // Add each category from other bin, ensuring uniqueness
    for (const auto& cat : other.categories) {
      add_category(cat);
    }
    
    // Update counts
    total_count += other.total_count;
    
    for (size_t i = 0; i < class_counts.size(); ++i) {
      class_counts[i] += other.class_counts[i];
    }
    
    update_class_rates();
  }
  
  // Update cached class rates
  inline void update_class_rates() {
    if (total_count > 0) {
      for (size_t i = 0; i < class_counts.size(); ++i) {
        class_rates[i] = static_cast<double>(class_counts[i]) / total_count;
      }
    }
  }
  
  // Compute M-WoE and IV metrics with Laplace smoothing
  inline void compute_metrics(const std::vector<int>& total_class_counts) {
    // Pre-allocate vectors for better performance
    std::vector<int> other_counts(class_counts.size() - 1);
    std::vector<int> total_other_counts(class_counts.size() - 1);
    
    for (size_t class_idx = 0; class_idx < class_counts.size(); ++class_idx) {
      // Extract counts for other classes
      size_t other_idx = 0;
      for (size_t i = 0; i < class_counts.size(); ++i) {
        if (i != class_idx) {
          other_counts[other_idx] = class_counts[i];
          total_other_counts[other_idx] = total_class_counts[i];
          other_idx++;
        }
      }
      
      // Calculate M-WoE and IV using utility functions with Laplace smoothing
      woes[class_idx] = utils::calculate_mwoe(
        class_counts[class_idx], total_class_counts[class_idx],
                                                   other_counts, total_other_counts, LAPLACE_ALPHA);
      
      ivs[class_idx] = utils::calculate_iv(
        class_counts[class_idx], total_class_counts[class_idx],
                                                   other_counts, total_other_counts, LAPLACE_ALPHA);
    }
  }
  
  // Calculate statistical divergence from another bin
  inline double divergence_from(const MultiCatBinInfo& other, 
                                const std::vector<int>& total_class_counts) const {
    return utils::calculate_divergence(class_counts, other.class_counts, total_class_counts);
  }
};

// Main class for multinomial categorical binning
class OptimalBinningCategoricalJEDIMWoE {
private:
  std::vector<std::string> feature_;
  std::vector<int> target_;
  size_t n_classes_;
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  std::vector<MultiCatBinInfo> bins_;
  std::vector<int> total_class_counts_;
  std::unique_ptr<MWoECache> mwoe_cache_;
  bool converged_;
  int iterations_run_;
  bool use_divergence_;  // Flag to toggle between IV and divergence-based merging
  
  // Advanced input validation
  void validate_inputs() {
    if (feature_.empty() || feature_.size() != target_.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same non-empty length");
    }
    
    // Find max class value and validate class values
    int max_class = -1;
    std::unordered_set<int> class_set;
    
    for (int t : target_) {
      if (t < 0) {
        throw std::invalid_argument("Target values must be non-negative integers");
      }
      
      max_class = std::max(max_class, t);
      class_set.insert(t);
    }
    
    // Check for consecutive classes starting from 0
    n_classes_ = max_class + 1;
    
    if (class_set.size() < 2) {
      throw std::invalid_argument("Target must have at least 2 distinct classes");
    }
    
    // Ensure all classes from 0 to max_class are present
    for (int i = 0; i < static_cast<int>(n_classes_); ++i) {
      if (class_set.find(i) == class_set.end()) {
        throw std::invalid_argument("Target classes must be consecutive integers starting from 0");
      }
    }
    
    // Validate parameters
    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1");
    }
    if (max_bins_ < min_bins_) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive)");
    }
    if (max_n_prebins_ < min_bins_) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins");
    }
  }
  
  // Optimized initial binning
  void initial_binning() {
    // Estimate number of unique categories
    size_t est_cats = std::min(feature_.size() / 4, static_cast<size_t>(1024));
    std::unordered_map<std::string, MultiCatBinInfo> bin_map;
    bin_map.reserve(est_cats);
    
    // Initialize class counts
    total_class_counts_ = std::vector<int>(n_classes_, 0);
    
    // Process in a single pass
    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string& cat = feature_[i];
      int class_idx = target_[i];
      
      // Initialize bin if needed
      if (bin_map.find(cat) == bin_map.end()) {
        bin_map[cat] = MultiCatBinInfo(n_classes_);
      }
      
      // Update counts
      bin_map[cat].add_instance(cat, class_idx, n_classes_);
      total_class_counts_[class_idx]++;
    }
    
    // Transfer to bins vector
    bins_.clear();
    bins_.reserve(bin_map.size());
    
    for (auto& kv : bin_map) {
      bins_.push_back(std::move(kv.second));
    }
    
    // Initialize M-WoE cache
    mwoe_cache_ = std::make_unique<MWoECache>(bins_.size(), n_classes_, bins_.size() > 10);
  }
  
  // Enhanced merging of low frequency categories
  void merge_low_freq() {
    // Calculate total count
    int total_count = std::accumulate(bins_.begin(), bins_.end(), 0,
                                      [](int sum, const MultiCatBinInfo& bin) {
                                        return sum + bin.total_count;
                                      });
    double cutoff_count = total_count * bin_cutoff_;
    
    // Sort bins by count (ascending)
    std::sort(bins_.begin(), bins_.end(), 
              [](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                return a.total_count < b.total_count;
              });
    
    // Prepare new bins with reserved space
    std::vector<MultiCatBinInfo> new_bins;
    new_bins.reserve(bins_.size());
    
    // Bin for rare categories
    MultiCatBinInfo rare_bin(n_classes_);
    bool has_rare = false;
    
    // Process each bin
    for (auto& bin : bins_) {
      if (bin.total_count >= cutoff_count || static_cast<int>(new_bins.size()) < min_bins_) {
        // Bin with adequate frequency
        new_bins.push_back(std::move(bin));
      } else {
        // Rare bin, merge
        rare_bin.merge_with(bin);
        has_rare = true;
      }
    }
    
    // Add the rare categories bin if it exists
    if (has_rare && rare_bin.total_count > 0) {
      new_bins.push_back(std::move(rare_bin));
    }
    
    bins_ = std::move(new_bins);
    mwoe_cache_->resize(bins_.size());
  }
  
  // Calculate class-specific IV with caching
  double calculate_class_iv(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    double iv = 0.0;
    
    for (const auto& bin : current_bins) {
      // Add contribution from this bin
      iv += bin.ivs[class_idx];
    }
    
    return iv;
  }
  
  // Compute M-WoE and IV for all bins
  void compute_metrics() {
    for (auto& bin : bins_) {
      bin.compute_metrics(total_class_counts_);
    }
  }
  
  // Check monotonicity for a specific class
  bool is_monotonic_for_class(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    if (current_bins.size() <= 2) return true;
    
    // Determine monotonicity direction from first two bins
    bool should_increase = true;
    bool should_decrease = true;
    
    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woes[class_idx] < current_bins[i-1].woes[class_idx] - EPSILON) {
        should_increase = false;
      }
      if (current_bins[i].woes[class_idx] > current_bins[i-1].woes[class_idx] + EPSILON) {
        should_decrease = false;
      }
      
      // If neither pattern holds, not monotonic
      if (!should_increase && !should_decrease) {
        return false;
      }
    }
    
    return true;
  }
  
  // Check monotonicity across all classes
  bool is_monotonic(const std::vector<MultiCatBinInfo>& current_bins) const {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      if (!is_monotonic_for_class(current_bins, class_idx)) {
        return false;
      }
    }
    return true;
  }
  
  // Main optimization algorithm
  void optimize() {
    // Initialize previous IVs
    std::vector<double> prev_ivs(n_classes_);
    for (size_t i = 0; i < n_classes_; ++i) {
      prev_ivs[i] = calculate_class_iv(bins_, i);
    }
    
    converged_ = false;
    iterations_run_ = 0;
    
    // Main optimization loop
    while (iterations_run_ < max_iterations_) {
      // Check stopping criteria
      if (is_monotonic(bins_) && 
          static_cast<int>(bins_.size()) <= max_bins_ && 
          static_cast<int>(bins_.size()) >= min_bins_) {
        converged_ = true;
        break;
      }
      
      // Decide action based on current state
      if (static_cast<int>(bins_.size()) > min_bins_) {
        if (static_cast<int>(bins_.size()) > max_bins_) {
          // Need to reduce number of bins
          if (use_divergence_) {
            merge_most_similar_bins();
          } else {
            merge_adjacent_bins();
          }
          // Toggle strategy
          use_divergence_ = !use_divergence_;
        } else {
          // Need to improve monotonicity
          improve_monotonicity();
        }
      } else {
        // Cannot merge more (reached min_bins)
        break;
      }
      
      // Check convergence
      std::vector<double> current_ivs(n_classes_);
      bool all_converged = true;
      
      for (size_t i = 0; i < n_classes_; ++i) {
        current_ivs[i] = calculate_class_iv(bins_, i);
        if (std::abs(current_ivs[i] - prev_ivs[i]) >= convergence_threshold_) {
          all_converged = false;
        }
      }
      
      if (all_converged) {
        converged_ = true;
        break;
      }
      
      prev_ivs = std::move(current_ivs);
      iterations_run_++;
    }
    
    // Final adjustments to meet max_bins
    while (static_cast<int>(bins_.size()) > max_bins_) {
      merge_adjacent_bins();
    }
    
    // Ensure monotonic ordering
    ensure_monotonic_order();
    compute_metrics();
  }
  
  // Find and merge statistically most similar bins
  void merge_most_similar_bins() {
    double min_divergence = std::numeric_limits<double>::max();
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 0;
    
    // Find pair with minimal statistical divergence
    for (size_t i = 0; i < bins_.size(); ++i) {
      for (size_t j = i + 1; j < bins_.size(); ++j) {
        // Check cache first
        double div = mwoe_cache_->get_divergence(i, j);
        
        if (div < 0.0) {
          // Not in cache, calculate
          div = bins_[i].divergence_from(bins_[j], total_class_counts_);
          mwoe_cache_->set_divergence(i, j, div);
        }
        
        // Prefer adjacent bins if divergence is similar
        if (j == i + 1) {
          div *= 0.95;  // Small bias towards adjacent bins
        }
        
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx1 = i;
          merge_idx2 = j;
        }
      }
    }
    
    // Perform the merge with minimal divergence
    if (merge_idx2 < merge_idx1) std::swap(merge_idx1, merge_idx2);
    merge_bins(merge_idx1, merge_idx2);
  }
  
  // Optimized merging of adjacent bins based on IV loss
  void merge_adjacent_bins() {
    if (bins_.size() <= 2) return;
    
    double min_total_iv_loss = std::numeric_limits<double>::max();
    size_t best_merge_idx = 0;
    
    // Calculate original IVs for each class
    std::vector<double> original_ivs(n_classes_);
    for (size_t i = 0; i < n_classes_; ++i) {
      original_ivs[i] = calculate_class_iv(bins_, i);
    }
    
    // Test each adjacent bin pair
    for (size_t i = 0; i < bins_.size() - 1; ++i) {
      // Check cache
      double total_iv_loss = 0.0;
      bool all_cached = true;
      
      for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
        double cached_iv = mwoe_cache_->get_class_iv(i, i+1, class_idx);
        
        if (cached_iv >= 0.0) {
          total_iv_loss += original_ivs[class_idx] - cached_iv;
        } else {
          all_cached = false;
          break;
        }
      }
      
      // If not cached, calculate
      if (!all_cached) {
        // Create temporary bins with merge applied
        auto temp_bins = bins_;
        temp_bins[i].merge_with(temp_bins[i+1]);
        temp_bins.erase(temp_bins.begin() + i + 1);
        
        // Compute metrics for merged bin
        for (auto& bin : temp_bins) {
          bin.compute_metrics(total_class_counts_);
        }
        
        // Calculate IV loss
        total_iv_loss = 0.0;
        for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
          double new_iv = calculate_class_iv(temp_bins, class_idx);
          mwoe_cache_->set_class_iv(i, i+1, class_idx, new_iv);
          total_iv_loss += original_ivs[class_idx] - new_iv;
        }
      }
      
      // Update best merge if necessary
      if (total_iv_loss < min_total_iv_loss) {
        min_total_iv_loss = total_iv_loss;
        best_merge_idx = i;
      }
    }
    
    // Perform the best merge
    merge_bins(best_merge_idx, best_merge_idx + 1);
  }
  
  // Merge bins with caching updates
  void merge_bins(size_t idx1, size_t idx2) {
    if (idx1 >= bins_.size() || idx2 >= bins_.size() || idx1 == idx2) return;
    if (idx2 < idx1) std::swap(idx1, idx2);
    
    // Merge bins
    bins_[idx1].merge_with(bins_[idx2]);
    bins_.erase(bins_.begin() + idx2);
    
    // Recompute metrics
    compute_metrics();
    
    // Update cache
    mwoe_cache_->invalidate_bin(idx1);
    mwoe_cache_->resize(bins_.size());
  }
  
  // Improved algorithm for monotonicity correction
  void improve_monotonicity() {
    // For each class, check and fix monotonicity issues
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      // Find most severe violation
      double max_violation = 0.0;
      size_t violation_idx = 0;
      bool found_violation = false;
      
      // Identify the most significant monotonicity violation
      for (size_t i = 1; i < bins_.size(); ++i) {
        double curr_violation = 0.0;
        
        // Check if this bin violates monotonicity with neighbors
        bool is_peak = (i > 0 && i + 1 < bins_.size() &&
                        bins_[i].woes[class_idx] > bins_[i-1].woes[class_idx] + EPSILON && 
                        bins_[i].woes[class_idx] > bins_[i+1].woes[class_idx] + EPSILON);
        
        bool is_valley = (i > 0 && i + 1 < bins_.size() &&
                          bins_[i].woes[class_idx] < bins_[i-1].woes[class_idx] - EPSILON && 
                          bins_[i].woes[class_idx] < bins_[i+1].woes[class_idx] - EPSILON);
        
        if (is_peak) {
          curr_violation = std::max(
            bins_[i].woes[class_idx] - bins_[i-1].woes[class_idx],
                                                      bins_[i].woes[class_idx] - bins_[i+1].woes[class_idx]);
        } else if (is_valley) {
          curr_violation = std::max(
            bins_[i-1].woes[class_idx] - bins_[i].woes[class_idx],
                                                      bins_[i+1].woes[class_idx] - bins_[i].woes[class_idx]);
        }
        
        if (curr_violation > max_violation) {
          max_violation = curr_violation;
          violation_idx = i;
          found_violation = true;
        }
      }
      
      // If found a violation, fix it with minimal IV loss
      if (found_violation) {
        // Calculate IVs for potential merges
        auto original_ivs = std::vector<double>(n_classes_);
        for (size_t i = 0; i < n_classes_; ++i) {
          original_ivs[i] = calculate_class_iv(bins_, i);
        }
        
        // Try merge with previous bin
        double loss_prev = 0.0;
        if (violation_idx > 0) {
          auto temp_bins = bins_;
          temp_bins[violation_idx-1].merge_with(temp_bins[violation_idx]);
          temp_bins.erase(temp_bins.begin() + violation_idx);
          
          for (auto& bin : temp_bins) {
            bin.compute_metrics(total_class_counts_);
          }
          
          for (size_t i = 0; i < n_classes_; ++i) {
            loss_prev += original_ivs[i] - calculate_class_iv(temp_bins, i);
          }
        } else {
          loss_prev = std::numeric_limits<double>::max();
        }
        
        // Try merge with next bin
        double loss_next = 0.0;
        if (violation_idx + 1 < bins_.size()) {
          auto temp_bins = bins_;
          temp_bins[violation_idx].merge_with(temp_bins[violation_idx+1]);
          temp_bins.erase(temp_bins.begin() + violation_idx + 1);
          
          for (auto& bin : temp_bins) {
            bin.compute_metrics(total_class_counts_);
          }
          
          for (size_t i = 0; i < n_classes_; ++i) {
            loss_next += original_ivs[i] - calculate_class_iv(temp_bins, i);
          }
        } else {
          loss_next = std::numeric_limits<double>::max();
        }
        
        // Perform merge with minimal IV loss
        if (loss_prev <= loss_next && violation_idx > 0) {
          merge_bins(violation_idx - 1, violation_idx);
        } else if (violation_idx + 1 < bins_.size()) {
          merge_bins(violation_idx, violation_idx + 1);
        }
        
        // After a successful merge, exit this class's loop
        break;
      }
    }
  }
  
  // Ensure monotonic ordering of bins
  void ensure_monotonic_order() {
    // Sort bins for each class to ensure monotonicity
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      // Only reorder if not already monotonic
      if (!is_monotonic_for_class(bins_, class_idx)) {
        std::stable_sort(bins_.begin(), bins_.end(),
                         [class_idx](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                           return a.woes[class_idx] < b.woes[class_idx];
                         });
        
        // Recompute metrics after reordering
        compute_metrics();
      }
    }
  }
  
public:
  // Enhanced constructor with better validation
  OptimalBinningCategoricalJEDIMWoE(
    const std::vector<std::string>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : feature_(feature),
  target_(target),
  n_classes_(0),  // Will be set in validate_inputs
  min_bins_(min_bins), 
  max_bins_(max_bins),
  bin_cutoff_(bin_cutoff), 
  max_n_prebins_(max_n_prebins),
  bin_separator_(bin_separator),
  convergence_threshold_(convergence_threshold),
  max_iterations_(max_iterations),
  converged_(false), 
  iterations_run_(0),
  use_divergence_(true)  // Start with divergence-based merging
  {
    validate_inputs();
    
    // Adjust parameters based on unique categories
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    // Cap max_bins at number of unique categories
    max_bins_ = std::min(max_bins_, ncat);
    
    // Ensure min_bins is valid
    min_bins_ = std::min(min_bins_, max_bins_);
    
    // Ensure max_n_prebins is sufficient
    if (max_n_prebins_ < min_bins_) {
      max_n_prebins_ = min_bins_;
    }
  }
  
  // Optimized fit method
  void fit() {
    // Handle special case of few categories
    std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    if (ncat <= 2) {
      // Trivial case: ≤2 categories
      initial_binning();
      compute_metrics();
      converged_ = true;
      iterations_run_ = 0;
      return;
    }
    
    // Normal flow for many categories
    initial_binning();
    merge_low_freq();
    compute_metrics();
    
    // Reduce number of pre-bins if needed
    while (static_cast<int>(bins_.size()) > max_n_prebins_) {
      merge_most_similar_bins();
    }
    
    // Optimize bins
    optimize();
  }
  
  // Get results with enhanced structure
  Rcpp::List get_results() const {
    size_t n_bins = bins_.size();
    
    // Pre-allocate result vectors
    CharacterVector bin_names(n_bins);
    NumericMatrix woes(n_bins, n_classes_);
    NumericMatrix ivs(n_bins, n_classes_);
    IntegerVector counts(n_bins);
    IntegerMatrix class_counts(n_bins, n_classes_);
    NumericMatrix class_rates(n_bins, n_classes_);
    NumericVector ids(n_bins);
    NumericVector total_ivs(n_classes_);
    
    // Fill results
    for (size_t i = 0; i < n_bins; ++i) {
      bin_names[i] = utils::join_categories(bins_[i].categories, bin_separator_);
      counts[i] = bins_[i].total_count;
      ids[i] = i + 1;
      
      for (size_t j = 0; j < n_classes_; ++j) {
        woes(i,j) = bins_[i].woes[j];
        ivs(i,j) = bins_[i].ivs[j];
        class_counts(i,j) = bins_[i].class_counts[j];
        class_rates(i,j) = bins_[i].class_rates[j];
        
        // Add to total IV for this class
        total_ivs[j] += std::fabs(bins_[i].ivs[j]);
      }
    }
    
    // Return enhanced results
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("count") = counts,
      Named("class_counts") = class_counts,
      Named("class_rates") = class_rates,
      Named("converged") = converged_,
      Named("iterations") = iterations_run_,
      Named("n_classes") = static_cast<int>(n_classes_),
      Named("total_iv") = total_ivs
    );
  }
};

//' @title Optimal Binning for Categorical Variables with Multinomial Target using JEDI-MWoE
//'
//' @description
//' Implements an optimized categorical binning algorithm that extends the JEDI (Joint Entropy 
//' Discretization and Integration) framework to handle multinomial response variables using 
//' M-WOE (Multinomial Weight of Evidence). This implementation provides a robust solution for
//' categorical feature discretization in multinomial classification problems while maintaining
//' monotonic relationships and optimizing information value.
//'
//' @details
//' The algorithm implements a sophisticated binning strategy based on information theory
//' and extends the traditional binary WOE to handle multiple classes. 
//'
//' ## Mathematical Framework
//'
//' 1. M-WOE Calculation (with Laplace smoothing):
//' For each bin i and class k:
//' \deqn{M-WOE_{i,k} = \ln\left(\frac{P(X = x_i|Y = k)}{P(X = x_i|Y \neq k)}\right)}
//' \deqn{= \ln\left(\frac{(n_{k,i} + \alpha)/(N_k + 2\alpha)}{(\sum_{j \neq k} n_{j,i} + \alpha)/(\sum_{j \neq k} N_j + 2\alpha)}\right)}
//'
//' where:
//' \itemize{
//'   \item \eqn{n_{k,i}} is the count of class k in bin i
//'   \item \eqn{N_k} is the total count of class k
//'   \item \eqn{\alpha} is the Laplace smoothing parameter (default: 0.5)
//'   \item The denominator represents the proportion in all other classes combined
//' }
//'
//' 2. Information Value:
//' For each class k:
//' \deqn{IV_k = \sum_{i=1}^{n} \left(P(X = x_i|Y = k) - P(X = x_i|Y \neq k)\right) \times M-WOE_{i,k}}
//'
//' 3. Jensen-Shannon Divergence:
//' For measuring statistical similarity between bins:
//' \deqn{JS(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)}
//'
//' where:
//' \itemize{
//'   \item \eqn{KL} is the Kullback-Leibler divergence
//'   \item \eqn{M = \frac{1}{2}(P+Q)} is the midpoint distribution
//'   \item \eqn{P} and \eqn{Q} are the class distributions of two bins
//' }
//'
//' 4. Optimization Objective:
//' \deqn{maximize \sum_{k=1}^{K} IV_k}
//' subject to:
//' \itemize{
//'   \item Monotonicity constraints for each class
//'   \item Minimum bin size constraints
//'   \item Number of bins constraints
//' }
//'
//' ## Algorithm Phases
//' \enumerate{
//'   \item Initial Binning: Creates individual bins for unique categories
//'   \item Low Frequency Treatment: Merges rare categories based on bin_cutoff
//'   \item Monotonicity Optimization: Iteratively merges bins while maintaining monotonicity
//'   \item Final Adjustment: Ensures constraints on number of bins are met
//' }
//'
//' ## Merging Strategy
//' The algorithm alternates between two merging strategies:
//' \itemize{
//'   \item Statistical similarity-based merging using Jensen-Shannon divergence
//'   \item Information value-based merging that minimizes IV loss
//' }
//'
//' ## Statistical Robustness
//' \itemize{
//'   \item Employs Laplace smoothing for stable probability estimates
//'   \item Uses epsilon protection against numerical instability
//'   \item Detects and resolves monotonicity violations efficiently
//' }
//'
//' @param target Integer vector of class labels (0 to n_classes-1). Must be consecutive
//'        integers starting from 0.
//'
//' @param feature Character vector of categorical values to be binned. Must have the
//'        same length as target.
//'
//' @param min_bins Minimum number of bins in the output (default: 3). Will be 
//'        automatically adjusted if number of unique categories is less than min_bins.
//'        Value must be >= 1.
//'
//' @param max_bins Maximum number of bins allowed in the output (default: 5). Must be
//'        >= min_bins. Algorithm will merge bins if necessary to meet this constraint.
//'
//' @param bin_cutoff Minimum relative frequency threshold for individual bins 
//'        (default: 0.05). Categories with frequency below this threshold will be
//'        candidates for merging. Value must be between 0 and 1.
//'
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//'        Controls initial complexity before optimization phase. Must be >= min_bins.
//'
//' @param bin_separator String separator used when combining category names 
//'        (default: "%;%"). Used to create readable bin labels.
//'
//' @param convergence_threshold Convergence threshold for Information Value change
//'        (default: 1e-6). Algorithm stops when IV change is below this value.
//'
//' @param max_iterations Maximum number of optimization iterations (default: 1000).
//'        Prevents infinite loops in edge cases.
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Numeric identifiers for each bin.
//'   \item bin: Character vector of bin names (concatenated categories).
//'   \item woe: Numeric matrix (n_bins × n_classes) of M-WOE values for each class.
//'   \item iv: Numeric matrix (n_bins × n_classes) of IV contributions for each class.
//'   \item count: Integer vector of total observation counts per bin.
//'   \item class_counts: Integer matrix (n_bins × n_classes) of counts per class per bin.
//'   \item class_rates: Numeric matrix (n_bins × n_classes) of class rates per bin.
//'   \item converged: Logical indicating whether algorithm converged.
//'   \item iterations: Integer count of optimization iterations performed.
//'   \item n_classes: Integer indicating number of classes detected.
//'   \item total_iv: Numeric vector of total IV per class.
//' }
//'
//' @examples
//' # Basic usage with 3 classes
//' feature <- c("A", "B", "A", "C", "B", "D", "A")
//' target <- c(0, 1, 2, 1, 0, 2, 1)
//' result <- optimal_binning_categorical_jedi_mwoe(target, feature)
//'
//' # With custom parameters
//' result <- optimal_binning_categorical_jedi_mwoe(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 4,
//'   bin_cutoff = 0.1,
//'   max_n_prebins = 15,
//'   convergence_threshold = 1e-8
//' )
//'
//' @references
//' \itemize{
//'   \item Beltrami, M. et al. (2021). JEDI: Joint Entropy Discretization and Integration. arXiv preprint arXiv:2101.03228.
//'   \item Thomas, L.C. (2009). Consumer Credit Models: Pricing, Profit and Portfolios. Oxford University Press.
//'   \item Good, I.J. (1950). Probability and the Weighing of Evidence. Charles Griffin & Company.
//'   \item Kullback, S. (1959). Information Theory and Statistics. John Wiley & Sons.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @note
//' Performance Considerations:
//' \itemize{
//'   \item Time complexity: O(n_classes * n_samples * log(n_samples))
//'   \item Space complexity: O(n_classes * n_bins)
//'   \item For large datasets, initial binning phase may be memory-intensive
//' }
//'
//' Edge Cases:
//' \itemize{
//'   \item Single category: Returns original category as single bin
//'   \item All samples in one class: Creates degenerate case with warning
//'   \item Missing values: Treated as a special category "__MISSING__"
//' }
//'
//' @seealso
//' \itemize{
//'   \item optimal_binning_categorical_jedi for binary classification
//'   \item woe_transformation for applying WOE transformation
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi_mwoe(
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
 try {
   // Handle missing values in feature
   std::vector<std::string> feature_vec;
   feature_vec.reserve(feature.size());
   
   for (R_xlen_t i = 0; i < feature.size(); ++i) {
     if (feature[i] == NA_STRING) {
       feature_vec.push_back(MISSING_VALUE);
     } else {
       feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
     }
   }
   
   // Check for missing values in target
   std::vector<int> target_vec;
   target_vec.reserve(target.size());
   
   for (R_xlen_t i = 0; i < target.size(); ++i) {
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values");
     }
     target_vec.push_back(target[i]);
   }
   
   // Execute optimized algorithm
   OptimalBinningCategoricalJEDIMWoE jedi(
       feature_vec, target_vec,
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold,
       max_iterations
   );
   jedi.fit();
   return jedi.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal_binning_categorical_jedi_mwoe: " + std::string(e.what()));
 }
}











// // [[Rcpp::depends(Rcpp)]]
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// #include <sstream>
// #include <memory>
// 
// using namespace Rcpp;
// 
// // Constantes globais
// static constexpr double EPS = 1e-10;
// static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// 
// // Cache para valores de MWoE e IV
// class MWoECache {
// private:
//   std::vector<std::vector<std::vector<double>>> bin_pair_iv_cache;
//   size_t n_classes;
//   bool enabled;
//   
// public:
//   MWoECache(size_t max_bins, size_t num_classes, bool use_cache = true) 
//     : n_classes(num_classes), enabled(use_cache && max_bins > 0) {
//     if (enabled) {
//       // Três dimensões: bin1 x bin2 x classe
//       bin_pair_iv_cache.resize(max_bins);
//       for (auto& matrix : bin_pair_iv_cache) {
//         matrix.resize(max_bins);
//         for (auto& row : matrix) {
//           row.resize(n_classes, -1.0);
//         }
//       }
//     }
//   }
//   
//   inline double get_class_iv(size_t bin1, size_t bin2, size_t class_idx) {
//     if (!enabled || bin1 >= bin_pair_iv_cache.size() || bin2 >= bin_pair_iv_cache[bin1].size()) {
//       return -1.0;
//     }
//     return bin_pair_iv_cache[bin1][bin2][class_idx];
//   }
//   
//   inline void set_class_iv(size_t bin1, size_t bin2, size_t class_idx, double value) {
//     if (!enabled || bin1 >= bin_pair_iv_cache.size() || bin2 >= bin_pair_iv_cache[bin1].size()) {
//       return;
//     }
//     bin_pair_iv_cache[bin1][bin2][class_idx] = value;
//   }
//   
//   inline void invalidate_bin(size_t bin_idx) {
//     if (!enabled || bin_idx >= bin_pair_iv_cache.size()) {
//       return;
//     }
//     
//     // Limpa toda a linha e coluna relacionada ao bin
//     for (size_t i = 0; i < bin_pair_iv_cache.size(); ++i) {
//       if (i < bin_pair_iv_cache[bin_idx].size()) {
//         std::fill(bin_pair_iv_cache[bin_idx][i].begin(), bin_pair_iv_cache[bin_idx][i].end(), -1.0);
//       }
//       if (bin_idx < bin_pair_iv_cache[i].size()) {
//         std::fill(bin_pair_iv_cache[i][bin_idx].begin(), bin_pair_iv_cache[i][bin_idx].end(), -1.0);
//       }
//     }
//   }
//   
//   inline void resize(size_t new_size) {
//     if (!enabled) return;
//     
//     bin_pair_iv_cache.resize(new_size);
//     for (auto& matrix : bin_pair_iv_cache) {
//       matrix.resize(new_size);
//       for (auto& row : matrix) {
//         row.resize(n_classes, -1.0);
//       }
//     }
//   }
// };
// 
// // Estrutura otimizada para armazenar informações de bins multinomiais
// struct MultiCatBinInfo {
//   std::vector<std::string> categories;
//   int total_count;
//   std::vector<int> class_counts;
//   std::vector<double> woes;
//   std::vector<double> ivs;
//   std::vector<double> class_rates; // Cache para taxas de classe
//   
//   // Construtor padrão otimizado
//   MultiCatBinInfo() : total_count(0) {}
//   
//   // Construtor com número de classes
//   MultiCatBinInfo(size_t n_classes) 
//     : total_count(0), 
//       class_counts(n_classes, 0),
//       woes(n_classes, 0.0),
//       ivs(n_classes, 0.0),
//       class_rates(n_classes, 0.0) {
//         categories.reserve(8); // Pré-alocação para categorias típicas
//   }
//   
//   // Método para adicionar uma categoria com sua classe
//   inline void add_category(const std::string& cat, int class_idx, size_t n_classes) {
//     if (categories.empty()) {
//       // Se é o primeiro, inicialize vetores
//       class_counts.resize(n_classes, 0);
//       woes.resize(n_classes, 0.0);
//       ivs.resize(n_classes, 0.0);
//       class_rates.resize(n_classes, 0.0);
//     }
//     
//     categories.push_back(cat);
//     total_count++;
//     class_counts[class_idx]++;
//     update_class_rates();
//   }
//   
//   // Método para mesclar com outro bin
//   inline void merge_with(const MultiCatBinInfo& other) {
//     // Pré-aloca para evitar realocações
//     categories.reserve(categories.size() + other.categories.size());
//     categories.insert(categories.end(), other.categories.begin(), other.categories.end());
//     
//     total_count += other.total_count;
//     
//     for (size_t i = 0; i < class_counts.size(); ++i) {
//       class_counts[i] += other.class_counts[i];
//     }
//     
//     update_class_rates();
//   }
//   
//   // Método para atualizar taxas de classe
//   inline void update_class_rates() {
//     if (total_count > 0) {
//       for (size_t i = 0; i < class_counts.size(); ++i) {
//         class_rates[i] = static_cast<double>(class_counts[i]) / total_count;
//       }
//     }
//   }
//   
//   // Método para calcular MWoE e IV para cada classe
//   inline void compute_metrics(const std::vector<int>& total_class_counts) {
//     for (size_t class_idx = 0; class_idx < class_counts.size(); ++class_idx) {
//       // Taxa de classe atual
//       double class_rate = static_cast<double>(class_counts[class_idx]) / 
//         static_cast<double>(total_class_counts[class_idx]);
//       
//       // Total e taxa para "outras classes"
//       int others_total = 0;
//       for (size_t i = 0; i < class_counts.size(); ++i) {
//         if (i != class_idx) {
//           others_total += class_counts[i];
//         }
//       }
//       
//       int total_others = 0;
//       for (size_t i = 0; i < total_class_counts.size(); ++i) {
//         if (i != class_idx) {
//           total_others += total_class_counts[i];
//         }
//       }
//       
//       double others_rate = (total_others > 0) ? 
//       static_cast<double>(others_total) / static_cast<double>(total_others) : 0.0;
//       
//       // MWoE e IV para esta classe
//       double safe_class = std::max(class_rate, EPS);
//       double safe_others = std::max(others_rate, EPS);
//       
//       woes[class_idx] = std::log(safe_class / safe_others);
//       ivs[class_idx] = (class_rate - others_rate) * woes[class_idx];
//     }
//   }
// };
// 
// // Classe principal otimizada
// class OptimalBinningCategoricalJEDIMWoE {
// private:
//   std::vector<std::string> feature_;
//   std::vector<int> target_;
//   size_t n_classes_;
//   int min_bins_;
//   int max_bins_;
//   double bin_cutoff_;
//   int max_n_prebins_;
//   std::string bin_separator_;
//   double convergence_threshold_;
//   int max_iterations_;
//   
//   std::vector<MultiCatBinInfo> bins_;
//   std::vector<int> total_class_counts_;
//   std::unique_ptr<MWoECache> mwoe_cache_;
//   bool converged_;
//   int iterations_run_;
//   
//   // Validação de entrada melhorada
//   void validate_inputs() {
//     if (feature_.empty() || feature_.size() != target_.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same non-empty length");
//     }
//     
//     // Encontra o valor máximo em target_ para determinar n_classes_
//     int max_class = -1;
//     std::unordered_set<int> class_set;
//     
//     for (int t : target_) {
//       if (t < 0) {
//         throw std::invalid_argument("Target values must be non-negative integers");
//       }
//       
//       max_class = std::max(max_class, t);
//       class_set.insert(t);
//     }
//     
//     // Verifica se target tem classes consecutivas começando de 0
//     n_classes_ = max_class + 1;
//     
//     if (class_set.size() < 2) {
//       throw std::invalid_argument("Target must have at least 2 distinct classes");
//     }
//     
//     for (int i = 0; i < static_cast<int>(n_classes_); ++i) {
//       if (class_set.find(i) == class_set.end()) {
//         throw std::invalid_argument("Target classes must be consecutive integers starting from 0");
//       }
//     }
//   }
//   
//   // Cálculo eficiente de M-WoE para uma classe
//   inline double calculate_mwoe(const MultiCatBinInfo& bin, size_t class_idx) const {
//     if (total_class_counts_[class_idx] <= 0) return 0.0;
//     
//     // Taxa para a classe atual
//     double class_rate = static_cast<double>(bin.class_counts[class_idx]) / 
//       static_cast<double>(total_class_counts_[class_idx]);
//     
//     // Taxa para as outras classes combinadas
//     int others_total = 0;
//     int total_others = 0;
//     
//     for (size_t i = 0; i < n_classes_; ++i) {
//       if (i != class_idx) {
//         others_total += bin.class_counts[i];
//         total_others += total_class_counts_[i];
//       }
//     }
//     
//     double others_rate = (total_others > 0) ? 
//     static_cast<double>(others_total) / static_cast<double>(total_others) : 0.0;
//     
//     // Proteção contra divisão por zero
//     double safe_class = std::max(class_rate, EPS);
//     double safe_others = std::max(others_rate, EPS);
//     
//     return std::log(safe_class / safe_others);
//   }
//   
//   // Cálculo eficiente de IV para uma classe
//   double calculate_class_iv(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
//     double iv = 0.0;
//     
//     for (const auto& bin : current_bins) {
//       if (bin.total_count <= 0) continue;
//       
//       // Usa valores pré-calculados se disponíveis
//       if (!bin.ivs.empty() && class_idx < bin.ivs.size()) {
//         iv += bin.ivs[class_idx];
//         continue;
//       }
//       
//       // Taxa para a classe atual
//       double class_rate = static_cast<double>(bin.class_counts[class_idx]) / 
//         static_cast<double>(total_class_counts_[class_idx]);
//       
//       // Taxa para as outras classes combinadas
//       int others_total = 0;
//       int total_others = 0;
//       
//       for (size_t i = 0; i < n_classes_; ++i) {
//         if (i != class_idx) {
//           others_total += bin.class_counts[i];
//           total_others += total_class_counts_[i];
//         }
//       }
//       
//       double others_rate = (total_others > 0) ? 
//       static_cast<double>(others_total) / static_cast<double>(total_others) : 0.0;
//       
//       if (class_rate > 0 && others_rate > 0) {
//         double woe = std::log(std::max(class_rate, EPS) / std::max(others_rate, EPS));
//         iv += (class_rate - others_rate) * woe;
//       }
//     }
//     
//     return iv;
//   }
//   
//   // Cálculo otimizado de WoE e IV para todos os bins
//   void compute_woe_iv(std::vector<MultiCatBinInfo>& current_bins) {
//     for (auto& bin : current_bins) {
//       bin.compute_metrics(total_class_counts_);
//     }
//   }
//   
//   // Verificação de monotonicidade otimizada para uma classe
//   bool is_monotonic_for_class(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
//     if (current_bins.size() <= 2) return true;
//     
//     // Determina a direção da monotonicidade com base nos primeiros dois bins
//     bool should_increase = current_bins[1].woes[class_idx] >= current_bins[0].woes[class_idx];
//     
//     // Verifica o restante dos bins
//     for (size_t i = 2; i < current_bins.size(); ++i) {
//       if (should_increase && current_bins[i].woes[class_idx] < current_bins[i-1].woes[class_idx] - EPS) {
//         return false;
//       }
//       if (!should_increase && current_bins[i].woes[class_idx] > current_bins[i-1].woes[class_idx] + EPS) {
//         return false;
//       }
//     }
//     
//     return true;
//   }
//   
//   // Verificação da monotonicidade para todas as classes
//   bool is_monotonic(const std::vector<MultiCatBinInfo>& current_bins) const {
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       if (!is_monotonic_for_class(current_bins, class_idx)) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   // Binning inicial otimizado
//   void initial_binning() {
//     // Estima o número de categorias únicas
//     size_t est_cats = std::min(feature_.size() / 4, static_cast<size_t>(1024));
//     std::unordered_map<std::string, MultiCatBinInfo> bin_map;
//     bin_map.reserve(est_cats);
//     
//     // Inicializa contagens por classe
//     total_class_counts_ = std::vector<int>(n_classes_, 0);
//     
//     // Processa em uma única passagem
//     for (size_t i = 0; i < feature_.size(); ++i) {
//       const std::string& cat = feature_[i];
//       int class_idx = target_[i];
//       
//       // Inicializa o bin se necessário
//       if (bin_map.find(cat) == bin_map.end()) {
//         bin_map[cat] = MultiCatBinInfo(n_classes_);
//         bin_map[cat].categories.push_back(cat);
//       }
//       
//       // Atualiza contagens
//       bin_map[cat].total_count++;
//       bin_map[cat].class_counts[class_idx]++;
//       bin_map[cat].update_class_rates();
//       
//       total_class_counts_[class_idx]++;
//     }
//     
//     // Transfere para o vetor de bins
//     bins_.clear();
//     bins_.reserve(bin_map.size());
//     
//     for (auto& kv : bin_map) {
//       bins_.push_back(std::move(kv.second));
//     }
//     
//     // Inicializa cache de MWoE
//     mwoe_cache_ = std::make_unique<MWoECache>(bins_.size(), n_classes_, bins_.size() > 10);
//   }
//   
//   // Mescla eficiente de categorias raras
//   void merge_low_freq() {
//     int total_count = 0;
//     for (const auto& bin : bins_) {
//       total_count += bin.total_count;
//     }
//     double cutoff_count = total_count * bin_cutoff_;
//     
//     // Ordena bins por contagem (crescente)
//     std::sort(bins_.begin(), bins_.end(), 
//               [](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
//                 return a.total_count < b.total_count;
//               });
//     
//     // Prepara novos bins, com espaço reservado
//     std::vector<MultiCatBinInfo> new_bins;
//     new_bins.reserve(bins_.size());
//     
//     // Bin para categorias raras
//     MultiCatBinInfo rare_bin(n_classes_);
//     bool has_rare = false;
//     
//     // Processa cada bin
//     for (auto& bin : bins_) {
//       if (bin.total_count >= cutoff_count || static_cast<int>(new_bins.size()) < min_bins_) {
//         // Bin com frequência adequada
//         new_bins.push_back(std::move(bin));
//       } else {
//         // Bin raro, mesclar
//         rare_bin.merge_with(bin);
//         has_rare = true;
//       }
//     }
//     
//     // Adiciona o bin de categorias raras, se existir
//     if (has_rare) {
//       if (rare_bin.categories.empty()) {
//         rare_bin.categories.push_back("Others");
//       }
//       new_bins.push_back(std::move(rare_bin));
//     }
//     
//     bins_ = std::move(new_bins);
//     mwoe_cache_->resize(bins_.size());
//   }
//   
//   // Algoritmo de otimização principal
//   void optimize() {
//     // Vetor para rastrear IVs anteriores
//     std::vector<double> prev_ivs(n_classes_);
//     for (size_t i = 0; i < n_classes_; ++i) {
//       prev_ivs[i] = calculate_class_iv(bins_, i);
//     }
//     
//     converged_ = false;
//     iterations_run_ = 0;
//     
//     // Loop principal de otimização
//     while (iterations_run_ < max_iterations_) {
//       // Verifica se já atingimos os critérios de parada
//       if (is_monotonic(bins_) && 
//           static_cast<int>(bins_.size()) <= max_bins_ && 
//           static_cast<int>(bins_.size()) >= min_bins_) {
//         converged_ = true;
//         break;
//       }
//       
//       // Mescla bins conforme necessário
//       if (static_cast<int>(bins_.size()) > min_bins_) {
//         if (static_cast<int>(bins_.size()) > max_bins_) {
//           // Precisamos reduzir o número de bins
//           merge_adjacent_bins();
//         } else {
//           // Precisamos melhorar a monotonicidade
//           improve_monotonicity();
//         }
//       } else {
//         // Não podemos mesclar mais (atingimos min_bins)
//         break;
//       }
//       
//       // Verifica convergência
//       std::vector<double> current_ivs(n_classes_);
//       bool all_converged = true;
//       
//       for (size_t i = 0; i < n_classes_; ++i) {
//         current_ivs[i] = calculate_class_iv(bins_, i);
//         if (std::abs(current_ivs[i] - prev_ivs[i]) >= convergence_threshold_) {
//           all_converged = false;
//         }
//       }
//       
//       if (all_converged) {
//         converged_ = true;
//         break;
//       }
//       
//       prev_ivs = std::move(current_ivs);
//       iterations_run_++;
//     }
//     
//     // Ajusta para garantir max_bins
//     while (static_cast<int>(bins_.size()) > max_bins_) {
//       merge_adjacent_bins();
//     }
//     
//     // Ordena bins para garantir monotonicidade
//     ensure_monotonic_order();
//     compute_woe_iv(bins_);
//   }
//   
//   // Mescla otimizada de bins adjacentes
//   void merge_adjacent_bins() {
//     if (bins_.size() <= 2) return;
//     
//     double min_total_iv_loss = std::numeric_limits<double>::max();
//     size_t best_merge_idx = 0;
//     
//     // Calcula IVs originais para cada classe
//     std::vector<double> original_ivs(n_classes_);
//     for (size_t i = 0; i < n_classes_; ++i) {
//       original_ivs[i] = calculate_class_iv(bins_, i);
//     }
//     
//     // Testa cada par de bins adjacentes
//     for (size_t i = 0; i < bins_.size() - 1; ++i) {
//       // Verifica cache
//       double total_iv_loss = 0.0;
//       bool all_cached = true;
//       
//       for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//         double cached_iv = mwoe_cache_->get_class_iv(i, i+1, class_idx);
//         
//         if (cached_iv >= 0.0) {
//           total_iv_loss += original_ivs[class_idx] - cached_iv;
//         } else {
//           all_cached = false;
//           break;
//         }
//       }
//       
//       // Se não estiver em cache, calcula
//       if (!all_cached) {
//         auto temp_bins = bins_;
//         merge_bins_in_temp(temp_bins, i, i + 1);
//         
//         total_iv_loss = 0.0;
//         for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//           double new_iv = calculate_class_iv(temp_bins, class_idx);
//           mwoe_cache_->set_class_iv(i, i+1, class_idx, new_iv);
//           total_iv_loss += original_ivs[class_idx] - new_iv;
//         }
//       }
//       
//       // Atualiza melhor mesclagem se necessário
//       if (total_iv_loss < min_total_iv_loss) {
//         min_total_iv_loss = total_iv_loss;
//         best_merge_idx = i;
//       }
//     }
//     
//     // Realiza a melhor mesclagem
//     merge_bins(best_merge_idx, best_merge_idx + 1);
//   }
//   
//   // Mesclagem eficiente em um vetor temporário
//   void merge_bins_in_temp(std::vector<MultiCatBinInfo>& temp, size_t idx1, size_t idx2) {
//     if (idx1 >= temp.size() || idx2 >= temp.size() || idx1 == idx2) return;
//     if (idx2 < idx1) std::swap(idx1, idx2);
//     
//     // Mescla bins
//     temp[idx1].merge_with(temp[idx2]);
//     temp.erase(temp.begin() + idx2);
//     
//     // Recalcula métricas
//     compute_woe_iv(temp);
//   }
//   
//   // Mesclagem eficiente em bins_
//   void merge_bins(size_t idx1, size_t idx2) {
//     if (idx1 >= bins_.size() || idx2 >= bins_.size() || idx1 == idx2) return;
//     if (idx2 < idx1) std::swap(idx1, idx2);
//     
//     // Mescla bins
//     bins_[idx1].merge_with(bins_[idx2]);
//     bins_.erase(bins_.begin() + idx2);
//     
//     // Recalcula métricas
//     compute_woe_iv(bins_);
//     
//     // Atualiza cache
//     mwoe_cache_->invalidate_bin(idx1);
//     mwoe_cache_->resize(bins_.size());
//   }
//   
//   // Melhoria de monotonicidade otimizada
//   void improve_monotonicity() {
//     // Para cada classe, verifica e corrige problemas de monotonicidade
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       for (size_t i = 1; i + 1 < bins_.size(); ++i) {
//         // Verifica padrões não monotônicos (picos ou vales)
//         bool is_peak = (bins_[i].woes[class_idx] > bins_[i-1].woes[class_idx] + EPS && 
//                         bins_[i].woes[class_idx] > bins_[i+1].woes[class_idx] + EPS);
//         
//         bool is_valley = (bins_[i].woes[class_idx] < bins_[i-1].woes[class_idx] - EPS && 
//                           bins_[i].woes[class_idx] < bins_[i+1].woes[class_idx] - EPS);
//         
//         if (is_peak || is_valley) {
//           // Calcula IVs originais
//           std::vector<double> orig_ivs(n_classes_);
//           for (size_t j = 0; j < n_classes_; ++j) {
//             orig_ivs[j] = calculate_class_iv(bins_, j);
//           }
//           
//           // Testa mesclagem com bin anterior
//           double loss1 = 0.0;
//           {
//             bool all_cached = true;
//             for (size_t j = 0; j < n_classes_; ++j) {
//               double cached_iv = mwoe_cache_->get_class_iv(i-1, i, j);
//               if (cached_iv >= 0.0) {
//                 loss1 += orig_ivs[j] - cached_iv;
//               } else {
//                 all_cached = false;
//                 break;
//               }
//             }
//             
//             if (!all_cached) {
//               std::vector<MultiCatBinInfo> temp1 = bins_;
//               merge_bins_in_temp(temp1, i-1, i);
//               loss1 = 0.0;
//               
//               for (size_t j = 0; j < n_classes_; ++j) {
//                 double new_iv = calculate_class_iv(temp1, j);
//                 mwoe_cache_->set_class_iv(i-1, i, j, new_iv);
//                 loss1 += orig_ivs[j] - new_iv;
//               }
//             }
//           }
//           
//           // Testa mesclagem com bin posterior
//           double loss2 = 0.0;
//           {
//             bool all_cached = true;
//             for (size_t j = 0; j < n_classes_; ++j) {
//               double cached_iv = mwoe_cache_->get_class_iv(i, i+1, j);
//               if (cached_iv >= 0.0) {
//                 loss2 += orig_ivs[j] - cached_iv;
//               } else {
//                 all_cached = false;
//                 break;
//               }
//             }
//             
//             if (!all_cached) {
//               std::vector<MultiCatBinInfo> temp2 = bins_;
//               merge_bins_in_temp(temp2, i, i+1);
//               loss2 = 0.0;
//               
//               for (size_t j = 0; j < n_classes_; ++j) {
//                 double new_iv = calculate_class_iv(temp2, j);
//                 mwoe_cache_->set_class_iv(i, i+1, j, new_iv);
//                 loss2 += orig_ivs[j] - new_iv;
//               }
//             }
//           }
//           
//           // Escolhe a mescla com menor perda
//           if (loss1 < loss2) {
//             merge_bins(i-1, i);
//           } else {
//             merge_bins(i, i+1);
//           }
//           break;
//         }
//       }
//     }
//   }
//   
//   // Ordenação eficiente para garantir monotonicidade
//   void ensure_monotonic_order() {
//     for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
//       // Só reordena se necessário
//       if (!is_monotonic_for_class(bins_, class_idx)) {
//         std::stable_sort(bins_.begin(), bins_.end(),
//                          [class_idx](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
//                            return a.woes[class_idx] < b.woes[class_idx];
//                          });
//         
//         // Recalcula métricas após reordenação
//         compute_woe_iv(bins_);
//       }
//     }
//   }
//   
//   // Junção eficiente de categorias
//   static std::string join_categories(const std::vector<std::string>& cats, const std::string& sep) {
//     if (cats.empty()) return "";
//     if (cats.size() == 1) return cats[0];
//     
//     // Estima tamanho total para pré-alocação
//     size_t total_length = 0;
//     for (const auto& cat : cats) {
//       total_length += cat.size();
//     }
//     total_length += sep.size() * (cats.size() - 1);
//     
//     std::string result;
//     result.reserve(total_length);
//     
//     result = cats[0];
//     for (size_t i = 1; i < cats.size(); ++i) {
//       result += sep;
//       result += cats[i];
//     }
//     
//     return result;
//   }
//   
// public:
//   // Construtor otimizado
//   OptimalBinningCategoricalJEDIMWoE(
//     const std::vector<std::string>& feature,
//     const std::vector<int>& target,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     std::string bin_separator = "%;%",
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000
//   ) : feature_(feature),
//   target_(target),
//   n_classes_(0), // Será definido em validate_inputs
//   min_bins_(min_bins), 
//   max_bins_(max_bins),
//   bin_cutoff_(bin_cutoff), 
//   max_n_prebins_(max_n_prebins),
//   bin_separator_(bin_separator),
//   convergence_threshold_(convergence_threshold),
//   max_iterations_(max_iterations),
//   converged_(false), 
//   iterations_run_(0)
//   {
//     validate_inputs();
//     
//     // Ajusta parâmetros baseado no número de categorias únicas
//     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
//     int ncat = static_cast<int>(unique_cats.size());
//     
//     if (ncat < min_bins_) {
//       min_bins_ = std::max(1, ncat);
//     }
//     if (max_bins_ < min_bins_) {
//       max_bins_ = min_bins_;
//     }
//     if (max_n_prebins_ < min_bins_) {
//       max_n_prebins_ = min_bins_;
//     }
//   }
//   
//   // Método fit otimizado
//   void fit() {
//     // Detecta caso de poucas categorias
//     std::unordered_set<std::string> unique_cats(feature_.begin(), feature_.end());
//     int ncat = static_cast<int>(unique_cats.size());
//     
//     if (ncat <= 2) {
//       // Caso trivial: <=2 categorias
//       initial_binning();
//       compute_woe_iv(bins_);
//       converged_ = true;
//       iterations_run_ = 0;
//       return;
//     }
//     
//     // Fluxo normal para muitas categorias
//     initial_binning();
//     merge_low_freq();
//     compute_woe_iv(bins_);
//     
//     // Reduz número de pré-bins se necessário
//     if (static_cast<int>(bins_.size()) > max_n_prebins_) {
//       while (static_cast<int>(bins_.size()) > max_n_prebins_) {
//         merge_adjacent_bins();
//       }
//     }
//     
//     // Otimiza bins
//     optimize();
//   }
//   
//   // Preparação de resultados otimizada
//   Rcpp::List get_results() const {
//     size_t n_bins = bins_.size();
//     
//     // Pré-aloca vetores de resultado
//     CharacterVector bin_names(n_bins);
//     NumericMatrix woes(n_bins, n_classes_);
//     NumericMatrix ivs(n_bins, n_classes_);
//     IntegerVector counts(n_bins);
//     IntegerMatrix class_counts(n_bins, n_classes_);
//     NumericVector ids(n_bins);
//     
//     // Preenche resultados
//     for (size_t i = 0; i < n_bins; ++i) {
//       bin_names[i] = join_categories(bins_[i].categories, bin_separator_);
//       counts[i] = bins_[i].total_count;
//       ids[i] = i + 1;
//       
//       for (size_t j = 0; j < n_classes_; ++j) {
//         woes(i,j) = bins_[i].woes[j];
//         ivs(i,j) = bins_[i].ivs[j];
//         class_counts(i,j) = bins_[i].class_counts[j];
//       }
//     }
//     
//     return Rcpp::List::create(
//       Named("id") = ids,
//       Named("bin") = bin_names,
//       Named("woe") = woes,
//       Named("iv") = ivs,
//       Named("count") = counts,
//       Named("class_counts") = class_counts,
//       Named("converged") = converged_,
//       Named("iterations") = iterations_run_,
//       Named("n_classes") = static_cast<int>(n_classes_)
//     );
//   }
// };
// 
// 
// //' @description
// //' Implements an optimized categorical binning algorithm that extends the JEDI (Joint Entropy 
// //' Discretization and Integration) framework to handle multinomial response variables using 
// //' M-WOE (Multinomial Weight of Evidence). This implementation provides a robust solution for
// //' categorical feature discretization in multinomial classification problems while maintaining
// //' monotonic relationships and optimizing information value.
// //'
// //' @details
// //' The algorithm implements a sophisticated binning strategy based on information theory
// //' and extends the traditional binary WOE to handle multiple classes. 
// //'
// //' Mathematical Framework:
// //'
// //' 1. M-WOE Calculation:
// //' For each bin i and class k:
// //' \deqn{M-WOE_{i,k} = \ln(\frac{P(X = x_i|Y = k)}{P(X = x_i|Y \neq k)})}
// //' \deqn{= \ln(\frac{n_{k,i}/N_k}{\sum_{j \neq k} n_{j,i}/N_j})}
// //'
// //' where:
// //' \itemize{
// //'   \item \eqn{n_{k,i}} is the count of class k in bin i
// //'   \item \eqn{N_k} is the total count of class k
// //'   \item The denominator represents the proportion in all other classes
// //' }
// //'
// //' 2. Information Value:
// //' For each class k:
// //' \deqn{IV_k = \sum_{i=1}^{n} (P(X = x_i|Y = k) - P(X = x_i|Y \neq k)) \times M-WOE_{i,k}}
// //'
// //' 3. Optimization Objective:
// //' \deqn{maximize \sum_{k=1}^{K} IV_k}
// //' subject to:
// //' \itemize{
// //'   \item Monotonicity constraints for each class
// //'   \item Minimum bin size constraints
// //'   \item Number of bins constraints
// //' }
// //'
// //' Algorithm Phases:
// //' \enumerate{
// //'   \item Initial Binning: Creates individual bins for unique categories
// //'   \item Low Frequency Treatment: Merges rare categories based on bin_cutoff
// //'   \item Monotonicity Optimization: Iteratively merges bins while maintaining monotonicity
// //'   \item Final Adjustment: Ensures constraints on number of bins are met
// //' }
// //'
// //' Numerical Stability:
// //' \itemize{
// //'   \item Uses epsilon-based protection against zero probabilities
// //'   \item Implements log-sum-exp trick for numerical stability
// //'   \item Handles edge cases and infinity values
// //' }
// //'
// //' @param target Integer vector of class labels (0 to n_classes-1). Must be consecutive
// //'        integers starting from 0.
// //'
// //' @param feature Character vector of categorical values to be binned. Must have the
// //'        same length as target.
// //'
// //' @param min_bins Minimum number of bins in the output (default: 3). Will be 
// //'        automatically adjusted if number of unique categories is less than min_bins.
// //'        Value must be >= 2.
// //'
// //' @param max_bins Maximum number of bins allowed in the output (default: 5). Must be
// //'        >= min_bins. Algorithm will merge bins if necessary to meet this constraint.
// //'
// //' @param bin_cutoff Minimum relative frequency threshold for individual bins 
// //'        (default: 0.05). Categories with frequency below this threshold will be
// //'        candidates for merging. Value must be between 0 and 1.
// //'
// //' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
// //'        Controls initial complexity before optimization phase. Must be >= min_bins.
// //'
// //' @param bin_separator String separator used when combining category names 
// //'        (default: "%;%"). Used to create readable bin labels.
// //'
// //' @param convergence_threshold Convergence threshold for Information Value change
// //'        (default: 1e-6). Algorithm stops when IV change is below this value.
// //'
// //' @param max_iterations Maximum number of optimization iterations (default: 1000).
// //'        Prevents infinite loops in edge cases.
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bin: Character vector of bin names (concatenated categories)
// //'   \item woe: Numeric matrix (n_bins × n_classes) of M-WOE values for each class
// //'   \item iv: Numeric matrix (n_bins × n_classes) of IV contributions for each class
// //'   \item count: Integer vector of total observation counts per bin
// //'   \item class_counts: Integer matrix (n_bins × n_classes) of counts per class per bin
// //'   \item converged: Logical indicating whether algorithm converged
// //'   \item iterations: Integer count of optimization iterations performed
// //'   \item n_classes: Integer indicating number of classes detected
// //' }
// //'
// //' @examples
// //' # Basic usage with 3 classes
// //' feature <- c("A", "B", "A", "C", "B", "D", "A")
// //' target <- c(0, 1, 2, 1, 0, 2, 1)
// //' result <- optimal_binning_categorical_jedi_mwoe(target, feature)
// //'
// //' # With custom parameters
// //' result <- optimal_binning_categorical_jedi_mwoe(
// //'   target = target,
// //'   feature = feature,
// //'   min_bins = 2,
// //'   max_bins = 4,
// //'   bin_cutoff = 0.1,
// //'   max_n_prebins = 15,
// //'   convergence_threshold = 1e-8
// //' )
// //'
// //' @references
// //' \itemize{
// //'   \item Beltrami, M. et al. (2021). JEDI: Joint Entropy Discretization and Integration
// //'   \item Thomas, L.C. (2009). Consumer Credit Models: Pricing, Profit and Portfolios
// //'   \item Good, I.J. (1950). Probability and the Weighing of Evidence
// //'   \item Kullback, S. (1959). Information Theory and Statistics
// //' }
// //'
// //' @note
// //' Performance Considerations:
// //' \itemize{
// //'   \item Time complexity: O(n_classes * n_samples * log(n_samples))
// //'   \item Space complexity: O(n_classes * n_bins)
// //'   \item For large datasets, initial binning phase may be memory-intensive
// //' }
// //'
// //' Edge Cases:
// //' \itemize{
// //'   \item Single category: Returns original category as single bin
// //'   \item All samples in one class: Creates degenerate case with warning
// //'   \item Missing values: Should be treated as separate category before input
// //' }
// //'
// //' @seealso
// //' \itemize{
// //'   \item optimal_binning_categorical_jedi for binary classification
// //'   \item woe_transformation for applying WOE transformation
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_jedi_mwoe(
//    Rcpp::IntegerVector target,
//    Rcpp::StringVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Verificações preliminares
//  if (feature.size() == 0 || target.size() == 0) {
//    stop("Feature and target vectors cannot be empty");
//  }
//  
//  if (feature.size() != target.size()) {
//    stop("Feature and target vectors must have the same length");
//  }
//  
//  // Conversão otimizada de R para C++
//  std::vector<std::string> feature_vec;
//  std::vector<int> target_vec;
//  
//  feature_vec.reserve(feature.size());
//  target_vec.reserve(target.size());
//  
//  for (R_xlen_t i = 0; i < feature.size(); ++i) {
//    // Tratamento de NAs
//    if (feature[i] == NA_STRING) {
//      feature_vec.push_back("NA");
//    } else {
//      feature_vec.push_back(as<std::string>(feature[i]));
//    }
//    
//    if (IntegerVector::is_na(target[i])) {
//      stop("Target cannot contain NA values");
//    } else {
//      target_vec.push_back(target[i]);
//    }
//  }
//  
//  try {
//    // Executa algoritmo otimizado
//    OptimalBinningCategoricalJEDIMWoE jedi(
//        feature_vec, target_vec,
//        min_bins, max_bins,
//        bin_cutoff, max_n_prebins,
//        bin_separator, convergence_threshold,
//        max_iterations
//    );
//    jedi.fit();
//    return jedi.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal_binning_categorical_jedi_mwoe: " + std::string(e.what()));
//  }
// }
