// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <memory>

using namespace Rcpp;

// Constants for better readability and precision
constexpr double EPSILON = 1e-10;
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5; // Adjustable prior strength for smoothing

// Optimized structure for storing bin information
struct BinInfo {
  std::vector<std::string> categories;
  int count = 0;
  int count_pos = 0;
  int count_neg = 0;
  double woe = 0.0;
  double iv = 0.0;
  double pos_rate = 0.0;  // Cache for pos_rate
  double neg_rate = 0.0;  // Cache for neg_rate
  
  // Constructor with pre-allocation for optimized memory usage
  BinInfo() {
    categories.reserve(8);  // Reasonable estimate to save on reallocations
  }
  
  // Efficient method to add a category
  inline void add_category(const std::string& cat, int cat_count, int cat_pos) {
    categories.push_back(cat);
    count += cat_count;
    count_pos += cat_pos;
    count_neg += (cat_count - cat_pos);
  }
  
  // Optimized method to merge bins
  inline void merge_with(const BinInfo& other) {
    // Reserve space to avoid reallocations
    categories.reserve(categories.size() + other.categories.size());
    categories.insert(categories.end(), other.categories.begin(), other.categories.end());
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
  }
  
  // Calculate WoE and IV with Bayesian smoothing
  inline void calculate_metrics(int total_pos, int total_neg) {
    // Calculate prior pseudo-counts based on overall prevalence
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
      (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    // Apply Bayesian smoothing to rates
    pos_rate = static_cast<double>(count_pos + prior_pos) / 
      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    neg_rate = static_cast<double>(count_neg + prior_neg) / 
      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    // Calculate WoE and IV with smoothed rates
    woe = std::log(pos_rate / neg_rate);
    iv = (pos_rate - neg_rate) * woe;
  }
};

// Class for incremental IV calculation caching
class IVCache {
private:
  std::vector<std::vector<double>> cache;
  bool enabled;
  
public:
  IVCache(size_t size, bool use_cache = true) : enabled(use_cache) {
    if (enabled) {
      cache.resize(size);
      for (auto& row : cache) {
        row.resize(size, -1.0);
      }
    }
  }
  
  double get(size_t i, size_t j) {
    if (!enabled || i >= cache.size() || j >= cache.size()) return -1.0;
    return cache[i][j];
  }
  
  void set(size_t i, size_t j, double value) {
    if (!enabled || i >= cache.size() || j >= cache.size()) return;
    cache[i][j] = value;
  }
  
  void invalidate_row(size_t i) {
    if (!enabled || i >= cache.size()) return;
    for (size_t j = 0; j < cache.size(); j++) {
      cache[i][j] = -1.0;
      if (i != j) cache[j][i] = -1.0;
    }
  }
  
  void resize(size_t new_size) {
    if (!enabled) return;
    cache.resize(new_size);
    for (auto& row : cache) {
      row.resize(new_size, -1.0);
    }
  }
};

class OptimalBinningCategoricalGMB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<BinInfo> bins;
  std::unique_ptr<IVCache> iv_cache;
  int total_pos = 0;
  int total_neg = 0;
  bool converged = false;
  int iterations_run = 0;
  
  // Enhanced input validation with more comprehensive checks
  void validateInput() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins.");
    }
    
    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
      return s.empty(); 
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }
    
    // Efficient check for binary target
    bool has_zero = false;
    bool has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must be binary (0 or 1).");
      
      // Early termination once we've seen both values
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
  }
  
  // Calculate WoE with protection against log(0) and improved smoothing
  inline double calculateWOE(int pos, int neg) const {
    // Calculate with Bayesian smoothing
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
      (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    double pos_rate = static_cast<double>(pos + prior_pos) / 
      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(neg + prior_neg) / 
      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    return std::log(pos_rate / neg_rate);
  }
  
  // Calculate total IV for a set of bins with improved numerical stability
  double calculateIV(const std::vector<BinInfo>& bins_to_check) const {
    double iv = 0.0;
    for (const auto& bin : bins_to_check) {
      // Use pre-calculated value if available
      if (bin.iv != 0.0 && std::isfinite(bin.iv)) {
        iv += bin.iv;
      } else {
        // Calculate with Bayesian smoothing
        double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / 
          (total_pos + total_neg);
        double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
        
        double pos_rate = static_cast<double>(bin.count_pos + prior_pos) / 
          static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
        double neg_rate = static_cast<double>(bin.count_neg + prior_neg) / 
          static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);
        
        double woe = std::log(pos_rate / neg_rate);
        double local_iv = (pos_rate - neg_rate) * woe;
        
        if (std::isfinite(local_iv)) {
          iv += local_iv;
        }
      }
    }
    return iv;
  }
  
  // Enhanced bin initialization with optimized counting
  void initializeBins() {
    // Efficient single-pass counting
    std::unordered_map<std::string, std::pair<int, int>> category_stats;
    category_stats.reserve(std::min(static_cast<size_t>(feature.size() / 4), static_cast<size_t>(1024)));
    
    total_pos = 0;
    total_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      auto& stats = category_stats[cat];
      
      if (target[i] == 1) {
        stats.first++;  // pos_count
        total_pos++;
      } else {
        stats.second++;  // neg_count
        total_neg++;
      }
    }
    
    // Check for extremely imbalanced datasets
    if (total_pos < 5 || total_neg < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    // Optimized bin creation
    bins.clear();
    bins.reserve(category_stats.size());
    
    for (const auto& [cat, stats] : category_stats) {
      BinInfo bin;
      bin.add_category(cat, stats.first + stats.second, stats.first);
      bins.push_back(std::move(bin));
    }
    
    // Sort by positive rate for consistent ordering
    for (auto& bin : bins) {
      bin.pos_rate = static_cast<double>(bin.count_pos) / static_cast<double>(std::max(bin.count, 1));
    }
    
    std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.pos_rate < b.pos_rate;
    });
    
    // Enhanced rare category handling
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const BinInfo& bin) { return sum + bin.count; });
    
    std::vector<BinInfo> merged_bins;
    merged_bins.reserve(bins.size());
    
    BinInfo current_rare_bin;
    bool has_rare_bin = false;
    
    for (auto& bin : bins) {
      double freq = static_cast<double>(bin.count) / static_cast<double>(total_count);
      
      if (freq < bin_cutoff) {
        // Merge rare bin into current_rare_bin
        current_rare_bin.merge_with(bin);
        has_rare_bin = true;
      } else {
        // Add accumulated rare bin if it exists
        if (has_rare_bin) {
          merged_bins.push_back(std::move(current_rare_bin));
          current_rare_bin = BinInfo();
          has_rare_bin = false;
        }
        merged_bins.push_back(std::move(bin));
      }
    }
    
    // Add final rare bin if it exists
    if (has_rare_bin) {
      merged_bins.push_back(std::move(current_rare_bin));
    }
    
    bins = std::move(merged_bins);
    
    // Limit number of pre-bins if necessary
    if (static_cast<int>(bins.size()) > max_n_prebins) {
      // Sort by bin size before limiting
      std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
        return a.count > b.count;  // Descending order
      });
      bins.resize(max_n_prebins);
      
      // Resort by positive rate
      for (auto& bin : bins) {
        bin.pos_rate = static_cast<double>(bin.count_pos) / static_cast<double>(std::max(bin.count, 1));
      }
      
      std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
        return a.pos_rate < b.pos_rate;
      });
    }
    
    // Calculate metrics for all bins
    for (auto& bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
    
    // Initialize IV cache
    iv_cache = std::make_unique<IVCache>(bins.size(), bins.size() > 10);
  }
  
  // Enhanced greedy merge with improved tie handling
  void greedyMerge() {
    // Early exit if we already have few enough bins
    if (static_cast<int>(bins.size()) <= max_bins) {
      converged = true;
      return;
    }
    
    double prev_iv = calculateIV(bins);
    double current_iv = prev_iv;
    
    while (static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      double best_merge_score = NEG_INFINITY;
      double second_best_score = NEG_INFINITY;
      size_t best_merge_index = 0;
      size_t second_best_index = 0;
      
      // Enhanced evaluation of merge options
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        // Check cache first
        double cached_score = iv_cache->get(i, i+1);
        if (cached_score >= 0.0) {
          if (cached_score > best_merge_score) {
            second_best_score = best_merge_score;
            second_best_index = best_merge_index;
            best_merge_score = cached_score;
            best_merge_index = i;
          } else if (cached_score > second_best_score) {
            second_best_score = cached_score;
            second_best_index = i;
          }
          continue;
        }
        
        // Simulate merging
        BinInfo merged_bin;
        merged_bin.merge_with(bins[i]);
        merged_bin.merge_with(bins[i+1]);
        merged_bin.calculate_metrics(total_pos, total_neg);
        
        // Create simulated bin configuration
        std::vector<BinInfo> temp_bins;
        temp_bins.reserve(bins.size() - 1);
        
        // Add bins before the merge
        temp_bins.insert(temp_bins.end(), bins.begin(), bins.begin() + i);
        
        // Add merged bin
        temp_bins.push_back(merged_bin);
        
        // Add bins after the merge
        if (i + 2 < bins.size()) {
          temp_bins.insert(temp_bins.end(), bins.begin() + i + 2, bins.end());
        }
        
        // Calculate merge score
        double merge_score = calculateIV(temp_bins);
        iv_cache->set(i, i+1, merge_score);
        
        // Early exit if we find an excellent merge (5% improvement)
        if (merge_score > current_iv * 1.05 && std::isfinite(merge_score)) {
          best_merge_score = merge_score;
          best_merge_index = i;
          break;  // Early exit for significant improvement
        }
        
        // Track best and second best options
        if (merge_score > best_merge_score && std::isfinite(merge_score)) {
          second_best_score = best_merge_score;
          second_best_index = best_merge_index;
          best_merge_score = merge_score;
          best_merge_index = i;
        } else if (merge_score > second_best_score && std::isfinite(merge_score)) {
          second_best_score = merge_score;
          second_best_index = i;
        }
      }
      
      // Tie handling: If best and second best are very close, prefer more balanced bins
      if (std::abs(best_merge_score - second_best_score) < convergence_threshold * 10) {
        int size_diff_best = std::abs(bins[best_merge_index].count - bins[best_merge_index+1].count);
        int size_diff_second = std::abs(bins[second_best_index].count - bins[second_best_index+1].count);
        
        if (size_diff_second < size_diff_best * 0.8) {  // Second option is significantly more balanced
          best_merge_index = second_best_index;
          best_merge_score = second_best_score;
        }
      }
      
      // Execute the best merge
      BinInfo& bin1 = bins[best_merge_index];
      BinInfo& bin2 = bins[best_merge_index + 1];
      
      bin1.merge_with(bin2);
      bin1.calculate_metrics(total_pos, total_neg);
      
      bins.erase(bins.begin() + best_merge_index + 1);
      
      // Invalidate cache for affected rows
      iv_cache->invalidate_row(best_merge_index);
      iv_cache->resize(bins.size());
      
      // Recalculate IV after merging
      current_iv = calculateIV(bins);
      
      // Check convergence
      if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_iv = current_iv;
      iterations_run++;
      
      // Stop if we've reached max_bins
      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }
  }
  
  // Enhanced monotonicity enforcement with gradient relaxation
  void ensureMonotonicity() {
    if (bins.size() <= 1) return;
    
    bool monotonic = false;
    const int max_attempts = static_cast<int>(bins.size() * 3); // Safe limit
    int attempts = 0;
    
    // Calculate average bin WoE gap for context-aware monotonicity check
    double avg_woe_gap = 0.0;
    if (bins.size() > 1) {
      double total_gap = 0.0;
      for (size_t i = 1; i < bins.size(); i++) {
        total_gap += std::abs(bins[i].woe - bins[i-1].woe);
      }
      avg_woe_gap = total_gap / (bins.size() - 1);
    }
    
    // Adaptive threshold based on average gap
    double monotonicity_threshold = std::min(EPSILON, avg_woe_gap * 0.01);
    
    while (!monotonic && static_cast<int>(bins.size()) > min_bins && attempts < max_attempts) {
      monotonic = true;
      
      for (size_t i = 1; i < bins.size(); ++i) {
        // Check if monotonicity is violated with context-aware threshold
        if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
          // Merge bins i-1 and i
          bins[i-1].merge_with(bins[i]);
          bins[i-1].calculate_metrics(total_pos, total_neg);
          bins.erase(bins.begin() + i);
          
          // Recalculate adaptive threshold
          if (bins.size() > 1) {
            double total_gap = 0.0;
            for (size_t j = 1; j < bins.size(); j++) {
              total_gap += std::abs(bins[j].woe - bins[j-1].woe);
            }
            avg_woe_gap = total_gap / (bins.size() - 1);
            monotonicity_threshold = std::min(EPSILON, avg_woe_gap * 0.01);
          }
          
          monotonic = false;
          break;
        }
      }
      
      attempts++;
    }
    
    if (attempts >= max_attempts) {
      Rcpp::warning("Could not ensure monotonicity in %d attempts", max_attempts);
    }
  }
  
  // Efficient category name joining for bin representation
  std::string joinCategoryNames(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    if (categories.size() == 1) return categories[0];
    
    // Estimate total size for pre-allocation
    size_t total_size = 0;
    for (const auto& cat : categories) {
      total_size += cat.size();
    }
    total_size += bin_separator.size() * (categories.size() - 1);
    
    std::string result;
    result.reserve(total_size);
    
    result = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      result += bin_separator;
      result += categories[i];
    }
    
    return result;
  }
  
public:
  OptimalBinningCategoricalGMB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20,
                               std::string bin_separator = "%;%",
                               double convergence_threshold = 1e-6,
                               int max_iterations = 1000)
    : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
      convergence_threshold(convergence_threshold), max_iterations(max_iterations) {
    
    validateInput();
    
    // Adjust max_bins if necessary, efficiently
    std::unordered_set<std::string> unique_cats;
    unique_cats.reserve(std::min(feature.size(), static_cast<size_t>(1024)));
    
    for (const auto& cat : feature) {
      unique_cats.insert(cat);
    }
    
    int ncat = static_cast<int>(unique_cats.size());
    max_bins = std::min(max_bins, ncat);
    min_bins = std::min(min_bins, max_bins);
  }
  
  Rcpp::List fit() {
    // Initialization
    initializeBins();
    
    // Greedy merging
    greedyMerge();
    
    // Monotonicity enforcement
    ensureMonotonicity();
    
    // Prepare result efficiently
    const size_t n_bins = bins.size();
    
    Rcpp::NumericVector ids(n_bins);
    Rcpp::CharacterVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector count_pos_values(n_bins);
    Rcpp::IntegerVector count_neg_values(n_bins);
    
    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = i + 1;
      bin_names[i] = joinCategoryNames(bins[i].categories);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      count_values[i] = bins[i].count;
      count_pos_values[i] = bins[i].count_pos;
      count_neg_values[i] = bins[i].count_neg;
    }
    
    // Calculate total IV for the binning
    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

//' @title Categorical Optimal Binning with Greedy Merge Binning
//'
//' @description
//' Implements optimal binning for categorical variables using a Greedy Merge approach,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator used for merging category names (default: "%;%").
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations (default: 1000).
//'
//' @return A list with the following elements:
//' \itemize{
//'   \item id: Numeric vector of bin identifiers.
//'   \item bin: Character vector of bin names (merged categories).
//'   \item woe: Numeric vector of Weight of Evidence values for each bin.
//'   \item iv: Numeric vector of Information Value for each bin.
//'   \item count: Integer vector of total count for each bin.
//'   \item count_pos: Integer vector of positive class count for each bin.
//'   \item count_neg: Integer vector of negative class count for each bin.
//'   \item total_iv: Total Information Value of the binning.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations performed.
//' }
//'
//' @details
//' The Greedy Merge Binning (GMB) algorithm finds an optimal binning solution by iteratively 
//' merging adjacent bins to maximize Information Value (IV) while respecting constraints 
//' on the number of bins.
//'
//' The Weight of Evidence (WoE) measures the predictive power of a bin and is defined as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{n^+_i/N^+}{n^-_i/N^-}\right)}
//' 
//' where:
//' \itemize{
//'   \item \eqn{n^+_i} is the number of positive cases in bin i
//'   \item \eqn{n^-_i} is the number of negative cases in bin i
//'   \item \eqn{N^+} is the total number of positive cases
//'   \item \eqn{N^-} is the total number of negative cases
//' }
//'
//' The Information Value (IV) quantifies the predictive power of the entire binning and is:
//'
//' \deqn{IV = \sum_{i=1}^{n} (p_i - q_i) \times WoE_i}
//'
//' where:
//' \itemize{
//'   \item \eqn{p_i = n^+_i/N^+} is the proportion of positive cases in bin i
//'   \item \eqn{q_i = n^-_i/N^-} is the proportion of negative cases in bin i
//' }
//'
//' This algorithm applies Bayesian smoothing to WoE calculations to improve stability, particularly
//' with small sample sizes or rare categories. The smoothing applies pseudo-counts based on the
//' overall population prevalence.
//'
//' The algorithm includes the following main steps:
//' \enumerate{
//'   \item Initialize bins with each unique category.
//'   \item Merge rare categories based on the bin_cutoff.
//'   \item Iteratively merge adjacent bins that result in the highest IV.
//'   \item Stop merging when the number of bins reaches min_bins or max_bins.
//'   \item Ensure monotonicity of WoE values across bins.
//'   \item Calculate final WoE and IV for each bin.
//' }
//'
//' Edge cases are handled as follows:
//' \itemize{
//'   \item Empty strings in feature are rejected during input validation
//'   \item Extremely imbalanced datasets (< 5 samples in either class) produce a warning
//'   \item When merging bins, ties in IV improvement are resolved by preferring more balanced bins
//'   \item Monotonicity violations are addressed with an adaptive threshold based on average WoE gaps
//' }
//'
//' @examples
//' \dontrun{
//' # Example data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_gmb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result)
//' }
//'
//' @author
//' Lopes, J. E.
//'
//' @references
//' \itemize{
//'   \item Beltrami, M., Mach, M., & Dall'Aglio, M. (2021). Monotonic Optimal Binning Algorithm for Credit Risk Modeling. Risks, 9(3), 58.
//'   \item Siddiqi, N. (2006). Credit risk scorecards: developing and implementing intelligent credit scoring (Vol. 3). John Wiley & Sons.
//'   \item García-Magariño, I., Medrano, C., Lombas, A. S., & Barrasa, A. (2019). A hybrid approach with agent-based simulation and clustering for sociograms. Information Sciences, 499, 47-61.
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
//'   \item Lin, X., Wang, G., & Zhang, T. (2022). Efficient monotonic binning for predictive modeling in high-dimensional spaces. Knowledge-Based Systems, 235, 107629.
//'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//' }
//' @export
//'
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 // Preliminary validations with improved error messages
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Input vectors cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length (got %d and %d).", 
              feature.size(), target.size());
 }
 
 // Optimized conversion of R vectors to C++
 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;
 
 feature_vec.reserve(feature.size());
 target_vec.reserve(target.size());
 
 // Count NAs for more informative error messages
 int na_feature_count = 0;
 int na_target_count = 0;
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NA in feature
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
     na_feature_count++;
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
   
   // Handle NA in target
   if (IntegerVector::is_na(target[i])) {
     na_target_count++;
     Rcpp::stop("Target cannot contain missing values. Found NA at position %d.", i+1);
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
   OptimalBinningCategoricalGMB binner(feature_vec, target_vec, min_bins, max_bins, 
                                       bin_cutoff, max_n_prebins, bin_separator, 
                                       convergence_threshold, max_iterations);
   return binner.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: %s", e.what());
 }
}

