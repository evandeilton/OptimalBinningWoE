#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>
#include <sstream>
#include <numeric>

using namespace Rcpp;

// Enumeration for monotonicity direction
enum class MonotonicityDirection {
  NONE,           // No monotonicity constraint
  INCREASING,     // WoE must be non-decreasing
  DECREASING,     // WoE must be non-increasing
  AUTO            // Automatically determine direction based on correlation
};

// Helper function to convert string to MonotonicityDirection
MonotonicityDirection string_to_monotonicity_direction(const std::string& str) {
  if (str == "increasing") return MonotonicityDirection::INCREASING;
  if (str == "decreasing") return MonotonicityDirection::DECREASING;
  if (str == "auto") return MonotonicityDirection::AUTO;
  return MonotonicityDirection::NONE; // Default
}

// Enhanced bin structure with additional statistics
struct Bin {
  double lower;              // Lower bound of the bin
  double upper;              // Upper bound of the bin
  double woe;                // Weight of Evidence
  double iv;                 // Information Value contribution
  double event_rate;         // Event rate in the bin
  int count;                 // Total observations in the bin
  int count_pos;             // Positive observations in the bin
  int count_neg;             // Negative observations in the bin
  
  // Constructor with default initialization
  Bin() : lower(0.0), upper(0.0), woe(0.0), iv(0.0), event_rate(0.0),
  count(0), count_pos(0), count_neg(0) {}
};

// Comparator for sorting bins
bool compareBins(const Bin &a, const Bin &b) {
  return a.lower < b.lower;
}

// Forward declarations
Rcpp::List handle_few_unique_values(
    const std::vector<double>& feature_vec,
    const std::vector<int>& target_vec,
    double laplace_smoothing
);

// Class for Optimal Binning with enhanced features
class OptimalBinningNumericalUDT {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double laplace_smoothing;
  MonotonicityDirection monotonicity_direction;
  std::vector<Bin> bins;
  Bin missing_bin; // Not mutable, we'll handle it differently
  double convergence_threshold;
  int max_iterations;
  bool converged;
  int iterations_run;
  double total_iv;
  double gini_index;
  double ks_statistic;
  
public:
  OptimalBinningNumericalUDT(
    std::vector<double> feature,
    std::vector<int> target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    double laplace_smoothing = 0.5,
    MonotonicityDirection monotonicity_direction = MonotonicityDirection::NONE,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000) 
    : feature(std::move(feature)), 
      target(std::move(target)), 
      min_bins(std::max(2, min_bins)),
      max_bins(std::max(this->min_bins, max_bins)),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      laplace_smoothing(laplace_smoothing),
      monotonicity_direction(monotonicity_direction),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0),
      total_iv(0.0),
      gini_index(0.0),
      ks_statistic(0.0) {
    // Initialize missing_bin
    missing_bin.lower = std::numeric_limits<double>::quiet_NaN();
    missing_bin.upper = std::numeric_limits<double>::quiet_NaN();
  }
  
  void fit() {
    // Handle missing values
    std::vector<double> feature_clean;
    std::vector<int> target_clean;
    handle_missing_values(feature, target, feature_clean, target_clean, missing_bin);
    
    // Auto-detect monotonicity if needed
    if (monotonicity_direction == MonotonicityDirection::AUTO) {
      double correlation = calculate_correlation(feature_clean, target_clean);
      monotonicity_direction = (correlation >= 0) ? 
      MonotonicityDirection::INCREASING : 
        MonotonicityDirection::DECREASING;
    }
    
    // Initial binning using entropy-based approach
    std::vector<double> cut_points = get_entropy_based_cutpoints(feature_clean, target_clean, max_n_prebins);
    
    // Initial binning
    bins = initial_binning(feature_clean, target_clean, cut_points);
    
    // Rare bin merging
    merge_rare_bins();
    
    // Bin optimization
    optimize_bins();
    
    // Calculate final statistics for both regular bins and missing bin
    calculate_woe_iv_all();
    total_iv = calculate_total_iv();
    gini_index = calculate_gini_index();
    ks_statistic = calculate_ks_statistic();
  }
  
  Rcpp::List get_result() const {
    size_t n_bins = bins.size();
    CharacterVector bin_intervals(n_bins);
    NumericVector woe_values(n_bins), iv_values(n_bins), event_rates(n_bins);
    IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
    NumericVector cutpoints(n_bins > 1 ? n_bins - 1 : 0);
    
    for (size_t i = 0; i < n_bins; ++i) {
      bin_intervals[i] = create_interval(bins[i].lower, bins[i].upper);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      event_rates[i] = bins[i].event_rate;
      counts[i] = bins[i].count;
      counts_pos[i] = bins[i].count_pos;
      counts_neg[i] = bins[i].count_neg;
      if (i < n_bins - 1) {
        cutpoints[i] = bins[i].upper;
      }
    }
    
    // Handle missing values bin if it has data
    bool has_missing = missing_bin.count > 0;
    CharacterVector all_bin_intervals = bin_intervals;
    NumericVector all_woe_values = woe_values;
    NumericVector all_iv_values = iv_values;
    NumericVector all_event_rates = event_rates;
    IntegerVector all_counts = counts;
    IntegerVector all_counts_pos = counts_pos;
    IntegerVector all_counts_neg = counts_neg;
    
    if (has_missing) {
      // Add missing bin information
      all_bin_intervals = CharacterVector(n_bins + 1);
      all_woe_values = NumericVector(n_bins + 1);
      all_iv_values = NumericVector(n_bins + 1);
      all_event_rates = NumericVector(n_bins + 1);
      all_counts = IntegerVector(n_bins + 1);
      all_counts_pos = IntegerVector(n_bins + 1);
      all_counts_neg = IntegerVector(n_bins + 1);
      
      // Add regular bins
      for (size_t i = 0; i < n_bins; ++i) {
        all_bin_intervals[i] = bin_intervals[i];
        all_woe_values[i] = woe_values[i];
        all_iv_values[i] = iv_values[i];
        all_event_rates[i] = event_rates[i];
        all_counts[i] = counts[i];
        all_counts_pos[i] = counts_pos[i];
        all_counts_neg[i] = counts_neg[i];
      }
      
      // Add missing bin at the end
      all_bin_intervals[n_bins] = "NA";
      all_woe_values[n_bins] = missing_bin.woe;
      all_iv_values[n_bins] = missing_bin.iv;
      all_event_rates[n_bins] = missing_bin.event_rate;
      all_counts[n_bins] = missing_bin.count;
      all_counts_pos[n_bins] = missing_bin.count_pos;
      all_counts_neg[n_bins] = missing_bin.count_neg;
    }
    
    // Create vector of IDs with the same size as bins (maintaining compatibility)
    Rcpp::NumericVector ids(all_bin_intervals.size());
    for(int i = 0; i < all_bin_intervals.size(); i++) {
      ids[i] = i + 1;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = all_bin_intervals,
      Rcpp::Named("woe") = all_woe_values,
      Rcpp::Named("iv") = all_iv_values,
      Rcpp::Named("event_rate") = all_event_rates,
      Rcpp::Named("count") = all_counts,
      Rcpp::Named("count_pos") = all_counts_pos,
      Rcpp::Named("count_neg") = all_counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("gini") = gini_index,
      Rcpp::Named("ks") = ks_statistic,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
  
  // Main method to calculate WoE and IV for regular bins
  // Does not modify the missing bin - that's handled separately in calculate_woe_iv_all
  void calculate_woe_iv(std::vector<Bin> &bins, double laplace_smoothing = 0.5) {
    int total_pos = 0, total_neg = 0;
    for (const auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    double total_pos_d = static_cast<double>(total_pos);
    double total_neg_d = static_cast<double>(total_neg);
    
    for (auto &bin : bins) {
      // Apply Laplace smoothing to handle zero counts
      double dist_pos = (static_cast<double>(bin.count_pos) + laplace_smoothing) / 
        (total_pos_d + laplace_smoothing * bins.size());
      double dist_neg = (static_cast<double>(bin.count_neg) + laplace_smoothing) / 
        (total_neg_d + laplace_smoothing * bins.size());
      
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
      
      // Calculate event rate
      bin.event_rate = bin.count > 0 ? 
      static_cast<double>(bin.count_pos) / static_cast<double>(bin.count) : 0.0;
    }
  }
  
  // New method to calculate WoE and IV for all bins including missing bin
  void calculate_woe_iv_all() {
    // First calculate for regular bins
    calculate_woe_iv(bins, laplace_smoothing);
    
    // Then handle missing bin separately (if it has data)
    if (missing_bin.count > 0) {
      int total_pos = 0, total_neg = 0;
      
      // Calculate totals including both regular bins and missing bin
      for (const auto &bin : bins) {
        total_pos += bin.count_pos;
        total_neg += bin.count_neg;
      }
      total_pos += missing_bin.count_pos;
      total_neg += missing_bin.count_neg;
      
      double total_pos_d = static_cast<double>(total_pos);
      double total_neg_d = static_cast<double>(total_neg);
      
      // Calculate WoE and IV for missing bin using the same formula
      double dist_pos_missing = (static_cast<double>(missing_bin.count_pos) + laplace_smoothing) / 
        (total_pos_d + laplace_smoothing * (bins.size() + 1));
      double dist_neg_missing = (static_cast<double>(missing_bin.count_neg) + laplace_smoothing) / 
        (total_neg_d + laplace_smoothing * (bins.size() + 1));
      
      missing_bin.woe = std::log(dist_pos_missing / dist_neg_missing);
      missing_bin.iv = (dist_pos_missing - dist_neg_missing) * missing_bin.woe;
      missing_bin.event_rate = missing_bin.count > 0 ? 
      static_cast<double>(missing_bin.count_pos) / static_cast<double>(missing_bin.count) : 0.0;
    }
  }
  
  // This is a pure function that doesn't modify state and should be const
  std::string create_interval(double lower, double upper) const {
    std::stringstream ss;
    ss << "(";
    if (std::isinf(lower) && lower < 0) {
      ss << "-Inf";
    } else if (std::isinf(lower)) {
      ss << "Inf";
    } else {
      ss << lower;
    }
    ss << ";";
    if (std::isinf(upper) && upper > 0) {
      ss << "Inf";
    } else if (std::isinf(upper)) {
      ss << "-Inf";
    } else {
      ss << upper;
    }
    ss << "]";
    return ss.str();
  }
  
private:
  void handle_missing_values(
      const std::vector<double> &feature, 
      const std::vector<int> &target, 
      std::vector<double> &feature_clean,
      std::vector<int> &target_clean,
      Bin &missing_bin) {
    
    missing_bin.count = 0;
    missing_bin.count_pos = 0;
    missing_bin.count_neg = 0;
    
    feature_clean.reserve(feature.size());
    target_clean.reserve(feature.size());
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (std::isnan(feature[i]) || std::isinf(feature[i])) {
        missing_bin.count++;
        if (target[i] == 1) {
          missing_bin.count_pos++;
        } else {
          missing_bin.count_neg++;
        }
      } else {
        feature_clean.push_back(feature[i]);
        target_clean.push_back(target[i]);
      }
    }
  }
  
  double calculate_correlation(const std::vector<double> &x, const std::vector<int> &y) {
    if (x.empty() || y.empty() || x.size() != y.size()) {
      return 0.0;
    }
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
    double sum_x2 = 0.0, sum_y2 = 0.0;
    size_t n = x.size();
    
    for (size_t i = 0; i < n; ++i) {
      double x_i = x[i];
      double y_i = static_cast<double>(y[i]);
      
      sum_x += x_i;
      sum_y += y_i;
      sum_xy += x_i * y_i;
      sum_x2 += x_i * x_i;
      sum_y2 += y_i * y_i;
    }
    
    double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
    if (std::abs(denominator) < 1e-10) {
      return 0.0;
    }
    
    return (n * sum_xy - sum_x * sum_y) / denominator;
  }
  
  double calculate_entropy(const std::vector<int> &y) {
    if (y.empty()) {
      return 0.0;
    }
    
    int count = y.size();
    int count_pos = std::accumulate(y.begin(), y.end(), 0);
    int count_neg = count - count_pos;
    
    double p_pos = static_cast<double>(count_pos) / count;
    double p_neg = static_cast<double>(count_neg) / count;
    
    double entropy = 0.0;
    if (p_pos > 0) entropy -= p_pos * std::log2(p_pos);
    if (p_neg > 0) entropy -= p_neg * std::log2(p_neg);
    
    return entropy;
  }
  
  double calculate_info_gain(const std::vector<double> &x, const std::vector<int> &y, 
                             double split_point) {
    if (x.empty() || y.empty() || x.size() != y.size()) {
      return 0.0;
    }
    
    std::vector<int> left_y, right_y;
    
    for (size_t i = 0; i < x.size(); ++i) {
      if (x[i] <= split_point) {
        left_y.push_back(y[i]);
      } else {
        right_y.push_back(y[i]);
      }
    }
    
    if (left_y.empty() || right_y.empty()) {
      return 0.0;
    }
    
    double entropy_before = calculate_entropy(y);
    double entropy_left = calculate_entropy(left_y);
    double entropy_right = calculate_entropy(right_y);
    
    double p_left = static_cast<double>(left_y.size()) / y.size();
    double p_right = static_cast<double>(right_y.size()) / y.size();
    
    return entropy_before - (p_left * entropy_left + p_right * entropy_right);
  }
  
  std::vector<double> get_entropy_based_cutpoints(
      const std::vector<double> &x, 
      const std::vector<int> &y, 
      int max_cuts) {
    
    if (x.empty() || y.empty() || x.size() != y.size() || max_cuts < 1) {
      return {};
    }
    
    // Create pairs of (x, y) for sorting
    std::vector<std::pair<double, int>> data_pairs;
    data_pairs.reserve(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
      data_pairs.push_back({x[i], y[i]});
    }
    
    // Sort by x value
    std::sort(data_pairs.begin(), data_pairs.end());
    
    // Extract unique x values as potential split points
    std::vector<double> unique_values;
    for (size_t i = 1; i < data_pairs.size(); ++i) {
      if (data_pairs[i].first > data_pairs[i-1].first) {
        unique_values.push_back((data_pairs[i].first + data_pairs[i-1].first) / 2.0);
      }
    }
    
    if (unique_values.empty()) {
      return {};
    }
    
    // If we have fewer unique values than requested cuts, return all
    if (unique_values.size() <= static_cast<size_t>(max_cuts)) {
      return unique_values;
    }
    
    // Calculate information gain for each potential split point
    std::vector<std::pair<double, double>> info_gains; // (split_point, info_gain)
    for (double split_point : unique_values) {
      double gain = calculate_info_gain(x, y, split_point);
      info_gains.push_back({split_point, gain});
    }
    
    // Sort by information gain in descending order
    std::sort(info_gains.begin(), info_gains.end(), 
              [](const auto &a, const auto &b) { return a.second > b.second; });
    
    // Take top max_cuts points
    std::vector<double> cutpoints;
    cutpoints.reserve(max_cuts);
    for (size_t i = 0; i < std::min(static_cast<size_t>(max_cuts), info_gains.size()); ++i) {
      cutpoints.push_back(info_gains[i].first);
    }
    
    // Sort cutpoints in ascending order
    std::sort(cutpoints.begin(), cutpoints.end());
    
    return cutpoints;
  }
  
  std::vector<Bin> initial_binning(
      const std::vector<double> &feature,
      const std::vector<int> &target,
      const std::vector<double> &cut_points) {
    
    if (cut_points.empty()) {
      // If no cut points, create a single bin
      Bin single_bin;
      single_bin.lower = -std::numeric_limits<double>::infinity();
      single_bin.upper = std::numeric_limits<double>::infinity();
      single_bin.count = feature.size();
      single_bin.count_pos = 0;
      single_bin.count_neg = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        if (target[i] == 1) single_bin.count_pos++;
        else single_bin.count_neg++;
      }
      
      return {single_bin};
    }
    
    std::vector<Bin> initial_bins;
    std::vector<double> boundaries = cut_points;
    boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
    boundaries.push_back(std::numeric_limits<double>::infinity());
    size_t n_bins = boundaries.size() - 1;
    
    initial_bins.resize(n_bins);
    for (size_t i = 0; i < n_bins; ++i) {
      initial_bins[i].lower = boundaries[i];
      initial_bins[i].upper = boundaries[i + 1];
      initial_bins[i].count = 0;
      initial_bins[i].count_pos = 0;
      initial_bins[i].count_neg = 0;
    }
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int bin_idx = find_bin_index(val, boundaries);
      if (bin_idx >= 0 && static_cast<size_t>(bin_idx) < n_bins) {
        initial_bins[bin_idx].count++;
        if (target[i] == 1) {
          initial_bins[bin_idx].count_pos++;
        } else {
          initial_bins[bin_idx].count_neg++;
        }
      }
    }
    
    // Calculate initial statistics
    calculate_woe_iv(initial_bins, laplace_smoothing);
    
    return initial_bins;
  }
  
  int find_bin_index(double value, const std::vector<double> &boundaries) {
    // Optimized binary search to find bin index
    int left = 0;
    int right = boundaries.size() - 1;
    
    while (left < right - 1) {
      int mid = left + (right - left) / 2;
      if (value <= boundaries[mid]) {
        right = mid;
      } else {
        left = mid;
      }
    }
    
    return left;
  }
  
  void merge_rare_bins() {
    size_t total_count = 0;
    for (const auto &bin : bins) {
      total_count += bin.count;
    }
    
    double cutoff_count = bin_cutoff * static_cast<double>(total_count);
    bool merged = false;
    
    do {
      merged = false;
      for (size_t i = 0; i < bins.size(); ++i) {
        if (static_cast<double>(bins[i].count) < cutoff_count && 
            bins.size() > static_cast<size_t>(min_bins)) {
          
          // Find best candidate for merging (prefer adjacent bins with similar event rates)
          size_t merge_idx = i;
          double min_diff = std::numeric_limits<double>::max();
          
          if (i > 0) {
            double diff = std::abs(bins[i].event_rate - bins[i-1].event_rate);
            if (diff < min_diff) {
              min_diff = diff;
              merge_idx = i - 1;
            }
          }
          
          if (i < bins.size() - 1) {
            double diff = std::abs(bins[i].event_rate - bins[i+1].event_rate);
            if (diff < min_diff) {
              merge_idx = i;
            }
          }
          
          // Perform merge
          if (merge_idx == i && i < bins.size() - 1) {
            bins[i] = merge_bins(bins[i], bins[i + 1]);
            bins.erase(bins.begin() + i + 1);
          } else {
            bins[merge_idx] = merge_bins(bins[merge_idx], bins[merge_idx + 1]);
            bins.erase(bins.begin() + merge_idx + 1);
          }
          
          merged = true;
          break;
        }
      }
      
      if (merged) {
        calculate_woe_iv(bins, laplace_smoothing);
      }
    } while (merged && bins.size() > static_cast<size_t>(min_bins));
  }
  
  Bin merge_bins(const Bin &bin1, const Bin &bin2) {
    Bin merged;
    merged.lower = bin1.lower;
    merged.upper = bin2.upper;
    merged.count = bin1.count + bin2.count;
    merged.count_pos = bin1.count_pos + bin2.count_pos;
    merged.count_neg = bin1.count_neg + bin2.count_neg;
    
    // Calculate event rate for the merged bin
    merged.event_rate = merged.count > 0 ? 
    static_cast<double>(merged.count_pos) / static_cast<double>(merged.count) : 0.0;
    
    return merged;
  }
  
  void optimize_bins() {
    double prev_total_iv = 0.0;
    iterations_run = 0;
    
    while (iterations_run < max_iterations) {
      // Ensure number of bins within max_bins
      while (static_cast<int>(bins.size()) > max_bins) {
        merge_optimal_bins();
      }
      
      // Enforce monotonicity if requested
      if (monotonicity_direction != MonotonicityDirection::NONE) {
        enforce_monotonicity();
      }
      
      // Ensure minimum number of bins
      while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
        split_optimal_bin();
      }
      
      // Calculate total IV
      double current_iv = 0.0;
      for (const auto &bin : bins) {
        current_iv += bin.iv;
      }
      
      // Check convergence
      if (std::abs(current_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = current_iv;
      iterations_run++;
    }
  }
  
  void merge_optimal_bins() {
    if (bins.size() <= 2) return;
    
    size_t best_merge_idx = 0;
    double min_iv_loss = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      // Calculate IV before merge
      double iv_before = bins[i].iv + bins[i + 1].iv;
      
      // Calculate IV after merge
      Bin merged = merge_bins(bins[i], bins[i + 1]);
      std::vector<Bin> temp_bins = {merged};
      calculate_woe_iv(temp_bins, laplace_smoothing);
      double iv_after = temp_bins[0].iv;
      
      // Calculate IV loss
      double iv_loss = iv_before - iv_after;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        best_merge_idx = i;
      }
    }
    
    // Perform the optimal merge
    bins[best_merge_idx] = merge_bins(bins[best_merge_idx], bins[best_merge_idx + 1]);
    bins.erase(bins.begin() + best_merge_idx + 1);
    calculate_woe_iv(bins, laplace_smoothing);
  }
  
  void enforce_monotonicity() {
    bool is_monotonic = false;
    
    while (!is_monotonic && bins.size() > static_cast<size_t>(min_bins)) {
      is_monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        bool violates_monotonicity = 
          (monotonicity_direction == MonotonicityDirection::INCREASING && bins[i].woe < bins[i - 1].woe) ||
          (monotonicity_direction == MonotonicityDirection::DECREASING && bins[i].woe > bins[i - 1].woe);
        
        if (violates_monotonicity) {
          // Merge bins that violate monotonicity
          bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
          bins.erase(bins.begin() + i);
          is_monotonic = false;
          break;
        }
      }
      
      if (!is_monotonic) {
        calculate_woe_iv(bins, laplace_smoothing);
      }
    }
  }
  
  void split_optimal_bin() {
    if (bins.empty()) return;
    
    // Find the bin with highest internal variance to split
    size_t best_bin_idx = 0;
    double max_improvement = -std::numeric_limits<double>::max();
    double best_split_point = 0.0;
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const Bin &current_bin = bins[i];
      
      // Skip bins with too few observations
      if (current_bin.count < 10) continue;
      
      // Find observations in this bin
      std::vector<double> bin_features;
      std::vector<int> bin_targets;
      
      for (size_t j = 0; j < feature.size(); ++j) {
        if (feature[j] >= current_bin.lower && feature[j] < current_bin.upper) {
          bin_features.push_back(feature[j]);
          bin_targets.push_back(target[j]);
        }
      }
      
      if (bin_features.size() < 10) continue;
      
      // Find best split point within this bin
      double entropy_before = calculate_entropy(bin_targets);
      
      // Create pairs for sorting
      std::vector<std::pair<double, int>> pairs;
      for (size_t j = 0; j < bin_features.size(); ++j) {
        pairs.push_back({bin_features[j], bin_targets[j]});
      }
      std::sort(pairs.begin(), pairs.end());
      
      // Find potential split points (unique values)
      std::vector<double> split_candidates;
      for (size_t j = 1; j < pairs.size(); ++j) {
        if (pairs[j].first > pairs[j-1].first) {
          split_candidates.push_back((pairs[j].first + pairs[j-1].first) / 2.0);
        }
      }
      
      // Find best split point
      double max_gain = 0.0;
      double best_point = 0.0;
      
      for (double split_point : split_candidates) {
        double gain = calculate_info_gain(bin_features, bin_targets, split_point);
        if (gain > max_gain) {
          max_gain = gain;
          best_point = split_point;
        }
      }
      
      if (max_gain > max_improvement) {
        max_improvement = max_gain;
        best_bin_idx = i;
        best_split_point = best_point;
      }
    }
    
    // If we found a good split
    if (max_improvement > 0) {
      // Create two new bins
      Bin left_bin = bins[best_bin_idx];
      Bin right_bin = bins[best_bin_idx];
      
      left_bin.upper = best_split_point;
      right_bin.lower = best_split_point;
      
      // Redistribute observations
      left_bin.count = 0;
      left_bin.count_pos = 0;
      left_bin.count_neg = 0;
      
      right_bin.count = 0;
      right_bin.count_pos = 0;
      right_bin.count_neg = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        double val = feature[i];
        if (val >= left_bin.lower && val < left_bin.upper) {
          left_bin.count++;
          if (target[i] == 1) left_bin.count_pos++;
          else left_bin.count_neg++;
        } else if (val >= right_bin.lower && val < right_bin.upper) {
          right_bin.count++;
          if (target[i] == 1) right_bin.count_pos++;
          else right_bin.count_neg++;
        }
      }
      
      // Replace original bin with two new bins
      bins[best_bin_idx] = left_bin;
      bins.insert(bins.begin() + best_bin_idx + 1, right_bin);
      
      // Update statistics
      calculate_woe_iv(bins, laplace_smoothing);
    }
  }
  
  double calculate_total_iv() {
    double total = 0.0;
    for (const auto &bin : bins) {
      total += bin.iv;
    }
    if (missing_bin.count > 0) {
      total += missing_bin.iv;
    }
    return total;
  }
  
  double calculate_gini_index() {
    int total_pos = 0, total_neg = 0;
    for (const auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    if (missing_bin.count > 0) {
      total_pos += missing_bin.count_pos;
      total_neg += missing_bin.count_neg;
    }
    
    if (total_pos == 0 || total_neg == 0) {
      return 0.0;
    }
    
    double gini = 0.0;
    double cum_pos = 0.0, cum_neg = 0.0;
    
    // Sort bins by WoE for Gini calculation
    std::vector<Bin> sorted_bins = bins;
    std::sort(sorted_bins.begin(), sorted_bins.end(), 
              [](const Bin &a, const Bin &b) { return a.woe < b.woe; });
    
    // Add missing bin if present
    if (missing_bin.count > 0) {
      sorted_bins.push_back(missing_bin);
      std::sort(sorted_bins.begin(), sorted_bins.end(), 
                [](const Bin &a, const Bin &b) { return a.woe < b.woe; });
    }
    
    for (const auto &bin : sorted_bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      
      cum_pos += pos_rate;
      cum_neg += neg_rate;
      
      gini += pos_rate * (cum_neg - 0.5 * neg_rate);
    }
    
    return 2.0 * gini;
  }
  
  double calculate_ks_statistic() {
    int total_pos = 0, total_neg = 0;
    for (const auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    if (missing_bin.count > 0) {
      total_pos += missing_bin.count_pos;
      total_neg += missing_bin.count_neg;
    }
    
    if (total_pos == 0 || total_neg == 0) {
      return 0.0;
    }
    
    double max_diff = 0.0;
    double cum_pos = 0.0, cum_neg = 0.0;
    
    // Sort bins by WoE for KS calculation
    std::vector<Bin> sorted_bins = bins;
    std::sort(sorted_bins.begin(), sorted_bins.end(), 
              [](const Bin &a, const Bin &b) { return a.woe < b.woe; });
    
    // Add missing bin if present
    if (missing_bin.count > 0) {
      sorted_bins.push_back(missing_bin);
      std::sort(sorted_bins.begin(), sorted_bins.end(), 
                [](const Bin &a, const Bin &b) { return a.woe < b.woe; });
    }
    
    for (const auto &bin : sorted_bins) {
      cum_pos += static_cast<double>(bin.count_pos) / total_pos;
      cum_neg += static_cast<double>(bin.count_neg) / total_neg;
      
      double diff = std::abs(cum_pos - cum_neg);
      if (diff > max_diff) {
        max_diff = diff;
      }
    }
    
    return max_diff;
  }
};

// Implementation of handle_few_unique_values
Rcpp::List handle_few_unique_values(
    const std::vector<double>& feature_vec,
    const std::vector<int>& target_vec,
    double laplace_smoothing
) {
  std::vector<Bin> unique_bins;
  std::unordered_set<double> unique_values_set;
  
  // Handle potential NaN values
  for (double val : feature_vec) {
    if (!std::isnan(val) && !std::isinf(val)) {
      unique_values_set.insert(val);
    }
  }
  
  size_t unique_size = unique_values_set.size();
  
  if (unique_size == 0) {
    // All values are NaN/Inf, create a single bin for missing values
    Bin b;
    b.lower = std::numeric_limits<double>::quiet_NaN();
    b.upper = std::numeric_limits<double>::quiet_NaN();
    b.count = feature_vec.size();
    b.count_pos = 0;
    b.count_neg = 0;
    
    for (size_t i = 0; i < feature_vec.size(); ++i) {
      if (target_vec[i] == 1) b.count_pos++;
      else b.count_neg++;
    }
    
    b.event_rate = b.count > 0 ? 
    static_cast<double>(b.count_pos) / static_cast<double>(b.count) : 0.0;
    
    unique_bins.push_back(b);
  } else if (unique_size == 1) {
    // Only one unique value, single bin
    double unique_val = *unique_values_set.begin();
    
    Bin b;
    b.lower = -std::numeric_limits<double>::infinity();
    b.upper = std::numeric_limits<double>::infinity();
    b.count = 0;
    b.count_pos = 0;
    b.count_neg = 0;
    
    for (size_t i = 0; i < feature_vec.size(); ++i) {
      if (!std::isnan(feature_vec[i]) && !std::isinf(feature_vec[i])) {
        b.count++;
        if (target_vec[i] == 1) b.count_pos++;
        else b.count_neg++;
      }
    }
    
    b.event_rate = b.count > 0 ? 
    static_cast<double>(b.count_pos) / static_cast<double>(b.count) : 0.0;
    
    unique_bins.push_back(b);
  } else if (unique_size == 2) {
    // Two unique values, create two bins with min value as cutpoint
    std::vector<double> unique_values(unique_values_set.begin(), unique_values_set.end());
    std::sort(unique_values.begin(), unique_values.end());
    
    double min_val = unique_values[0];
    double max_val = unique_values[1];
    double cutpoint = (min_val + max_val) / 2.0;
    
    Bin bin1, bin2;
    
    bin1.lower = -std::numeric_limits<double>::infinity();
    bin1.upper = cutpoint;
    bin1.count = 0;
    bin1.count_pos = 0;
    bin1.count_neg = 0;
    
    bin2.lower = cutpoint;
    bin2.upper = std::numeric_limits<double>::infinity();
    bin2.count = 0;
    bin2.count_pos = 0;
    bin2.count_neg = 0;
    
    for (size_t i = 0; i < feature_vec.size(); ++i) {
      double val = feature_vec[i];
      if (!std::isnan(val) && !std::isinf(val)) {
        if (val <= min_val) {
          bin1.count++;
          if (target_vec[i] == 1) bin1.count_pos++;
          else bin1.count_neg++;
        } else {
          bin2.count++;
          if (target_vec[i] == 1) bin2.count_pos++;
          else bin2.count_neg++;
        }
      }
    }
    
    bin1.event_rate = bin1.count > 0 ? 
    static_cast<double>(bin1.count_pos) / static_cast<double>(bin1.count) : 0.0;
    
    bin2.event_rate = bin2.count > 0 ? 
    static_cast<double>(bin2.count_pos) / static_cast<double>(bin2.count) : 0.0;
    
    unique_bins.push_back(bin1);
    unique_bins.push_back(bin2);
  }
  
  // Handle missing values
  Bin missing_bin;
  missing_bin.lower = std::numeric_limits<double>::quiet_NaN();
  missing_bin.upper = std::numeric_limits<double>::quiet_NaN();
  missing_bin.count = 0;
  missing_bin.count_pos = 0;
  missing_bin.count_neg = 0;
  
  for (size_t i = 0; i < feature_vec.size(); ++i) {
    if (std::isnan(feature_vec[i]) || std::isinf(feature_vec[i])) {
      missing_bin.count++;
      if (target_vec[i] == 1) missing_bin.count_pos++;
      else missing_bin.count_neg++;
    }
  }
  
  missing_bin.event_rate = missing_bin.count > 0 ? 
  static_cast<double>(missing_bin.count_pos) / static_cast<double>(missing_bin.count) : 0.0;
  
  // Calculate WOE and IV separately - not using the class to avoid const issues
  int total_pos = 0, total_neg = 0;
  
  for (const auto &bin : unique_bins) {
    total_pos += bin.count_pos;
    total_neg += bin.count_neg;
  }
  
  if (missing_bin.count > 0) {
    total_pos += missing_bin.count_pos;
    total_neg += missing_bin.count_neg;
  }
  
  double total_pos_d = static_cast<double>(total_pos);
  double total_neg_d = static_cast<double>(total_neg);
  
  for (auto &bin : unique_bins) {
    double dist_pos = (static_cast<double>(bin.count_pos) + laplace_smoothing) / 
      (total_pos_d + laplace_smoothing * (unique_bins.size() + (missing_bin.count > 0 ? 1 : 0)));
    double dist_neg = (static_cast<double>(bin.count_neg) + laplace_smoothing) / 
      (total_neg_d + laplace_smoothing * (unique_bins.size() + (missing_bin.count > 0 ? 1 : 0)));
    
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
  
  if (missing_bin.count > 0) {
    double dist_pos_missing = (static_cast<double>(missing_bin.count_pos) + laplace_smoothing) / 
      (total_pos_d + laplace_smoothing * (unique_bins.size() + 1));
    double dist_neg_missing = (static_cast<double>(missing_bin.count_neg) + laplace_smoothing) / 
      (total_neg_d + laplace_smoothing * (unique_bins.size() + 1));
    
    missing_bin.woe = std::log(dist_pos_missing / dist_neg_missing);
    missing_bin.iv = (dist_pos_missing - dist_neg_missing) * missing_bin.woe;
  }
  
  // Calculate Gini and KS statistic
  double gini = 0.0;
  double ks = 0.0;
  
  if (!unique_bins.empty()) {
    if (total_pos > 0 && total_neg > 0) {
      std::vector<Bin> all_bins = unique_bins;
      if (missing_bin.count > 0) {
        all_bins.push_back(missing_bin);
      }
      
      std::sort(all_bins.begin(), all_bins.end(), 
                [](const Bin &a, const Bin &b) { return a.woe < b.woe; });
      
      // Calculate Gini
      double cum_pos = 0.0, cum_neg = 0.0;
      for (const auto &bin : all_bins) {
        double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
        double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
        
        cum_pos += pos_rate;
        cum_neg += neg_rate;
        
        gini += pos_rate * (cum_neg - 0.5 * neg_rate);
      }
      gini = 2.0 * gini;
      
      // Calculate KS
      cum_pos = 0.0;
      cum_neg = 0.0;
      for (const auto &bin : all_bins) {
        cum_pos += static_cast<double>(bin.count_pos) / total_pos;
        cum_neg += static_cast<double>(bin.count_neg) / total_neg;
        
        double diff = std::abs(cum_pos - cum_neg);
        if (diff > ks) {
          ks = diff;
        }
      }
    }
  }
  
  // Helper function to create interval string - implementation must match the class method
  auto create_interval = [](double lower, double upper) -> std::string {
    std::stringstream ss;
    ss << "(";
    if (std::isinf(lower) && lower < 0) {
      ss << "-Inf";
    } else if (std::isinf(lower)) {
      ss << "Inf";
    } else {
      ss << lower;
    }
    ss << ";";
    if (std::isinf(upper) && upper > 0) {
      ss << "Inf";
    } else if (std::isinf(upper)) {
      ss << "-Inf";
    } else {
      ss << upper;
    }
    ss << "]";
    return ss.str();
  };
  
  // Prepare output
  size_t n_bins = unique_bins.size();
  bool has_missing = missing_bin.count > 0;
  size_t total_bins = has_missing ? n_bins + 1 : n_bins;
  
  CharacterVector bin_intervals(total_bins);
  NumericVector woe_values(total_bins), iv_values(total_bins), event_rates(total_bins);
  IntegerVector counts(total_bins), counts_pos(total_bins), counts_neg(total_bins);
  NumericVector cutpoints(n_bins > 1 ? n_bins - 1 : 0);
  double total_iv = 0.0;
  
  // Fill regular bins
  for (size_t i = 0; i < n_bins; ++i) {
    bin_intervals[i] = create_interval(unique_bins[i].lower, unique_bins[i].upper);
    woe_values[i] = unique_bins[i].woe;
    iv_values[i] = unique_bins[i].iv;
    event_rates[i] = unique_bins[i].event_rate;
    counts[i] = unique_bins[i].count;
    counts_pos[i] = unique_bins[i].count_pos;
    counts_neg[i] = unique_bins[i].count_neg;
    total_iv += unique_bins[i].iv;
    
    if (i > 0 && i < n_bins) {
      cutpoints[i-1] = unique_bins[i-1].upper;
    }
  }
  
  // Add missing bin if present
  if (has_missing) {
    bin_intervals[n_bins] = "NA";
    woe_values[n_bins] = missing_bin.woe;
    iv_values[n_bins] = missing_bin.iv;
    event_rates[n_bins] = missing_bin.event_rate;
    counts[n_bins] = missing_bin.count;
    counts_pos[n_bins] = missing_bin.count_pos;
    counts_neg[n_bins] = missing_bin.count_neg;
    total_iv += missing_bin.iv;
  }
  
  // Create vector of IDs with the same size as bins
  Rcpp::NumericVector ids(bin_intervals.size());
  for(int i = 0; i < bin_intervals.size(); i++) {
    ids[i] = i + 1;
  }
  
  return Rcpp::List::create(
    Rcpp::Named("id") = ids,
    Rcpp::Named("bin") = bin_intervals,
    Rcpp::Named("woe") = woe_values,
    Rcpp::Named("iv") = iv_values,
    Rcpp::Named("event_rate") = event_rates,
    Rcpp::Named("count") = counts,
    Rcpp::Named("count_pos") = counts_pos,
    Rcpp::Named("count_neg") = counts_neg,
    Rcpp::Named("cutpoints") = cutpoints,
    Rcpp::Named("total_iv") = total_iv,
    Rcpp::Named("gini") = gini,
    Rcpp::Named("ks") = ks,
    Rcpp::Named("converged") = true,
    Rcpp::Named("iterations") = 0
  );
}

//' @title Optimal Binning for Numerical Variables using Unsupervised Decision Trees
//'
//' @description 
//' This function implements an optimal binning algorithm for numerical variables
//' using an Unsupervised Decision Tree (UDT) approach with Weight of Evidence (WoE)
//' and Information Value (IV) criteria. The algorithm creates bins that maximize 
//' the predictive power of the feature while maintaining interpretability.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin as a proportion (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial discretization (default: 20).
//' @param laplace_smoothing Smoothing parameter for WoE calculation to handle zero counts (default: 0.5).
//' @param monotonicity_direction Specify monotonicity constraint: "none", "increasing", "decreasing", or "auto" (default: "none").
//' @param convergence_threshold Threshold for convergence of the optimization process (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
//'
//' @return A list containing binning details:
//' \item{id}{A numeric vector of bin identifiers.}
//' \item{bin}{A character vector of bin intervals.}
//' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
//' \item{iv}{A numeric vector of Information Value for each bin.}
//' \item{event_rate}{A numeric vector of event rates (proportion of positives) for each bin.}
//' \item{count}{An integer vector of total observations in each bin.}
//' \item{count_pos}{An integer vector of positive observations in each bin.}
//' \item{count_neg}{An integer vector of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cut points between bins.}
//' \item{total_iv}{The total Information Value of the binning.}
//' \item{gini}{The Gini coefficient measuring discrimination power.}
//' \item{ks}{The Kolmogorov-Smirnov statistic measuring separation.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value of the number of iterations run.}
//'
//' @details
//' The Unsupervised Decision Tree (UDT) binning algorithm discretizes a continuous 
//' variable into bins that maximize the Information Value (IV) while respecting 
//' constraints on the number and size of bins.
//'
//' The algorithm follows these main steps:
//' 1. Initial discretization using an entropy-based decision tree approach
//' 2. Merging of rare bins based on the bin_cutoff parameter
//' 3. Bin optimization using IV and WoE criteria
//' 4. Optional enforcement of monotonicity in WoE across bins
//' 5. Adjustment of the number of bins to be within the specified range
//'
//' The mathematical formulation of the optimization problem is:
//'
//' \deqn{
//' \max_{\{c_1, c_2, ..., c_{m-1}\}} \sum_{i=1}^{m} (p_i - q_i) \cdot \ln\left(\frac{p_i + \epsilon}{q_i + \epsilon}\right)
//' }
//'
//' Subject to:
//' \itemize{
//'   \item \eqn{min\_bins \leq m \leq max\_bins}
//'   \item \eqn{\frac{n_i}{n} \geq bin\_cutoff} for all i
//'   \item Optionally, \eqn{WoE_1 \leq WoE_2 \leq ... \leq WoE_m} (for increasing monotonicity)
//' }
//'
//' Where:
//' \itemize{
//'   \item \eqn{p_i = \frac{n_{i,1}}{n_1}} is the proportion of positive observations in bin i
//'   \item \eqn{q_i = \frac{n_{i,0}}{n_0}} is the proportion of negative observations in bin i
//'   \item \eqn{\epsilon} is the Laplace smoothing parameter
//' }
//'
//' The algorithm includes special handling for missing values (NA/NaN) and extreme values
//' (±Inf), as well as proper treatment of variables with very few unique values.
//'
//' @references
//' Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice
//' and the classical bias-variance trade-off. Proceedings of the National Academy of Sciences, 116(32), 15849-15854.
//'
//' Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning. Springer.
//'
//' Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and Its Applications. SIAM.
//'
//' Siddiqi, N. (2017). Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards. Wiley.
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_udt(
//'   target, feature, 
//'   min_bins = 3, 
//'   max_bins = 5,
//'   monotonicity_direction = "auto",
//'   laplace_smoothing = 0.5
//' )
//'
//' # View binning results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_udt(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double laplace_smoothing = 0.5,
   Rcpp::String monotonicity_direction = "none",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Input validation
 if (feature.size() == 0) {
   Rcpp::stop("Feature vector is empty.");
 }
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must be of the same length.");
 }
 for (int i = 0; i < target.size(); ++i) {
   if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target vector must contain only binary values 0 and 1.");
   }
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
   Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
 }
 if (laplace_smoothing < 0) {
   Rcpp::stop("laplace_smoothing must be non-negative.");
 }
 
 // Convert monotonicity direction string to enum
 std::string mono_dir_str = monotonicity_direction;
 std::transform(mono_dir_str.begin(), mono_dir_str.end(), mono_dir_str.begin(), ::tolower);
 MonotonicityDirection mono_dir = string_to_monotonicity_direction(mono_dir_str);
 
 // Convert inputs to std::vector
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Count unique non-NA/non-Inf values
 std::unordered_set<double> unique_values;
 for (double val : feature_vec) {
   if (!std::isnan(val) && !std::isinf(val)) {
     unique_values.insert(val);
   }
 }
 
 // Handle features with <=2 unique values
 if (unique_values.size() <= 2) {
   return handle_few_unique_values(feature_vec, target_vec, laplace_smoothing);
 }
 
 // Perform optimal binning
 OptimalBinningNumericalUDT obj(
     feature_vec, target_vec, min_bins, max_bins, 
     bin_cutoff, max_n_prebins, laplace_smoothing, mono_dir,
     convergence_threshold, max_iterations);
 
 obj.fit();
 return obj.get_result();
}


// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <string>
// #include <unordered_set>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// // Helper structure for bins
// struct Bin {
//   double lower;
//   double upper;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
// };
// 
// // Comparator for sorting
// bool compareBins(const Bin &a, const Bin &b) {
//   return a.lower < b.lower;
// }
// 
// // Forward declaration of handle_few_unique_values
// Rcpp::List handle_few_unique_values(
//     const std::vector<double>& feature_vec,
//     const std::vector<int>& target_vec
// );
// 
// // Class for Optimal Binning
// class OptimalBinningNumericalUDT {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::vector<Bin> bins;
//   double convergence_threshold;
//   int max_iterations;
//   bool converged;
//   int iterations_run;
//   
// public:
//   OptimalBinningNumericalUDT(std::vector<double> feature,
//                              std::vector<int> target,
//                              int min_bins = 3,
//                              int max_bins = 5,
//                              double bin_cutoff = 0.05,
//                              int max_n_prebins = 20,
//                              double convergence_threshold = 1e-6,
//                              int max_iterations = 1000) 
//     : feature(std::move(feature)), 
//       target(std::move(target)), 
//       min_bins(std::max(2, min_bins)),
//       max_bins(std::max(this->min_bins, max_bins)),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(false),
//       iterations_run(0) {}
//   
//   void fit() {
//     // Pre-binning using quantiles
//     std::vector<double> cut_points = get_quantile_cutpoints(feature, max_n_prebins);
//     
//     // Initial binning
//     bins = initial_binning(feature, target, cut_points);
//     
//     // Rare bin merging
//     merge_rare_bins();
//     
//     // Bin optimization
//     optimize_bins();
//   }
//   
//   Rcpp::List get_result() const {
//     size_t n_bins = bins.size();
//     CharacterVector bin_intervals(n_bins);
//     NumericVector woe_values(n_bins), iv_values(n_bins);
//     IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
//     NumericVector cutpoints(n_bins - 1);
//     
//     for (size_t i = 0; i < n_bins; ++i) {
//       bin_intervals[i] = create_interval(bins[i].lower, bins[i].upper);
//       woe_values[i] = bins[i].woe;
//       iv_values[i] = bins[i].iv;
//       counts[i] = bins[i].count;
//       counts_pos[i] = bins[i].count_pos;
//       counts_neg[i] = bins[i].count_neg;
//       if (i < n_bins - 1) {
//         cutpoints[i] = bins[i].upper;
//       }
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_intervals,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("cutpoints") = cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
//   
//   void calculate_woe_iv(std::vector<Bin> &bins) const {
//     int total_pos = 0, total_neg = 0;
//     for (const auto &bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     double total_pos_d = total_pos > 0 ? static_cast<double>(total_pos) : 1.0;
//     double total_neg_d = total_neg > 0 ? static_cast<double>(total_neg) : 1.0;
//     
//     for (auto &bin : bins) {
//       double dist_pos = static_cast<double>(bin.count_pos) / total_pos_d;
//       double dist_neg = static_cast<double>(bin.count_neg) / total_neg_d;
//       if (dist_pos == 0.0) dist_pos = 0.0001;
//       if (dist_neg == 0.0) dist_neg = 0.0001;
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   std::string create_interval(double lower, double upper) const {
//     std::stringstream ss;
//     ss << "(";
//     if (std::isinf(lower))
//       ss << "-Inf";
//     else
//       ss << lower;
//     ss << ";";
//     if (std::isinf(upper))
//       ss << "+Inf";
//     else
//       ss << upper;
//     ss << "]";
//     return ss.str();
//   }
//   
// private:
//   std::vector<double> get_quantile_cutpoints(const std::vector<double> &data, int n_bins) const {
//     std::vector<double> quantiles;
//     if (n_bins <= 1) {
//       return quantiles;
//     }
//     
//     std::vector<double> sorted_data = data;
//     sorted_data.erase(std::remove_if(sorted_data.begin(), sorted_data.end(),
//                                      [](double x) { return std::isnan(x); }), sorted_data.end());
//     
//     if (sorted_data.empty()) {
//       return quantiles;
//     }
//     
//     std::sort(sorted_data.begin(), sorted_data.end());
//     
//     for (int i = 1; i < n_bins; ++i) {
//       double p = static_cast<double>(i) / n_bins;
//       double pos = p * (sorted_data.size() - 1);
//       size_t idx = static_cast<size_t>(std::floor(pos));
//       double frac = pos - static_cast<double>(idx);
//       double quantile;
//       if (idx + 1 < sorted_data.size()) {
//         quantile = sorted_data[idx] + frac * (sorted_data[idx + 1] - sorted_data[idx]);
//       } else {
//         quantile = sorted_data[idx];
//       }
//       quantiles.push_back(quantile);
//     }
//     
//     std::vector<double> unique_quantiles;
//     std::unordered_set<double> seen;
//     for (double q : quantiles) {
//       if (seen.find(q) == seen.end()) {
//         unique_quantiles.push_back(q);
//         seen.insert(q);
//       }
//     }
//     
//     return unique_quantiles;
//   }
//   
//   std::vector<Bin> initial_binning(const std::vector<double> &feature,
//                                    const std::vector<int> &target,
//                                    const std::vector<double> &cut_points) const {
//     std::vector<Bin> initial_bins;
//     std::vector<double> boundaries = cut_points;
//     boundaries.insert(boundaries.begin(), -std::numeric_limits<double>::infinity());
//     boundaries.push_back(std::numeric_limits<double>::infinity());
//     size_t n_bins = boundaries.size() - 1;
//     
//     initial_bins.resize(n_bins);
//     for (size_t i = 0; i < n_bins; ++i) {
//       initial_bins[i].lower = boundaries[i];
//       initial_bins[i].upper = boundaries[i + 1];
//       initial_bins[i].count = 0;
//       initial_bins[i].count_pos = 0;
//       initial_bins[i].count_neg = 0;
//       initial_bins[i].woe = 0.0;
//       initial_bins[i].iv = 0.0;
//     }
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       int bin_idx = find_bin_index(val, boundaries);
//       if (bin_idx >= 0 && static_cast<size_t>(bin_idx) < n_bins) {
//         initial_bins[bin_idx].count++;
//         if (target[i] == 1) {
//           initial_bins[bin_idx].count_pos++;
//         } else {
//           initial_bins[bin_idx].count_neg++;
//         }
//       }
//     }
//     
//     calculate_woe_iv(initial_bins);
//     
//     return initial_bins;
//   }
//   
//   int find_bin_index(double value, const std::vector<double> &boundaries) const {
//     int left = 0;
//     int right = boundaries.size() - 1;
//     while (left < right - 1) {
//       int mid = left + (right - left) / 2;
//       if (value <= boundaries[mid]) {
//         right = mid;
//       } else {
//         left = mid;
//       }
//     }
//     return left;
//   }
//   
//   void merge_rare_bins() {
//     size_t total_count = feature.size();
//     double cutoff_count = bin_cutoff * static_cast<double>(total_count);
//     bool merged = false;
//     
//     do {
//       merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (static_cast<double>(bins[i].count) < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
//           if (i > 0) {
//             bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
//             bins.erase(bins.begin() + i);
//           } else if (i < bins.size() - 1) {
//             bins[i + 1] = merge_bins(bins[i], bins[i + 1]);
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break;
//         }
//       }
//     } while (merged && bins.size() > static_cast<size_t>(min_bins));
//     
//     calculate_woe_iv(bins);
//   }
//   
//   Bin merge_bins(const Bin &bin1, const Bin &bin2) const {
//     Bin merged;
//     merged.lower = bin1.lower;
//     merged.upper = bin2.upper;
//     merged.count = bin1.count + bin2.count;
//     merged.count_pos = bin1.count_pos + bin2.count_pos;
//     merged.count_neg = bin1.count_neg + bin2.count_neg;
//     merged.woe = 0.0;
//     merged.iv = 0.0;
//     return merged;
//   }
//   
//   void optimize_bins() {
//     double prev_total_iv = 0.0;
//     iterations_run = 0;
//     
//     while (iterations_run < max_iterations) {
//       // Ensure number of bins within min_bins and max_bins
//       while (static_cast<int>(bins.size()) > max_bins) {
//         merge_adjacent_bins();
//       }
//       
//       // Enforce monotonicity
//       enforce_monotonicity();
//       
//       // Ensure minimum number of bins
//       while (static_cast<int>(bins.size()) < min_bins && bins.size() > 1) {
//         split_largest_bin();
//       }
//       
//       double total_iv = calculate_total_iv();
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations_run++;
//     }
//   }
//   
//   void merge_adjacent_bins() {
//     double min_iv_gain = std::numeric_limits<double>::max();
//     size_t merge_idx = 0;
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       double iv_gain = bins[i].iv + bins[i + 1].iv;
//       if (iv_gain < min_iv_gain) {
//         min_iv_gain = iv_gain;
//         merge_idx = i;
//       }
//     }
//     bins[merge_idx] = merge_bins(bins[merge_idx], bins[merge_idx + 1]);
//     bins.erase(bins.begin() + merge_idx + 1);
//     calculate_woe_iv(bins);
//   }
//   
//   void enforce_monotonicity() {
//     bool is_monotonic = false;
//     while (!is_monotonic && bins.size() > static_cast<size_t>(min_bins)) {
//       is_monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if (bins[i].woe < bins[i - 1].woe) {
//           bins[i - 1] = merge_bins(bins[i - 1], bins[i]);
//           bins.erase(bins.begin() + i);
//           is_monotonic = false;
//           break;
//         }
//       }
//       calculate_woe_iv(bins);
//     }
//   }
//   
//   void split_largest_bin() {
//     size_t largest_bin_idx = 0;
//     int max_count = 0;
//     for (size_t i = 0; i < bins.size(); ++i) {
//       if (bins[i].count > max_count) {
//         max_count = bins[i].count;
//         largest_bin_idx = i;
//       }
//     }
//     
//     Bin &largest_bin = bins[largest_bin_idx];
//     double mid_point = (largest_bin.lower + largest_bin.upper) / 2.0;
//     
//     Bin new_bin = largest_bin;
//     new_bin.upper = mid_point;
//     largest_bin.lower = mid_point;
//     
//     bins.insert(bins.begin() + largest_bin_idx, new_bin);
//     
//     redistribute_points(largest_bin_idx);
//     calculate_woe_iv(bins);
//   }
//   
//   void redistribute_points(size_t split_bin_idx) {
//     Bin &left_bin = bins[split_bin_idx];
//     Bin &right_bin = bins[split_bin_idx + 1];
//     
//     left_bin.count = 0;
//     left_bin.count_pos = 0;
//     left_bin.count_neg = 0;
//     right_bin.count = 0;
//     right_bin.count_pos = 0;
//     right_bin.count_neg = 0;
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       if (feature[i] >= left_bin.lower && feature[i] < left_bin.upper) {
//         left_bin.count++;
//         if (target[i] == 1) left_bin.count_pos++;
//         else left_bin.count_neg++;
//       } else if (feature[i] >= right_bin.lower && feature[i] <= right_bin.upper) {
//         right_bin.count++;
//         if (target[i] == 1) right_bin.count_pos++;
//         else right_bin.count_neg++;
//       }
//     }
//   }
//   
//   double calculate_total_iv() const {
//     double total_iv = 0.0;
//     for (const auto &bin : bins) {
//       total_iv += bin.iv;
//     }
//     return total_iv;
//   }
// };
// 
// // Implementation of handle_few_unique_values
// Rcpp::List handle_few_unique_values(
//     const std::vector<double>& feature_vec,
//     const std::vector<int>& target_vec
// ) {
//   std::vector<Bin> unique_bins;
//   std::unordered_set<double> unique_values_set(feature_vec.begin(), feature_vec.end());
//   size_t unique_size = unique_values_set.size();
//   
//   if (unique_size == 1) {
//     // Only one unique value, single bin
//     Bin b;
//     b.lower = -std::numeric_limits<double>::infinity();
//     b.upper = std::numeric_limits<double>::infinity();
//     b.count = feature_vec.size();
//     b.count_pos = 0;
//     b.count_neg = 0;
//     for (size_t i = 0; i < feature_vec.size(); ++i) {
//       if (target_vec[i] == 1) b.count_pos++;
//       else b.count_neg++;
//     }
//     unique_bins.push_back(b);
//   } else if (unique_size == 2) {
//     // Two unique values, create two bins with min value as cutpoint
//     double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
//     Bin bin1, bin2;
//     
//     bin1.lower = -std::numeric_limits<double>::infinity();
//     bin1.upper = min_val;
//     bin1.count = 0;
//     bin1.count_pos = 0;
//     bin1.count_neg = 0;
//     
//     bin2.lower = min_val;
//     bin2.upper = std::numeric_limits<double>::infinity();
//     bin2.count = 0;
//     bin2.count_pos = 0;
//     bin2.count_neg = 0;
//     
//     for (size_t i = 0; i < feature_vec.size(); ++i) {
//       double val = feature_vec[i];
//       if (val <= min_val) {
//         bin1.count++;
//         if (target_vec[i] == 1) bin1.count_pos++;
//         else bin1.count_neg++;
//       } else {
//         bin2.count++;
//         if (target_vec[i] == 1) bin2.count_pos++;
//         else bin2.count_neg++;
//       }
//     }
//     
//     unique_bins.push_back(bin1);
//     unique_bins.push_back(bin2);
//   }
//   
//   // Calculate WOE and IV
//   OptimalBinningNumericalUDT temp_obj(feature_vec, target_vec);
//   temp_obj.calculate_woe_iv(unique_bins);
//   
//   // Prepare output
//   size_t n_bins = unique_bins.size();
//   CharacterVector bin_intervals(n_bins);
//   NumericVector woe_values(n_bins), iv_values(n_bins);
//   IntegerVector counts(n_bins), counts_pos(n_bins), counts_neg(n_bins);
//   NumericVector cutpoints(n_bins - 1);
//   
//   for (size_t i = 0; i < n_bins; ++i) {
//     bin_intervals[i] = temp_obj.create_interval(unique_bins[i].lower, unique_bins[i].upper);
//     woe_values[i] = unique_bins[i].woe;
//     iv_values[i] = unique_bins[i].iv;
//     counts[i] = unique_bins[i].count;
//     counts_pos[i] = unique_bins[i].count_pos;
//     counts_neg[i] = unique_bins[i].count_neg;
//     if (i < n_bins - 1) {
//       cutpoints[i] = unique_bins[i].upper;
//     }
//   }
//   
//   // Criar vetor de IDs com o mesmo tamanho de bins
//   Rcpp::NumericVector ids(bin_intervals.size());
//   for(int i = 0; i < bin_intervals.size(); i++) {
//     ids[i] = i + 1;
//   }
//   
//   return Rcpp::List::create(
//     Rcpp::Named("id") = ids,
//     Rcpp::Named("bin") = bin_intervals,
//     Rcpp::Named("woe") = woe_values,
//     Rcpp::Named("iv") = iv_values,
//     Rcpp::Named("count") = counts,
//     Rcpp::Named("count_pos") = counts_pos,
//     Rcpp::Named("count_neg") = counts_neg,
//     Rcpp::Named("cutpoints") = cutpoints,
//     Rcpp::Named("converged") = true,
//     Rcpp::Named("iterations") = 0
//   );
// }
// 
// //' @title Optimal Binning for Numerical Variables using Unsupervised Decision Trees
// //'
// //' @description 
// //' This function implements an optimal binning algorithm for numerical variables
// //' using an Unsupervised Decision Tree (UDT) approach with Weight of Evidence (WoE)
// //' and Information Value (IV) criteria.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
// //' @param convergence_threshold Threshold for convergence of the optimization process (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
// //'
// //' @return A list containing binning details:
// //' \item{bins}{A character vector of bin intervals.}
// //' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
// //' \item{iv}{A numeric vector of Information Value for each bin.}
// //' \item{count}{An integer vector of total observations in each bin.}
// //' \item{count_pos}{An integer vector of positive observations in each bin.}
// //' \item{count_neg}{An integer vector of negative observations in each bin.}
// //' \item{cutpoints}{A numeric vector of cut points between bins.}
// //' \item{converged}{A logical value indicating whether the algorithm converged.}
// //' \item{iterations}{An integer value of the number of iterations run.}
// //'
// //' @details
// //' The optimal binning algorithm for numerical variables uses an Unsupervised Decision Tree
// //' approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that
// //' maximize the predictive power of the feature while maintaining interpretability.
// //'
// //' The algorithm follows these steps:
// //' 1. Initial discretization using quantile-based binning
// //' 2. Merging of rare bins based on the bin_cutoff parameter
// //' 3. Bin optimization using IV and WoE criteria
// //' 4. Enforcement of monotonicity in WoE across bins
// //' 5. Adjustment of the number of bins to be within the specified range
// //'
// //' @examples
// //' \dontrun{
// //' # Generate sample data
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //'
// //' # Apply optimal binning
// //' result <- optimal_binning_numerical_udt(target, feature, min_bins = 3, max_bins = 5)
// //'
// //' # View binning results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_udt(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Input validation
//  if (feature.size() == 0) {
//    Rcpp::stop("Feature vector is empty.");
//  }
//  if (target.size() != feature.size()) {
//    Rcpp::stop("Target and feature vectors must be of the same length.");
//  }
//  for (int i = 0; i < target.size(); ++i) {
//    if (target[i] != 0 && target[i] != 1) {
//      Rcpp::stop("Target vector must contain only binary values 0 and 1.");
//    }
//  }
//  if (min_bins < 2) {
//    Rcpp::stop("min_bins must be at least 2.");
//  }
//  if (max_bins < min_bins) {
//    Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//  }
//  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//    Rcpp::stop("bin_cutoff must be between 0 and 1.");
//  }
//  if (max_n_prebins < min_bins) {
//    Rcpp::stop("max_n_prebins must be greater than or equal to min_bins.");
//  }
//  
//  // Convert inputs to std::vector
//  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  // Handle features with <=2 unique values
//  std::unordered_set<double> unique_values(feature_vec.begin(), feature_vec.end());
//  if (unique_values.size() <= 2) {
//    return handle_few_unique_values(feature_vec, target_vec);
//  }
//  
//  // Perform optimal binning
//  OptimalBinningNumericalUDT obj(feature_vec, target_vec, min_bins, max_bins, 
//                                 bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
//  obj.fit();
//  return obj.get_result();
// }
