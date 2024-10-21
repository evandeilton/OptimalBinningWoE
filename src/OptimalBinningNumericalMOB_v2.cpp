#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <set>

using namespace Rcpp;

// Define a small epsilon to prevent log(0) and division by zero
const double EPSILON = 1e-10;

// Structure to hold bin metrics
struct BinMetrics {
  double lower;
  double upper;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
};

// Class for Optimal Binning Numerical MOB
class OptimalBinningNumericalMOB {
public:
  OptimalBinningNumericalMOB(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05,
                             int max_n_prebins_ = 20, double convergence_threshold_ = 1e-6,
                             int max_iterations_ = 1000)
    : min_bins(min_bins_), max_bins(std::max(min_bins_, max_bins_)),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_) {}
  
  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    validate_input();
    
    size_t n_unique = count_unique(feature);
    
    if (n_unique <= 2) {
      // Handle case with <=2 unique values
      handle_low_unique_values();
    } else {
      // Proceed with normal binning process
      initial_binning();
      optimize_bins();
      calculate_woe_iv();
    }
  }
  
  std::vector<BinMetrics> get_bin_metrics() const {
    std::vector<BinMetrics> bins;
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      BinMetrics bm;
      bm.lower = bin_edges[i];
      bm.upper = bin_edges[i + 1];
      bm.woe = woe_values[i];
      bm.iv = iv_values[i];
      bm.count = count[i];
      bm.count_pos = count_pos[i];
      bm.count_neg = count_neg[i];
      bins.push_back(bm);
    }
    return bins;
  }
  
  std::vector<double> get_cutpoints() const {
    return std::vector<double>(bin_edges.begin() + 1, bin_edges.end() - 1);
  }
  
  bool has_converged() const { return converged; }
  int get_iterations() const { return iterations; }
  
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  std::vector<double> feature;
  std::vector<int> target;
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count;
  std::vector<int> count_pos;
  std::vector<int> count_neg;
  double total_pos;
  double total_neg;
  bool converged;
  int iterations;
  
  // Helper function to count unique values in a vector
  size_t count_unique(const std::vector<double>& vec) const {
    if (vec.empty()) return 0;
    size_t unique = 1;
    std::vector<double> sorted_vec = vec;
    std::sort(sorted_vec.begin(), sorted_vec.end());
    for (size_t i = 1; i < sorted_vec.size(); ++i) {
      if (sorted_vec[i] != sorted_vec[i - 1]) unique++;
    }
    return unique;
  }
  
  void validate_input() {
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    for (int t : target) {
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target vector must be binary (0 and 1).");
      }
    }
    total_pos = std::accumulate(target.begin(), target.end(), 0.0);
    total_neg = feature.size() - total_pos;
    if (total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target vector must contain both classes (0 and 1).");
    }
    if (min_bins < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 0.5) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 0.5.");
    }
    if (max_n_prebins < min_bins) {
      max_n_prebins = min_bins;
      Rcpp::warning("max_n_prebins adjusted to be at least min_bins.");
    }
  }
  
  void handle_low_unique_values() {
    // Use min value as the single cutpoint to create two bins
    double min_val = *std::min_element(feature.begin(), feature.end());
    bin_edges = { -std::numeric_limits<double>::infinity(), min_val, std::numeric_limits<double>::infinity() };
    
    // Initialize counts
    count.assign(2, 0);
    count_pos.assign(2, 0);
    count_neg.assign(2, 0);
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (feature[i] <= min_val) {
        count[0]++;
        if (target[i] == 1) {
          count_pos[0]++;
        } else {
          count_neg[0]++;
        }
      } else {
        count[1]++;
        if (target[i] == 1) {
          count_pos[1]++;
        } else {
          count_neg[1]++;
        }
      }
    }
    
    calculate_woe_iv();
    converged = true;
    iterations = 0;
  }
  
  void initial_binning() {
    // Sort feature values along with target
    std::vector<std::pair<double, int>> feature_target(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_target[i] = std::make_pair(feature[i], target[i]);
    }
    std::sort(feature_target.begin(), feature_target.end());
    
    bin_edges.clear();
    bin_edges.push_back(-std::numeric_limits<double>::infinity());
    
    size_t n = feature_target.size();
    size_t n_distinct = 1;
    for (size_t i = 1; i < n; ++i) {
      if (feature_target[i].first != feature_target[i - 1].first) {
        n_distinct++;
      }
    }
    
    if (n_distinct <= static_cast<size_t>(min_bins)) {
      // If number of distinct values is less than or equal to min_bins, use all unique values
      for (size_t i = 1; i < n; ++i) {
        if (feature_target[i].first != feature_target[i - 1].first) {
          bin_edges.push_back(feature_target[i].first);
        }
      }
    } else {
      size_t n_bins = std::min(static_cast<size_t>(max_n_prebins), n_distinct);
      size_t bin_size = n / n_bins;
      
      // Collect bin edges based on quantiles
      for (size_t i = 1; i < n_bins; ++i) {
        size_t idx = i * bin_size;
        if (idx >= n) idx = n - 1;
        double edge = feature_target[idx].first;
        // Avoid duplicate edges
        if (edge > bin_edges.back()) {
          bin_edges.push_back(edge);
        }
      }
    }
    bin_edges.push_back(std::numeric_limits<double>::infinity());
    
    // Ensure we have at least min_bins+1 bin edges
    while (bin_edges.size() - 1 < static_cast<size_t>(min_bins)) {
      // Split bins with largest counts
      count.assign(bin_edges.size() - 1, 0);
      count_pos.assign(bin_edges.size() - 1, 0);
      count_neg.assign(bin_edges.size() - 1, 0);
      
      // Count observations in each bin
      for (const auto& ft : feature_target) {
        int bin_idx = find_bin(ft.first);
        if (bin_idx >= 0 && bin_idx < static_cast<int>(count.size())) {
          count[bin_idx]++;
          if (ft.second == 1) {
            count_pos[bin_idx]++;
          } else {
            count_neg[bin_idx]++;
          }
        }
      }
      
      // Find bin with largest count
      size_t max_count_idx = std::distance(count.begin(), std::max_element(count.begin(), count.end()));
      double lower = bin_edges[max_count_idx];
      double upper = bin_edges[max_count_idx + 1];
      
      // Find median value in this bin
      std::vector<double> bin_values;
      for (const auto& ft : feature_target) {
        if (ft.first > lower && ft.first <= upper) {
          bin_values.push_back(ft.first);
        }
      }
      if (bin_values.empty()) break; // Cannot split further
      
      // Split at median
      size_t median_idx = bin_values.size() / 2;
      std::nth_element(bin_values.begin(), bin_values.begin() + median_idx, bin_values.end());
      double median = bin_values[median_idx];
      if (median <= lower || median >= upper) break; // Cannot split further
      
      // Insert new edge
      bin_edges.insert(bin_edges.begin() + max_count_idx + 1, median);
    }
  }
  
  int find_bin(double value) const {
    auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
    int idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
    if (idx < 0) idx = 0;
    if (idx >= static_cast<int>(bin_edges.size() - 1)) idx = bin_edges.size() - 2;
    return idx;
  }
  
  bool is_monotonic(const std::vector<double>& woe) const {
    if (woe.size() < 2) return true;
    bool increasing = true, decreasing = true;
    for (size_t i = 1; i < woe.size(); ++i) {
      if (woe[i] < woe[i - 1]) increasing = false;
      if (woe[i] > woe[i - 1]) decreasing = false;
      if (!increasing && !decreasing) return false;
    }
    return true;
  }
  
  void merge_bins() {
    iterations = 0;
    converged = false;
    double previous_iv = calculate_total_iv();
    
    while (iterations++ < max_iterations && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
      bool merged = false;
      
      // Recalculate counts and WoE
      calculate_initial_woe();
      
      // Check for bins below bin_cutoff
      for (size_t i = 0; i < count.size(); ++i) {
        double freq = static_cast<double>(count[i]) / feature.size();
        if (freq < bin_cutoff && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
          if (i == 0 && count.size() > 1) {
            merge_with_next(0);
          } else if (i > 0) {
            merge_with_prev(i);
          }
          merged = true;
          break;
        }
      }
      if (merged) continue;
      
      // Check for monotonicity
      if (!is_monotonic(woe_values) && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
        size_t break_point = 1;
        bool increasing = woe_values[1] > woe_values[0];
        for (; break_point < woe_values.size(); ++break_point) {
          if ((increasing && woe_values[break_point] < woe_values[break_point - 1]) ||
              (!increasing && woe_values[break_point] > woe_values[break_point - 1])) {
            break;
          }
        }
        if (break_point < woe_values.size()) {
          merge_with_prev(break_point);
          merged = true;
        }
      }
      
      if (!merged) {
        // Check for convergence
        double current_iv = calculate_total_iv();
        if (std::abs(current_iv - previous_iv) < convergence_threshold) {
          converged = true;
          break;
        }
        previous_iv = current_iv;
      }
    }
    
    if (iterations >= max_iterations) {
      Rcpp::warning("Maximum iterations reached in merge_bins. Results may be suboptimal.");
    }
  }
  
  void merge_with_prev(size_t i) {
    if (i <= 0 || i >= bin_edges.size() - 1) return;
    bin_edges.erase(bin_edges.begin() + i);
    count[i - 1] += count[i];
    count_pos[i - 1] += count_pos[i];
    count_neg[i - 1] += count_neg[i];
    count.erase(count.begin() + i);
    count_pos.erase(count_pos.begin() + i);
    count_neg.erase(count_neg.begin() + i);
    woe_values.erase(woe_values.begin() + i);
    iv_values.erase(iv_values.begin() + i);
  }
  
  void merge_with_next(size_t i) {
    if (i >= bin_edges.size() - 2) return;
    bin_edges.erase(bin_edges.begin() + i + 1);
    count[i] += count[i + 1];
    count_pos[i] += count_pos[i + 1];
    count_neg[i] += count_neg[i + 1];
    count.erase(count.begin() + i + 1);
    count_pos.erase(count_pos.begin() + i + 1);
    count_neg.erase(count_neg.begin() + i + 1);
    woe_values.erase(woe_values.begin() + i + 1);
    iv_values.erase(iv_values.begin() + i + 1);
  }
  
  void optimize_bins() {
    // Initialize counts
    count.assign(bin_edges.size() - 1, 0);
    count_pos.assign(bin_edges.size() - 1, 0);
    count_neg.assign(bin_edges.size() - 1, 0);
    
    // Sort feature and target together
    std::vector<std::pair<double, int>> feature_target(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      feature_target[i] = std::make_pair(feature[i], target[i]);
    }
    std::sort(feature_target.begin(), feature_target.end());
    
    // Count observations in each bin
    for (const auto& ft : feature_target) {
      int bin_idx = find_bin(ft.first);
      if (bin_idx >= 0 && bin_idx < static_cast<int>(count.size())) {
        count[bin_idx]++;
        if (ft.second == 1) {
          count_pos[bin_idx]++;
        } else {
          count_neg[bin_idx]++;
        }
      }
    }
    
    calculate_initial_woe();
    merge_bins();
    
    // Ensure the number of bins does not exceed max_bins
    while (count.size() > static_cast<size_t>(max_bins)) {
      // Merge bins with smallest IV
      double min_iv = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      for (size_t i = 0; i < iv_values.size() - 1; ++i) {
        double combined_iv = iv_values[i] + iv_values[i + 1];
        if (combined_iv < min_iv) {
          min_iv = combined_iv;
          merge_idx = i;
        }
      }
      merge_with_next(merge_idx);
      calculate_initial_woe();
    }
  }
  
  void calculate_initial_woe() {
    total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
    total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
    woe_values.resize(count.size());
    iv_values.resize(count.size());
    for (size_t i = 0; i < count.size(); ++i) {
      double pct_pos = static_cast<double>(count_pos[i]) / total_pos;
      double pct_neg = static_cast<double>(count_neg[i]) / total_neg;
      pct_pos = std::max(pct_pos, EPSILON);
      pct_neg = std::max(pct_neg, EPSILON);
      woe_values[i] = std::log(pct_pos / pct_neg);
      iv_values[i] = (pct_pos - pct_neg) * woe_values[i];
    }
  }
  
  void calculate_woe_iv() {
    calculate_initial_woe();
  }
  
  double calculate_total_iv() const {
    return std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
  }
};


//' Perform Optimal Binning for Numerical Features using Monotonic Optimal Binning (MOB)
//'
//' This function implements the Monotonic Optimal Binning algorithm for numerical features.
//' It creates optimal bins while maintaining monotonicity in the Weight of Evidence (WoE) values.
//'
//' @param target An integer vector of binary target values (0 or 1)
//' @param feature A numeric vector of feature values to be binned
//' @param min_bins Minimum number of bins to create (default: 3)
//' @param max_bins Maximum number of bins to create (default: 5)
//' @param bin_cutoff Minimum frequency of observations in a bin (default: 0.05)
//' @param max_n_prebins Maximum number of prebins to create initially (default: 20)
//' @param convergence_threshold Threshold for convergence in the iterative process (default: 1e-6)
//' @param max_iterations Maximum number of iterations for the binning process (default: 1000)
//'
//' @return A list containing the following elements:
//'   \item{bin}{A character vector of bin labels}
//'   \item{woe}{A numeric vector of Weight of Evidence values for each bin}
//'   \item{iv}{A numeric vector of Information Value for each bin}
//'   \item{count}{An integer vector of total count of observations in each bin}
//'   \item{count_pos}{An integer vector of count of positive class observations in each bin}
//'   \item{count_neg}{An integer vector of count of negative class observations in each bin}
//'   \item{cutpoints}{A numeric vector of cutpoints used to create the bins}
//'   \item{converged}{A logical value indicating whether the algorithm converged}
//'   \item{iterations}{An integer value indicating the number of iterations run}
//'
//' @details
//' The algorithm starts by creating initial bins and then iteratively merges them
//' to achieve optimal binning while maintaining monotonicity in the WoE values.
//' It respects the minimum and maximum number of bins specified.
//'
//' @examples
//' \dontrun{
//' set.seed(42)
//' feature <- rnorm(1000)
//' target <- rbinom(1000, 1, 0.5)
//' result <- optimal_binning_numerical_mob(target, feature)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_mob(IntegerVector target, NumericVector feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20,
                                  double convergence_threshold = 1e-6, int max_iterations = 1000) {
 if (feature.size() != target.size()) {
   stop("Feature and target vectors must be of the same length.");
 }
 
 // Ensure max_bins is at least equal to min_bins
 max_bins = std::max(min_bins, max_bins);
 
 std::vector<double> feature_vec = as<std::vector<double>>(feature);
 std::vector<int> target_vec = as<std::vector<int>>(target);
 
 OptimalBinningNumericalMOB binning(min_bins, max_bins, bin_cutoff, max_n_prebins,
                                    convergence_threshold, max_iterations);
 
 try {
   binning.fit(feature_vec, target_vec);
 } catch (const std::exception& e) {
   stop(std::string("Error in binning process: ") + e.what());
 }
 
 std::vector<BinMetrics> bins = binning.get_bin_metrics();
 
 std::vector<std::string> bin_names;
 std::vector<double> bin_woe, bin_iv;
 std::vector<int> bin_count, bin_count_pos, bin_count_neg;
 std::vector<double> bin_cutpoints = binning.get_cutpoints();
 
 for (const auto& b : bins) {
   std::string lower_str = (std::isfinite(b.lower)) ? std::to_string(b.lower) : "-Inf";
   std::string upper_str = (std::isfinite(b.upper)) ? std::to_string(b.upper) : "+Inf";
   bin_names.push_back("(" + lower_str + ";" + upper_str + "]");
   bin_woe.push_back(b.woe);
   bin_iv.push_back(b.iv);
   bin_count.push_back(b.count);
   bin_count_pos.push_back(b.count_pos);
   bin_count_neg.push_back(b.count_neg);
 }
 
 return Rcpp::List::create(
   Named("bin") = bin_names,
   Named("woe") = bin_woe,
   Named("iv") = bin_iv,
   Named("count") = bin_count,
   Named("count_pos") = bin_count_pos,
   Named("count_neg") = bin_count_neg,
   Named("cutpoints") = bin_cutpoints,
   Named("converged") = binning.has_converged(),
   Named("iterations") = binning.get_iterations()
 );
}










// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <numeric>
// #include <string>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// // Define a small epsilon to prevent log(0) and division by zero
// const double EPSILON = 1e-10;
// 
// // Structure to hold bin metrics
// struct BinMetrics {
//   double lower;
//   double upper;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
// };
// 
// // Class for Optimal Binning Numerical MOB
// class OptimalBinningNumericalMOB {
// public:
//   OptimalBinningNumericalMOB(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05,
//                              int max_n_prebins_ = 20, double convergence_threshold_ = 1e-6,
//                              int max_iterations_ = 1000)
//     : min_bins(min_bins_), max_bins(std::max(min_bins_, max_bins_)),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//       convergence_threshold(convergence_threshold_), max_iterations(max_iterations_) {}
// 
//   void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
//     validate_input();
//     initial_binning();
//     optimize_bins();
//     calculate_woe_iv();
//   }
// 
//   std::vector<BinMetrics> get_bin_metrics() const {
//     std::vector<BinMetrics> bins;
//     for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
//       BinMetrics bm;
//       bm.lower = bin_edges[i];
//       bm.upper = bin_edges[i + 1];
//       bm.woe = woe_values[i];
//       bm.iv = iv_values[i];
//       bm.count = count[i];
//       bm.count_pos = count_pos[i];
//       bm.count_neg = count_neg[i];
//       bins.push_back(bm);
//     }
//     return bins;
//   }
// 
//   std::vector<double> get_cutpoints() const {
//     return std::vector<double>(bin_edges.begin() + 1, bin_edges.end() - 1);
//   }
// 
//   bool has_converged() const { return converged; }
//   int get_iterations() const { return iterations; }
// 
// private:
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   std::vector<double> feature;
//   std::vector<int> target;
//   std::vector<double> bin_edges;
//   std::vector<double> woe_values;
//   std::vector<double> iv_values;
//   std::vector<int> count;
//   std::vector<int> count_pos;
//   std::vector<int> count_neg;
//   double total_pos;
//   double total_neg;
//   bool converged;
//   int iterations;
// 
//   void validate_input() {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature and target vectors cannot be empty.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must be of the same length.");
//     }
//     for (int t : target) {
//       if (t != 0 && t != 1) {
//         throw std::invalid_argument("Target vector must be binary (0 and 1).");
//       }
//     }
//     total_pos = std::accumulate(target.begin(), target.end(), 0.0);
//     total_neg = feature.size() - total_pos;
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::invalid_argument("Target vector must contain both classes (0 and 1).");
//     }
//     if (min_bins < 1) {
//       throw std::invalid_argument("min_bins must be at least 1.");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff < 0 || bin_cutoff > 0.5) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 0.5.");
//     }
//     if (max_n_prebins < min_bins) {
//       max_n_prebins = min_bins;
//       Rcpp::warning("max_n_prebins adjusted to be at least min_bins.");
//     }
//   }
// 
//   void initial_binning() {
//     // Sort feature values along with target
//     std::vector<std::pair<double, int>> feature_target(feature.size());
//     for (size_t i = 0; i < feature.size(); ++i) {
//       feature_target[i] = std::make_pair(feature[i], target[i]);
//     }
//     std::sort(feature_target.begin(), feature_target.end());
// 
//     bin_edges.clear();
//     bin_edges.push_back(-std::numeric_limits<double>::infinity());
// 
//     size_t n = feature_target.size();
//     size_t n_distinct = 1;
//     for (size_t i = 1; i < n; ++i) {
//       if (feature_target[i].first != feature_target[i-1].first) {
//         n_distinct++;
//       }
//     }
// 
//     if (n_distinct <= static_cast<size_t>(min_bins)) {
//       // If number of distinct values is less than or equal to min_bins, use all unique values
//       for (size_t i = 1; i < n; ++i) {
//         if (feature_target[i].first != feature_target[i-1].first) {
//           bin_edges.push_back(feature_target[i].first);
//         }
//       }
//     } else {
//       size_t n_bins = std::min<size_t>(max_n_prebins, n_distinct);
//       size_t bin_size = n / n_bins;
// 
//       // Collect bin edges based on quantiles
//       for (size_t i = 1; i < n_bins; ++i) {
//         size_t idx = i * bin_size;
//         if (idx >= n) idx = n - 1;
//         double edge = feature_target[idx].first;
//         // Avoid duplicate edges
//         if (edge > bin_edges.back()) {
//           bin_edges.push_back(edge);
//         }
//       }
//     }
//     bin_edges.push_back(std::numeric_limits<double>::infinity());
// 
//     // Ensure we have at least min_bins+1 bin edges
//     while (bin_edges.size() - 1 < static_cast<size_t>(min_bins)) {
//       // Split bins with largest counts
//       count.assign(bin_edges.size() - 1, 0);
//       for (const auto& ft : feature_target) {
//         int bin_idx = find_bin(ft.first);
//         if (bin_idx >= 0 && bin_idx < static_cast<int>(count.size())) {
//           count[bin_idx]++;
//         }
//       }
//       // Find bin with largest count
//       size_t max_count_idx = std::distance(count.begin(), std::max_element(count.begin(), count.end()));
//       double lower = bin_edges[max_count_idx];
//       double upper = bin_edges[max_count_idx + 1];
// 
//       // Find median value in this bin
//       std::vector<double> bin_values;
//       for (const auto& ft : feature_target) {
//         if (ft.first > lower && ft.first <= upper) {
//           bin_values.push_back(ft.first);
//         }
//       }
//       if (bin_values.empty()) break; // Cannot split further
// 
//       // Split at median
//       std::nth_element(bin_values.begin(), bin_values.begin() + bin_values.size() / 2, bin_values.end());
//       double median = bin_values[bin_values.size() / 2];
//       if (median <= lower || median >= upper) break; // Cannot split further
// 
//       // Insert new edge
//       bin_edges.insert(bin_edges.begin() + max_count_idx + 1, median);
//     }
//   }
// 
//   int find_bin(double value) const {
//     auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), value);
//     int idx = static_cast<int>(std::distance(bin_edges.begin(), it)) - 1;
//     if (idx < 0) idx = 0;
//     if (idx >= static_cast<int>(bin_edges.size() - 1)) idx = bin_edges.size() - 2;
//     return idx;
//   }
// 
//   bool is_monotonic(const std::vector<double>& woe) const {
//     if (woe.size() < 2) return true;
//     bool increasing = true, decreasing = true;
//     for (size_t i = 1; i < woe.size(); ++i) {
//       if (woe[i] < woe[i-1]) increasing = false;
//       if (woe[i] > woe[i-1]) decreasing = false;
//       if (!increasing && !decreasing) return false;
//     }
//     return true;
//   }
// 
//   void merge_bins() {
//     iterations = 0;
//     converged = false;
// 
//     while (iterations++ < max_iterations && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
//       bool merged = false;
// 
//       // Recalculate counts and WoE
//       calculate_initial_woe();
// 
//       // Check for bins below bin_cutoff
//       for (size_t i = 0; i < count.size(); ++i) {
//         double freq = static_cast<double>(count[i]) / feature.size();
//         if (freq < bin_cutoff && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
//           if (i == 0 && count.size() > 1) {
//             merge_with_next(0);
//           } else if (i > 0) {
//             merge_with_prev(i);
//           }
//           merged = true;
//           break;
//         }
//       }
//       if (merged) continue;
// 
//       // Check for monotonicity
//       if (!is_monotonic(woe_values) && bin_edges.size() > static_cast<size_t>(min_bins + 1)) {
//         size_t break_point = 1;
//         bool increasing = woe_values[1] > woe_values[0];
//         for (; break_point < woe_values.size(); ++break_point) {
//           if ((increasing && woe_values[break_point] < woe_values[break_point-1]) ||
//               (!increasing && woe_values[break_point] > woe_values[break_point-1])) {
//             break;
//           }
//         }
//         if (break_point < woe_values.size()) {
//           merge_with_prev(break_point);
//           merged = true;
//         }
//       }
// 
//       if (!merged) {
//         converged = true;
//         break; // If no merges were performed, we're done
//       }
// 
//       // Check for convergence
//       if (std::abs(calculate_total_iv() - calculate_total_iv()) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//     }
// 
//     if (iterations >= max_iterations) {
//       Rcpp::warning("Maximum iterations reached in merge_bins. Results may be suboptimal.");
//     }
//   }
// 
//   void merge_with_prev(size_t i) {
//     if (i <= 0 || i >= bin_edges.size() - 1) return;
//     bin_edges.erase(bin_edges.begin() + i);
//     count[i-1] += count[i];
//     count_pos[i-1] += count_pos[i];
//     count_neg[i-1] += count_neg[i];
//     count.erase(count.begin() + i);
//     count_pos.erase(count_pos.begin() + i);
//     count_neg.erase(count_neg.begin() + i);
//     woe_values.erase(woe_values.begin() + i);
//     iv_values.erase(iv_values.begin() + i);
//   }
// 
//   void merge_with_next(size_t i) {
//     if (i >= bin_edges.size() - 2) return;
//     bin_edges.erase(bin_edges.begin() + i + 1);
//     count[i] += count[i+1];
//     count_pos[i] += count_pos[i+1];
//     count_neg[i] += count_neg[i+1];
//     count.erase(count.begin() + i + 1);
//     count_pos.erase(count_pos.begin() + i + 1);
//     count_neg.erase(count_neg.begin() + i + 1);
//     woe_values.erase(woe_values.begin() + i + 1);
//     iv_values.erase(iv_values.begin() + i + 1);
//   }
// 
//   void optimize_bins() {
//     // Initialize counts
//     count.assign(bin_edges.size() - 1, 0);
//     count_pos.assign(bin_edges.size() - 1, 0);
//     count_neg.assign(bin_edges.size() - 1, 0);
// 
//     // Count observations in each bin
//     for (size_t i = 0; i < feature.size(); ++i) {
//       int bin_idx = find_bin(feature[i]);
//       if (bin_idx >= 0 && bin_idx < static_cast<int>(count.size())) {
//         count[bin_idx]++;
//         if (target[i] == 1) {
//           count_pos[bin_idx]++;
//         } else {
//           count_neg[bin_idx]++;
//         }
//       }
//     }
// 
//     calculate_initial_woe();
//     merge_bins();
// 
//     while (count.size() > static_cast<size_t>(max_bins)) {
//       // Merge bins with smallest IV
//       double min_iv = std::numeric_limits<double>::max();
//       size_t merge_idx = 0;
//       for (size_t i = 0; i < iv_values.size() - 1; ++i) {
//         double combined_iv = iv_values[i] + iv_values[i+1];
//         if (combined_iv < min_iv) {
//           min_iv = combined_iv;
//           merge_idx = i;
//         }
//       }
//       merge_with_next(merge_idx);
//       calculate_initial_woe();
//     }
//   }
// 
//   void calculate_initial_woe() {
//     total_pos = std::accumulate(count_pos.begin(), count_pos.end(), 0.0);
//     total_neg = std::accumulate(count_neg.begin(), count_neg.end(), 0.0);
//     woe_values.resize(count.size());
//     iv_values.resize(count.size());
//     for (size_t i = 0; i < count.size(); ++i) {
//       double pct_pos = static_cast<double>(count_pos[i]) / total_pos;
//       double pct_neg = static_cast<double>(count_neg[i]) / total_neg;
//       pct_pos = std::max(pct_pos, EPSILON);
//       pct_neg = std::max(pct_neg, EPSILON);
//       woe_values[i] = std::log(pct_pos / pct_neg);
//       iv_values[i] = (pct_pos - pct_neg) * woe_values[i];
//     }
//   }
// 
//   void calculate_woe_iv() {
//     calculate_initial_woe();
//   }
// 
//   double calculate_total_iv() const {
//     return std::accumulate(iv_values.begin(), iv_values.end(), 0.0);
//   }
// };
// 
// 
// //' Perform Optimal Binning for Numerical Features using Monotonic Optimal Binning (MOB)
// //'
// //' This function implements the Monotonic Optimal Binning algorithm for numerical features.
// //' It creates optimal bins while maintaining monotonicity in the Weight of Evidence (WoE) values.
// //'
// //' @param target An integer vector of binary target values (0 or 1)
// //' @param feature A numeric vector of feature values to be binned
// //' @param min_bins Minimum number of bins to create (default: 3)
// //' @param max_bins Maximum number of bins to create (default: 5)
// //' @param bin_cutoff Minimum frequency of observations in a bin (default: 0.05)
// //' @param max_n_prebins Maximum number of prebins to create initially (default: 20)
// //' @param convergence_threshold Threshold for convergence in the iterative process (default: 1e-6)
// //' @param max_iterations Maximum number of iterations for the binning process (default: 1000)
// //'
// //' @return A list containing the following elements:
// //'   \item{bins}{A character vector of bin labels}
// //'   \item{woe}{A numeric vector of Weight of Evidence values for each bin}
// //'   \item{iv}{A numeric vector of Information Value for each bin}
// //'   \item{count}{An integer vector of total count of observations in each bin}
// //'   \item{count_pos}{An integer vector of count of positive class observations in each bin}
// //'   \item{count_neg}{An integer vector of count of negative class observations in each bin}
// //'   \item{cutpoints}{A numeric vector of cutpoints used to create the bins}
// //'   \item{converged}{A logical value indicating whether the algorithm converged}
// //'   \item{iterations}{An integer value indicating the number of iterations run}
// //'
// //' @details
// //' The algorithm starts by creating initial bins and then iteratively merges them
// //' to achieve optimal binning while maintaining monotonicity in the WoE values.
// //' It respects the minimum and maximum number of bins specified.
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(42)
// //' feature <- rnorm(1000)
// //' target <- rbinom(1000, 1, 0.5)
// //' result <- optimal_binning_numerical_mob(target, feature)
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_numerical_mob(IntegerVector target, NumericVector feature,
//                                    int min_bins = 3, int max_bins = 5,
//                                    double bin_cutoff = 0.05, int max_n_prebins = 20,
//                                    double convergence_threshold = 1e-6, int max_iterations = 1000) {
//   if (feature.size() != target.size()) {
//     stop("Feature and target vectors must be of the same length.");
//   }
// 
//   // Ensure max_bins is at least equal to min_bins
//   max_bins = std::max(min_bins, max_bins);
// 
//   std::vector<double> feature_vec = as<std::vector<double>>(feature);
//   std::vector<int> target_vec = as<std::vector<int>>(target);
// 
//   OptimalBinningNumericalMOB binning(min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                      convergence_threshold, max_iterations);
// 
//   try {
//     binning.fit(feature_vec, target_vec);
//   } catch (const std::exception& e) {
//     Rcpp::stop(std::string("Error in binning process: ") + e.what());
//   }
// 
//   std::vector<BinMetrics> bins = binning.get_bin_metrics();
// 
//   std::vector<std::string> bin_names;
//   std::vector<double> bin_woe, bin_iv;
//   std::vector<int> bin_count, bin_count_pos, bin_count_neg;
//   std::vector<double> bin_cutpoints = binning.get_cutpoints();
// 
//   for (const auto& b : bins) {
//     std::string lower_str = (std::isfinite(b.lower)) ? std::to_string(b.lower) : "-Inf";
//     std::string upper_str = (std::isfinite(b.upper)) ? std::to_string(b.upper) : "+Inf";
//     bin_names.push_back("(" + lower_str + ";" + upper_str + "]");
//     bin_woe.push_back(b.woe);
//     bin_iv.push_back(b.iv);
//     bin_count.push_back(b.count);
//     bin_count_pos.push_back(b.count_pos);
//     bin_count_neg.push_back(b.count_neg);
//   }
// 
//   return Rcpp::List::create(
//     Rcpp::Named("bin") = bin_names,
//     Rcpp::Named("woe") = bin_woe,
//     Rcpp::Named("iv") = bin_iv,
//     Rcpp::Named("count") = bin_count,
//     Rcpp::Named("count_pos") = bin_count_pos,
//     Rcpp::Named("count_neg") = bin_count_neg,
//     Rcpp::Named("cutpoints") = bin_cutpoints,
//     Rcpp::Named("converged") = binning.has_converged(),
//     Rcpp::Named("iterations") = binning.get_iterations()
//   );
// }
