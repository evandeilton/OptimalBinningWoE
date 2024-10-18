// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cmath>
#include <unordered_map>

using namespace Rcpp;

// Structure to hold category statistics
struct CategoryStats {
  std::string category;
  int count;
  int pos_count;
  int neg_count;
  double event_rate;
};

// Class for Optimal Binning Categorical IVB
class OptimalBinningCategoricalIVB {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  double bin_cutoff;
  int min_bins;
  int max_bins;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<CategoryStats> category_stats;
  std::vector<std::vector<double>> dp;
  std::vector<std::vector<int>> split_points;
  bool converged;
  int iterations_run;
  
  void validate_input();
  void preprocess_data();
  void merge_rare_categories();
  void ensure_max_prebins();
  void compute_and_sort_event_rates();
  void initialize_dp_structures();
  void perform_dynamic_programming();
  std::vector<int> backtrack_optimal_bins();
  double calculate_iv(int start, int end);
  bool check_monotonicity(const std::vector<int>& bins);
  void enforce_monotonicity(std::vector<int>& bins);
  
public:
  OptimalBinningCategoricalIVB(
    std::vector<std::string> feature,
    std::vector<int> target,
    double bin_cutoff,
    int min_bins,
    int max_bins,
    int max_n_prebins,
    std::string bin_separator,
    double convergence_threshold,
    int max_iterations
  );
  
  List perform_binning();
};

// Constructor
OptimalBinningCategoricalIVB::OptimalBinningCategoricalIVB(
  std::vector<std::string> feature,
  std::vector<int> target,
  double bin_cutoff,
  int min_bins,
  int max_bins,
  int max_n_prebins,
  std::string bin_separator,
  double convergence_threshold,
  int max_iterations
) : feature(feature), target(target), bin_cutoff(bin_cutoff), min_bins(min_bins),
max_bins(max_bins), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
convergence_threshold(convergence_threshold), max_iterations(max_iterations),
converged(false), iterations_run(0) {}

// Validate input parameters
void OptimalBinningCategoricalIVB::validate_input() {
  if (feature.empty() || target.empty()) {
    stop("Feature and target vectors cannot be empty");
  }
  if (feature.size() != target.size()) {
    stop("Feature and target vectors must have the same length");
  }
  if (min_bins < 2) {
    stop("min_bins must be at least 2");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins");
  }
  if (bin_cutoff < 0 || bin_cutoff > 1) {
    stop("bin_cutoff must be between 0 and 1");
  }
  for (int t : target) {
    if (t != 0 && t != 1) {
      stop("Target must be binary (0 or 1)");
    }
  }
}

// Preprocess data: count occurrences and compute event rates
void OptimalBinningCategoricalIVB::preprocess_data() {
  std::unordered_map<std::string, CategoryStats> stats_map;
  for (size_t i = 0; i < feature.size(); ++i) {
    auto& stats = stats_map[feature[i]];
    stats.category = feature[i];
    stats.count++;
    stats.pos_count += target[i];
    stats.neg_count += 1 - target[i];
  }
  
  category_stats.reserve(stats_map.size());
  for (const auto& pair : stats_map) {
    category_stats.push_back(pair.second);
  }
}

// Merge rare categories based on bin_cutoff
void OptimalBinningCategoricalIVB::merge_rare_categories() {
  double total_count = std::accumulate(category_stats.begin(), category_stats.end(), 0.0,
                                       [](double sum, const CategoryStats& stats) { return sum + stats.count; });
  
  std::vector<CategoryStats> merged_stats;
  std::vector<std::string> rare_categories;
  
  for (const auto& stats : category_stats) {
    if (stats.count / total_count >= bin_cutoff) {
      merged_stats.push_back(stats);
    } else {
      rare_categories.push_back(stats.category);
    }
  }
  
  if (!rare_categories.empty()) {
    CategoryStats merged_rare{"", 0, 0, 0, 0.0};
    for (const auto& cat : rare_categories) {
      if (!merged_rare.category.empty()) merged_rare.category += bin_separator;
      merged_rare.category += cat;
      const auto& stats = *std::find_if(category_stats.begin(), category_stats.end(),
                                        [&cat](const CategoryStats& s) { return s.category == cat; });
      merged_rare.count += stats.count;
      merged_rare.pos_count += stats.pos_count;
      merged_rare.neg_count += stats.neg_count;
    }
    merged_stats.push_back(merged_rare);
  }
  
  category_stats = std::move(merged_stats);
}

// Ensure max_n_prebins is not exceeded
void OptimalBinningCategoricalIVB::ensure_max_prebins() {
  if (static_cast<int>(category_stats.size()) > max_n_prebins) {
    std::partial_sort(category_stats.begin(), category_stats.begin() + max_n_prebins, category_stats.end(),
                      [](const CategoryStats& a, const CategoryStats& b) { return a.count > b.count; });
    category_stats.resize(max_n_prebins);
  }
}

// Compute event rates and sort categories
void OptimalBinningCategoricalIVB::compute_and_sort_event_rates() {
  for (auto& stats : category_stats) {
    stats.event_rate = static_cast<double>(stats.pos_count) / stats.count;
  }
  
  std::sort(category_stats.begin(), category_stats.end(),
            [](const CategoryStats& a, const CategoryStats& b) { return a.event_rate < b.event_rate; });
}

// Initialize dynamic programming structures
void OptimalBinningCategoricalIVB::initialize_dp_structures() {
  int n = category_stats.size();
  dp.resize(n + 1, std::vector<double>(max_bins + 1, -std::numeric_limits<double>::infinity()));
  split_points.resize(n + 1, std::vector<int>(max_bins + 1, 0));
  
  for (int i = 0; i <= n; ++i) {
    dp[i][1] = calculate_iv(0, i);
  }
}

// Perform dynamic programming to find optimal binning
void OptimalBinningCategoricalIVB::perform_dynamic_programming() {
  int n = category_stats.size();
  for (int k = 2; k <= max_bins; ++k) {
    for (int i = k; i <= n; ++i) {
      for (int j = k - 1; j < i; ++j) {
        double iv = dp[j][k - 1] + calculate_iv(j, i);
        if (iv > dp[i][k]) {
          dp[i][k] = iv;
          split_points[i][k] = j;
        }
      }
    }
  }
}

// Backtrack to find the optimal bins
std::vector<int> OptimalBinningCategoricalIVB::backtrack_optimal_bins() {
  int n = category_stats.size();
  int k = std::min(max_bins, n);
  std::vector<int> bins;
  
  while (k > 0) {
    bins.push_back(n);
    n = split_points[n][k];
    k--;
  }
  
  std::reverse(bins.begin(), bins.end());
  return bins;
}

// Calculate Information Value for a range of categories
double OptimalBinningCategoricalIVB::calculate_iv(int start, int end) {
  int pos = 0, neg = 0;
  for (int i = start; i < end; ++i) {
    pos += category_stats[i].pos_count;
    neg += category_stats[i].neg_count;
  }
  
  double total_pos = std::accumulate(category_stats.begin(), category_stats.end(), 0,
                                     [](int sum, const CategoryStats& stats) { return sum + stats.pos_count; });
  double total_neg = std::accumulate(category_stats.begin(), category_stats.end(), 0,
                                     [](int sum, const CategoryStats& stats) { return sum + stats.neg_count; });
  
  double pos_rate = pos / total_pos;
  double neg_rate = neg / total_neg;
  
  if (pos_rate == 0 || neg_rate == 0) {
    return 0;
  }
  
  double woe = std::log(pos_rate / neg_rate);
  return (pos_rate - neg_rate) * woe;
}

// Check if bins are monotonic in event rate
bool OptimalBinningCategoricalIVB::check_monotonicity(const std::vector<int>& bins) {
  double prev_event_rate = -1;
  int start = 0;
  for (int end : bins) {
    int pos = 0, total = 0;
    for (int i = start; i < end; ++i) {
      pos += category_stats[i].pos_count;
      total += category_stats[i].count;
    }
    double event_rate = static_cast<double>(pos) / total;
    if (event_rate < prev_event_rate) {
      return false;
    }
    prev_event_rate = event_rate;
    start = end;
  }
  return true;
}

// Enforce monotonicity in bins
void OptimalBinningCategoricalIVB::enforce_monotonicity(std::vector<int>& bins) {
  while (!check_monotonicity(bins) && bins.size() > min_bins) {
    auto it = std::adjacent_find(bins.begin(), bins.end(),
                                 [this](int a, int b) {
                                   double rate_a = static_cast<double>(category_stats[a-1].pos_count) / category_stats[a-1].count;
                                   double rate_b = static_cast<double>(category_stats[b-1].pos_count) / category_stats[b-1].count;
                                   return rate_a > rate_b;
                                 });
    if (it != bins.end() && std::next(it) != bins.end()) {
      bins.erase(std::next(it));
    } else {
      break;
    }
  }
}

// Perform the optimal binning
List OptimalBinningCategoricalIVB::perform_binning() {
  try {
    validate_input();
    preprocess_data();
    merge_rare_categories();
    ensure_max_prebins();
    compute_and_sort_event_rates();
    
    // Adjust max_bins if necessary
    max_bins = std::min(max_bins, static_cast<int>(category_stats.size()));
    
    std::vector<int> optimal_bins;
    
    // If max_bins is reached, no need to optimize
    if (static_cast<int>(category_stats.size()) <= max_bins) {
      converged = true;
      iterations_run = 1;
      optimal_bins.resize(category_stats.size());
      std::iota(optimal_bins.begin(), optimal_bins.end(), 1);
    } else {
      initialize_dp_structures();
      perform_dynamic_programming();
      
      optimal_bins = backtrack_optimal_bins();
      enforce_monotonicity(optimal_bins);
      
      // Check convergence
      double prev_iv = -std::numeric_limits<double>::infinity();
      for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
        double current_iv = dp[category_stats.size()][optimal_bins.size()];
        if (std::abs(current_iv - prev_iv) < convergence_threshold) {
          converged = true;
          break;
        }
        prev_iv = current_iv;
      }
    }
    
    // Prepare output
    std::vector<std::string> bin_names;
    std::vector<double> bin_iv, bin_woe;
    std::vector<int> bin_count, bin_count_pos, bin_count_neg;
    
    size_t start = 0;
    double total_pos = 0, total_neg = 0;
    for (const auto& stats : category_stats) {
      total_pos += stats.pos_count;
      total_neg += stats.neg_count;
    }
    
    for (size_t end : optimal_bins) {
      std::string bin_name;
      int count = 0, count_pos = 0, count_neg = 0;
      
      for (size_t i = start; i < end; ++i) {
        if (!bin_name.empty()) bin_name += bin_separator;
        bin_name += category_stats[i].category;
        count += category_stats[i].count;
        count_pos += category_stats[i].pos_count;
        count_neg += category_stats[i].neg_count;
      }
      
      bin_names.push_back(bin_name);
      bin_count.push_back(count);
      bin_count_pos.push_back(count_pos);
      bin_count_neg.push_back(count_neg);
      
      double pos_rate = count_pos / total_pos;
      double neg_rate = count_neg / total_neg;
      double woe = (pos_rate > 0 && neg_rate > 0) ? std::log(pos_rate / neg_rate) : 0;
      double iv = (pos_rate - neg_rate) * woe;
      
      bin_woe.push_back(woe);
      bin_iv.push_back(iv);
      
      start = end;
    }
    
    return List::create(
      Named("bin") = bin_names,
      Named("woe") = bin_woe,
      Named("iv") = bin_iv,
      Named("count") = bin_count,
      Named("count_pos") = bin_count_pos,
      Named("count_neg") = bin_count_neg,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}


//' @title Optimal Binning for Categorical Variables using Information Value-based Binning (IVB)
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector or factor of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator for merged category names (default: "%;%").
//' @param convergence_threshold Convergence threshold for optimization (default: 1e-6).
//' @param max_iterations Maximum number of iterations for optimization (default: 1000).
//'
//' @return A list containing:
//' \itemize{
//'   \item bins: Character vector of bin names.
//'   \item iv: Numeric vector of Information Values for each bin.
//'   \item count: Integer vector of total counts for each bin.
//'   \item count_pos: Integer vector of positive target counts for each bin.
//'   \item count_neg: Integer vector of negative target counts for each bin.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations run.
//' }
//'
//' @details
//' This function performs optimal binning for categorical variables using a dynamic
//' programming approach. It aims to maximize the total Information Value while
//' respecting the constraints on the number of bins. The algorithm handles rare
//' categories, ensures monotonicity in event rates, and provides stable results.
//'
//' @examples
//' \dontrun{
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//' result <- optimal_binning_categorical_ivb(target, feature, min_bins = 2, max_bins = 4)
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_categorical_ivb(
    IntegerVector target,
    SEXP feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000) {
  
  // Convert R types to C++ types
  std::vector<int> target_vec = as<std::vector<int>>(target);
  std::vector<std::string> feature_vec;
  
  // Handle factor or character vector input for feature
  if (Rf_isFactor(feature)) {
    IntegerVector levels = as<IntegerVector>(feature);
    CharacterVector level_names = levels.attr("levels");
    for (int i = 0; i < levels.size(); ++i) {
      feature_vec.push_back(as<std::string>(level_names[levels[i] - 1]));
    }
  } else if (TYPEOF(feature) == STRSXP) {
    feature_vec = as<std::vector<std::string>>(feature);
  } else {
    stop("feature must be a factor or character vector");
  }
  
  // Adjust max_bins if necessary
  int ncat = std::set<std::string>(feature_vec.begin(), feature_vec.end()).size();
  max_bins = std::min(max_bins, ncat);
  
  // Create and run the optimal binning algorithm
  OptimalBinningCategoricalIVB binner(
      feature_vec, target_vec, bin_cutoff, min_bins, max_bins, max_n_prebins,
      bin_separator, convergence_threshold, max_iterations
  );
  
  return binner.perform_binning();
}





// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::depends(RcppParallel)]]
// #include <Rcpp.h>
// #include <RcppParallel.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <numeric>
// #include <limits>
// #include <cmath>
// #include <unordered_map>
// 
// using namespace Rcpp;
// using namespace RcppParallel;
// 
// // Structure to hold category statistics
// struct CategoryStats {
//     std::string category;
//     int count;
//     int pos_count;
//     int neg_count;
//     double event_rate;
// };
// 
// // Class for Optimal Binning Categorical IVB
// class OptimalBinningCategoricalIVB {
// private:
//     std::vector<std::string> feature;
//     std::vector<int> target;
//     double bin_cutoff;
//     int min_bins;
//     int max_bins;
//     int max_n_prebins;
//     std::string bin_separator;
//     double convergence_threshold;
//     int max_iterations;
//     int nthreads;
// 
//     std::vector<CategoryStats> category_stats;
//     std::vector<std::vector<double>> dp;
//     std::vector<std::vector<int>> split_points;
//     bool converged;
//     int iterations_run;
// 
//     void validate_input();
//     void preprocess_data();
//     void merge_rare_categories();
//     void ensure_max_prebins();
//     void compute_and_sort_event_rates();
//     void initialize_dp_structures();
//     void perform_dynamic_programming();
//     std::vector<int> backtrack_optimal_bins();
//     double calculate_iv(int start, int end);
//     bool check_monotonicity(const std::vector<int>& bins);
//     void enforce_monotonicity(std::vector<int>& bins);
// 
// public:
//     OptimalBinningCategoricalIVB(
//         std::vector<std::string> feature,
//         std::vector<int> target,
//         double bin_cutoff,
//         int min_bins,
//         int max_bins,
//         int max_n_prebins,
//         std::string bin_separator,
//         double convergence_threshold,
//         int max_iterations,
//         int nthreads
//     );
// 
//     List perform_binning();
// };
// 
// // Constructor
// OptimalBinningCategoricalIVB::OptimalBinningCategoricalIVB(
//     std::vector<std::string> feature,
//     std::vector<int> target,
//     double bin_cutoff,
//     int min_bins,
//     int max_bins,
//     int max_n_prebins,
//     std::string bin_separator,
//     double convergence_threshold,
//     int max_iterations,
//     int nthreads
// ) : feature(feature), target(target), bin_cutoff(bin_cutoff), min_bins(min_bins),
//     max_bins(max_bins), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
//     convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//     nthreads(nthreads), converged(false), iterations_run(0) {}
// 
// // Validate input parameters
// void OptimalBinningCategoricalIVB::validate_input() {
//     if (feature.empty() || target.empty()) {
//         stop("Feature and target vectors cannot be empty");
//     }
//     if (feature.size() != target.size()) {
//         stop("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2) {
//         stop("min_bins must be at least 2");
//     }
//     if (max_bins < min_bins) {
//         stop("max_bins must be greater than or equal to min_bins");
//     }
//     if (bin_cutoff < 0 || bin_cutoff > 1) {
//         stop("bin_cutoff must be between 0 and 1");
//     }
//     for (int t : target) {
//         if (t != 0 && t != 1) {
//             stop("Target must be binary (0 or 1)");
//         }
//     }
// }
// 
// // Preprocess data: count occurrences and compute event rates
// void OptimalBinningCategoricalIVB::preprocess_data() {
//     std::unordered_map<std::string, CategoryStats> stats_map;
//     for (size_t i = 0; i < feature.size(); ++i) {
//         auto& stats = stats_map[feature[i]];
//         stats.category = feature[i];
//         stats.count++;
//         stats.pos_count += target[i];
//         stats.neg_count += 1 - target[i];
//     }
// 
//     category_stats.reserve(stats_map.size());
//     for (const auto& pair : stats_map) {
//         category_stats.push_back(pair.second);
//     }
// }
// 
// // Merge rare categories based on bin_cutoff
// void OptimalBinningCategoricalIVB::merge_rare_categories() {
//     double total_count = std::accumulate(category_stats.begin(), category_stats.end(), 0.0,
//         [](double sum, const CategoryStats& stats) { return sum + stats.count; });
// 
//     std::vector<CategoryStats> merged_stats;
//     std::vector<std::string> rare_categories;
// 
//     for (const auto& stats : category_stats) {
//         if (stats.count / total_count >= bin_cutoff) {
//             merged_stats.push_back(stats);
//         } else {
//             rare_categories.push_back(stats.category);
//         }
//     }
// 
//     if (!rare_categories.empty()) {
//         CategoryStats merged_rare{"", 0, 0, 0, 0.0};
//         for (const auto& cat : rare_categories) {
//             if (!merged_rare.category.empty()) merged_rare.category += bin_separator;
//             merged_rare.category += cat;
//             const auto& stats = *std::find_if(category_stats.begin(), category_stats.end(),
//                 [&cat](const CategoryStats& s) { return s.category == cat; });
//             merged_rare.count += stats.count;
//             merged_rare.pos_count += stats.pos_count;
//             merged_rare.neg_count += stats.neg_count;
//         }
//         merged_stats.push_back(merged_rare);
//     }
// 
//     category_stats = std::move(merged_stats);
// }
// 
// // Ensure max_n_prebins is not exceeded
// void OptimalBinningCategoricalIVB::ensure_max_prebins() {
//     if (static_cast<int>(category_stats.size()) > max_n_prebins) {
//         std::partial_sort(category_stats.begin(), category_stats.begin() + max_n_prebins, category_stats.end(),
//             [](const CategoryStats& a, const CategoryStats& b) { return a.count > b.count; });
//         category_stats.resize(max_n_prebins);
//     }
// }
// 
// // Compute event rates and sort categories
// void OptimalBinningCategoricalIVB::compute_and_sort_event_rates() {
//     for (auto& stats : category_stats) {
//         stats.event_rate = static_cast<double>(stats.pos_count) / stats.count;
//     }
// 
//     std::sort(category_stats.begin(), category_stats.end(),
//         [](const CategoryStats& a, const CategoryStats& b) { return a.event_rate < b.event_rate; });
// }
// 
// // Initialize dynamic programming structures
// void OptimalBinningCategoricalIVB::initialize_dp_structures() {
//     int n = category_stats.size();
//     dp.resize(n + 1, std::vector<double>(max_bins + 1, -std::numeric_limits<double>::infinity()));
//     split_points.resize(n + 1, std::vector<int>(max_bins + 1, 0));
// 
//     for (int i = 0; i <= n; ++i) {
//         dp[i][1] = calculate_iv(0, i);
//     }
// }
// 
// // Perform dynamic programming to find optimal binning
// void OptimalBinningCategoricalIVB::perform_dynamic_programming() {
//     int n = category_stats.size();
//     for (int k = 2; k <= max_bins; ++k) {
//         for (int i = k; i <= n; ++i) {
//             for (int j = k - 1; j < i; ++j) {
//                 double iv = dp[j][k - 1] + calculate_iv(j, i);
//                 if (iv > dp[i][k]) {
//                     dp[i][k] = iv;
//                     split_points[i][k] = j;
//                 }
//             }
//         }
//     }
// }
// 
// // Backtrack to find the optimal bins
// std::vector<int> OptimalBinningCategoricalIVB::backtrack_optimal_bins() {
//     int n = category_stats.size();
//     int k = std::min(max_bins, n);
//     std::vector<int> bins;
// 
//     while (k > 0) {
//         bins.push_back(n);
//         n = split_points[n][k];
//         k--;
//     }
// 
//     std::reverse(bins.begin(), bins.end());
//     return bins;
// }
// 
// // Calculate Information Value for a range of categories
// double OptimalBinningCategoricalIVB::calculate_iv(int start, int end) {
//     int pos = 0, neg = 0;
//     for (int i = start; i < end; ++i) {
//         pos += category_stats[i].pos_count;
//         neg += category_stats[i].neg_count;
//     }
// 
//     double total_pos = std::accumulate(category_stats.begin(), category_stats.end(), 0,
//         [](int sum, const CategoryStats& stats) { return sum + stats.pos_count; });
//     double total_neg = std::accumulate(category_stats.begin(), category_stats.end(), 0,
//         [](int sum, const CategoryStats& stats) { return sum + stats.neg_count; });
// 
//     double pos_rate = pos / total_pos;
//     double neg_rate = neg / total_neg;
// 
//     if (pos_rate == 0 || neg_rate == 0) {
//         return 0;
//     }
// 
//     double woe = std::log(pos_rate / neg_rate);
//     return (pos_rate - neg_rate) * woe;
// }
// 
// // Check if bins are monotonic in event rate
// bool OptimalBinningCategoricalIVB::check_monotonicity(const std::vector<int>& bins) {
//     double prev_event_rate = -1;
//     int start = 0;
//     for (int end : bins) {
//         int pos = 0, total = 0;
//         for (int i = start; i < end; ++i) {
//             pos += category_stats[i].pos_count;
//             total += category_stats[i].count;
//         }
//         double event_rate = static_cast<double>(pos) / total;
//         if (event_rate < prev_event_rate) {
//             return false;
//         }
//         prev_event_rate = event_rate;
//         start = end;
//     }
//     return true;
// }
// 
// // Enforce monotonicity in bins
// void OptimalBinningCategoricalIVB::enforce_monotonicity(std::vector<int>& bins) {
//     while (!check_monotonicity(bins) && bins.size() > min_bins) {
//         auto it = std::adjacent_find(bins.begin(), bins.end(),
//             [this](int a, int b) {
//                 double rate_a = static_cast<double>(category_stats[a-1].pos_count) / category_stats[a-1].count;
//                 double rate_b = static_cast<double>(category_stats[b-1].pos_count) / category_stats[b-1].count;
//                 return rate_a > rate_b;
//             });
//         if (it != bins.end() && std::next(it) != bins.end()) {
//             bins.erase(std::next(it));
//         } else {
//             break;
//         }
//     }
// }
// 
// // Perform the optimal binning
// List OptimalBinningCategoricalIVB::perform_binning() {
//     try {
//         validate_input();
//         preprocess_data();
//         merge_rare_categories();
//         ensure_max_prebins();
//         compute_and_sort_event_rates();
// 
//         // Adjust max_bins if necessary
//         max_bins = std::min(max_bins, static_cast<int>(category_stats.size()));
// 
//         std::vector<int> optimal_bins;
// 
//         // If max_bins is reached, no need to optimize
//         if (static_cast<int>(category_stats.size()) <= max_bins) {
//             converged = true;
//             iterations_run = 1;
//             optimal_bins.resize(category_stats.size());
//             std::iota(optimal_bins.begin(), optimal_bins.end(), 1);
//         } else {
//             initialize_dp_structures();
//             perform_dynamic_programming();
// 
//             optimal_bins = backtrack_optimal_bins();
//             enforce_monotonicity(optimal_bins);
// 
//             // Check convergence
//             double prev_iv = -std::numeric_limits<double>::infinity();
//             for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
//                 double current_iv = dp[category_stats.size()][optimal_bins.size()];
//                 if (std::abs(current_iv - prev_iv) < convergence_threshold) {
//                     converged = true;
//                     break;
//                 }
//                 prev_iv = current_iv;
//             }
//         }
// 
//         // Prepare output
//         std::vector<std::string> bin_names;
//         std::vector<double> bin_iv, bin_woe;
//         std::vector<int> bin_count, bin_count_pos, bin_count_neg;
// 
//         size_t start = 0;
//         double total_pos = 0, total_neg = 0;
//         for (const auto& stats : category_stats) {
//             total_pos += stats.pos_count;
//             total_neg += stats.neg_count;
//         }
// 
//         for (size_t end : optimal_bins) {
//             std::string bin_name;
//             int count = 0, count_pos = 0, count_neg = 0;
// 
//             for (size_t i = start; i < end; ++i) {
//                 if (!bin_name.empty()) bin_name += bin_separator;
//                 bin_name += category_stats[i].category;
//                 count += category_stats[i].count;
//                 count_pos += category_stats[i].pos_count;
//                 count_neg += category_stats[i].neg_count;
//             }
// 
//             bin_names.push_back(bin_name);
//             bin_count.push_back(count);
//             bin_count_pos.push_back(count_pos);
//             bin_count_neg.push_back(count_neg);
// 
//             double pos_rate = count_pos / total_pos;
//             double neg_rate = count_neg / total_neg;
//             double woe = (pos_rate > 0 && neg_rate > 0) ? std::log(pos_rate / neg_rate) : 0;
//             double iv = (pos_rate - neg_rate) * woe;
// 
//             bin_woe.push_back(woe);
//             bin_iv.push_back(iv);
// 
//             start = end;
//         }
// 
//         return List::create(
//             Named("bins") = bin_names,
//             Named("woe") = bin_woe,
//             Named("iv") = bin_iv,
//             Named("count") = bin_count,
//             Named("count_pos") = bin_count_pos,
//             Named("count_neg") = bin_count_neg,
//             Named("converged") = converged,
//             Named("iterations") = iterations_run
//         );
//     } catch (const std::exception& e) {
//         Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
// }
// 
// //' @param feature Character vector or factor of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator for merged category names (default: "%;%").
// //' @param convergence_threshold Convergence threshold for optimization (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for optimization (default: 1000).
// //' @param nthreads Number of threads for parallel processing (default: 0, meaning auto-detect).
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bins: Character vector of bin names.
// //'   \item iv: Numeric vector of Information Values for each bin.
// //'   \item count: Integer vector of total counts for each bin.
// //'   \item count_pos: Integer vector of positive target counts for each bin.
// //'   \item count_neg: Integer vector of negative target counts for each bin.
// //'   \item converged: Logical indicating whether the algorithm converged.
// //'   \item iterations: Integer indicating the number of iterations run.
// //' }
// //'
// //' @details
// //' This function performs optimal binning for categorical variables using a dynamic
// //' programming approach. It aims to maximize the total Information Value while
// //' respecting the constraints on the number of bins. The algorithm handles rare
// //' categories, ensures monotonicity in event rates, and provides stable results.
// //'
// //' @examples
// //' \dontrun{
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //' result <- optimal_binning_categorical_ivb(target, feature, min_bins = 2, max_bins = 4)
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// List optimal_binning_categorical_ivb(
//     IntegerVector target,
//     SEXP feature,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     std::string bin_separator = "%;%",
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000,
//     int nthreads = 0) {
// 
//     // Convert R types to C++ types
//     std::vector<int> target_vec = as<std::vector<int>>(target);
//     std::vector<std::string> feature_vec;
// 
//     // Handle factor or character vector input for feature
//     if (Rf_isFactor(feature)) {
//         IntegerVector levels = as<IntegerVector>(feature);
//         CharacterVector level_names = levels.attr("levels");
//         for (int i = 0; i < levels.size(); ++i) {
//             feature_vec.push_back(as<std::string>(level_names[levels[i] - 1]));
//         }
//     } else if (TYPEOF(feature) == STRSXP) {
//         feature_vec = as<std::vector<std::string>>(feature);
//     } else {
//         stop("feature must be a factor or character vector");
//     }
// 
//     // Adjust max_bins if necessary
//     int ncat = std::set<std::string>(feature_vec.begin(), feature_vec.end()).size();
//     max_bins = std::min(max_bins, ncat);
// 
//     // Create and run the optimal binning algorithm
//     OptimalBinningCategoricalIVB binner(
//         feature_vec, target_vec, bin_cutoff, min_bins, max_bins, max_n_prebins,
//         bin_separator, convergence_threshold, max_iterations, nthreads
//     );
// 
//     return binner.perform_binning();
// }