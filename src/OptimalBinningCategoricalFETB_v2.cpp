// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace Rcpp;

// Structure to hold bin information
struct BinInfo {
  std::unordered_set<std::string> categories;
  size_t count = 0;
  size_t count_pos = 0;
  size_t count_neg = 0;
  double woe = 0.0;
  double iv = 0.0;
};

class OptimalBinningCategoricalFETB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  const size_t min_bins;
  size_t max_bins;
  const double bin_cutoff;
  const size_t max_n_prebins;
  const double convergence_threshold;
  const size_t max_iterations;
  
  std::string bin_separator; // Separator for bin labels
  
  std::unordered_map<std::string, std::string> original_to_merged_category;
  std::unordered_map<std::string, std::unordered_set<std::string>> merged_to_original_categories;
  
  std::vector<BinInfo> bins;
  size_t total_pos = 0;
  size_t total_neg = 0;
  
  static constexpr double EPSILON = 1e-10;
  std::vector<double> log_factorials;
  
  bool converged = false;
  size_t iterations = 0;
  
  // Calculate Weight of Evidence (WOE) and Information Value (IV) for a bin
  void calculateWoeIv(BinInfo& bin) const {
    if (bin.count_pos == 0 || bin.count_neg == 0 || total_pos == 0 || total_neg == 0) {
      bin.woe = 0.0;
      bin.iv = 0.0;
      return;
    }
    
    double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
    double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
    
    bin.woe = std::log((dist_pos + EPSILON) / (dist_neg + EPSILON));
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
  
  // Perform Fisher's Exact Test
  double fisherExactTest(size_t a, size_t b, size_t c, size_t d) const {
    size_t n = a + b + c + d;
    size_t row1 = a + b;
    size_t row2 = c + d;
    size_t col1 = a + c;
    size_t col2 = b + d;
    
    double log_p = log_factorials[row1] + log_factorials[row2] + log_factorials[col1] + log_factorials[col2]
    - log_factorials[n] - log_factorials[a] - log_factorials[b] - log_factorials[c] - log_factorials[d];
    
    return std::exp(log_p);
  }
  
  // Merge rare categories based on bin_cutoff
  void mergeRareCategories() {
    std::unordered_map<std::string, size_t> category_counts;
    size_t total_count = feature.size();
    
    for (const auto& cat : feature) {
      category_counts[cat]++;
    }
    
    std::vector<std::pair<std::string, size_t>> sorted_categories(category_counts.begin(), category_counts.end());
    std::sort(sorted_categories.begin(), sorted_categories.end(),
              [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
                return a.second > b.second;
              });
    
    for (const auto& [cat, count] : sorted_categories) {
      double frequency = static_cast<double>(count) / total_count;
      if (frequency < bin_cutoff) {
        original_to_merged_category[cat] = sorted_categories.front().first;
      } else {
        original_to_merged_category[cat] = cat;
      }
    }
    
    for (const auto& [original_cat, merged_cat] : original_to_merged_category) {
      merged_to_original_categories[merged_cat].insert(original_cat);
    }
  }
  
  // Initialize bins
  void initializeBins() {
    std::unordered_map<std::string, size_t> bin_indices;
    bins.clear();
    
    for (const auto& [merged_cat, original_cats] : merged_to_original_categories) {
      bin_indices[merged_cat] = bins.size();
      bins.emplace_back();
      bins.back().categories = original_cats;
    }
    
    // Populate bin counts
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& original_cat = feature[i];
      auto it = original_to_merged_category.find(original_cat);
      if (it == original_to_merged_category.end()) continue;
      
      const std::string& merged_cat = it->second;
      auto bin_it = bin_indices.find(merged_cat);
      if (bin_it == bin_indices.end()) continue;
      
      size_t bin_index = bin_it->second;
      BinInfo& bin = bins[bin_index];
      
      bin.count++;
      if (target[i] == 1) {
        bin.count_pos++;
      } else {
        bin.count_neg++;
      }
    }
    
    for (auto& bin : bins) {
      calculateWoeIv(bin);
    }
    
    std::sort(bins.begin(), bins.end(), [&](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
  }
  
  // Merge bins to achieve monotonicity and respect min_bins
  void mergeBins() {
    while (bins.size() > min_bins) {
      bool merged = false;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        if (bins[i].woe > bins[i + 1].woe) {
          mergeTwoBins(i, i + 1);
          merged = true;
          break;
        }
      }
      if (!merged) break;
    }
  }
  
  // Merge two adjacent bins
  void mergeTwoBins(size_t index1, size_t index2) {
    BinInfo& bin1 = bins[index1];
    BinInfo& bin2 = bins[index2];
    
    bin1.count += bin2.count;
    bin1.count_pos += bin2.count_pos;
    bin1.count_neg += bin2.count_neg;
    bin1.categories.insert(bin2.categories.begin(), bin2.categories.end());
    
    calculateWoeIv(bin1);
    
    bins.erase(bins.begin() + index2);
  }
  
  // Join strings with a delimiter
  std::string joinStrings(const std::unordered_set<std::string>& strings) const {
    std::ostringstream oss;
    size_t i = 0;
    for (const auto& s : strings) {
      if (i > 0) oss << bin_separator;
      oss << s;
      i++;
    }
    return oss.str();
  }
  
public:
  OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
                                const std::vector<int>& target,
                                size_t min_bins = 3,
                                size_t max_bins = 5,
                                double bin_cutoff = 0.05,
                                size_t max_n_prebins = 20,
                                double convergence_threshold = 1e-6,
                                size_t max_iterations = 1000,
                                const std::string& bin_separator_input = "%;%" )
    : feature(feature), target(target),
      min_bins(min_bins), max_bins(max_bins),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
      convergence_threshold(convergence_threshold), max_iterations(max_iterations),
      bin_separator(bin_separator_input) {
    
    validateInput();
    
    // Precalculate logarithms of factorials
    size_t max_n = feature.size();
    log_factorials.resize(max_n + 1);
    log_factorials[0] = 0.0;
    for (size_t i = 1; i <= max_n; ++i) {
      log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
    }
    
    for (int val : target) {
      if (val == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
  }
  
  // Validate input parameters
  void validateInput() const {
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins");
    }
    for (int val : target) {
      if (val != 0 && val != 1) {
        throw std::invalid_argument("Target vector must contain only binary values (0 and 1)");
      }
    }
  }
  
  // Perform optimal binning
  void fit() {
    mergeRareCategories();
    initializeBins();
    
    size_t ncat = merged_to_original_categories.size();
    max_bins = std::min(max_bins, ncat);
    
    if (ncat <= max_bins) {
      converged = true;
      return;
    }
    
    double prev_total_iv = 0.0;
    while (bins.size() > max_bins && iterations < max_iterations) {
      double min_p_value = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        size_t a = bins[i].count_pos;
        size_t b = bins[i].count_neg;
        size_t c = bins[i + 1].count_pos;
        size_t d = bins[i + 1].count_neg;
        
        double p_value = fisherExactTest(a, b, c, d);
        
        if (p_value < min_p_value) {
          min_p_value = p_value;
          merge_index = i;
        }
      }
      
      mergeTwoBins(merge_index, merge_index + 1);
      
      double total_iv = 0.0;
      for (const auto& bin : bins) {
        total_iv += bin.iv;
      }
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = total_iv;
      iterations++;
    }
    
    mergeBins();
  }
  
  // Get the results of optimal binning
  Rcpp::List getResults() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    
    bin_labels.reserve(bins.size());
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    counts.reserve(bins.size());
    counts_pos.reserve(bins.size());
    counts_neg.reserve(bins.size());
    
    for (const auto& bin : bins) {
      std::string label = joinStrings(bin.categories);
      bin_labels.emplace_back(label);
      woe_values.emplace_back(bin.woe);
      iv_values.emplace_back(bin.iv);
      counts.emplace_back(static_cast<int>(bin.count));
      counts_pos.emplace_back(static_cast<int>(bin.count_pos));
      counts_neg.emplace_back(static_cast<int>(bin.count_neg));
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = static_cast<int>(iterations)
    );
  }
};

//' Categorical Optimal Binning with Fisher's Exact Test
//'
//' Implements optimal binning for categorical variables using Fisher's Exact Test,
//' calculating Weight of Evidence (WoE) and Information Value (IV).
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations (default: 1000).
//' @param bin_separator Separator for bin labels (default: "%;%").
//'
//' @return A list containing:
//' \itemize{
//'   \item bin: Character vector of bin labels (merged categories).
//'   \item woe: Numeric vector of Weight of Evidence values for each bin.
//'   \item iv: Numeric vector of Information Value for each bin.
//'   \item count: Integer vector of total count in each bin.
//'   \item count_pos: Integer vector of positive class count in each bin.
//'   \item count_neg: Integer vector of negative class count in each bin.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations performed.
//' }
//'
//' @details
//' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
//' the statistical significance of the difference between adjacent bins. It ensures
//' monotonicity in the resulting bins and respects the minimum number of bins specified.
//'
//' @examples
//' \dontrun{
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2, 
//' max_bins = 4, bin_separator = "|")
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_fetb(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000,
   std::string bin_separator = "%;%"
) {
 // Convert Rcpp vectors to std::vector
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 try {
   // Create and run the binner
   OptimalBinningCategoricalFETB binner(
       feature_vec, target_vec,
       static_cast<size_t>(min_bins),
       static_cast<size_t>(max_bins),
       bin_cutoff,
       static_cast<size_t>(max_n_prebins),
       convergence_threshold,
       static_cast<size_t>(max_iterations),
       bin_separator
   );
   
   binner.fit();
   
   // Return the results
   return binner.getResults();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
 


// // [[Rcpp::plugins(cpp11)]]
// // [[Rcpp::plugins(openmp)]]
// 
// // [[Rcpp::depends(RcppParallel)]]
// #include <Rcpp.h>
// #include <RcppParallel.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <unordered_map>
// #include <unordered_set>
// #include <limits>
// #include <sstream>
// #include <stdexcept>
// 
// using namespace Rcpp;
// using namespace RcppParallel;
// 
// // Structure to hold bin information
// struct BinInfo {
//     std::unordered_set<std::string> categories;
//     size_t count = 0;
//     size_t count_pos = 0;
//     size_t count_neg = 0;
//     double woe = 0.0;
//     double iv = 0.0;
// };
// 
// // Worker for parallel processing
// struct BinWorker : public Worker {
//     // Input data
//     const std::vector<std::string>& feature;
//     const std::vector<int>& target;
//     const std::unordered_map<std::string, std::string>& original_to_merged_category;
//     
//     // Output data
//     std::vector<BinInfo>& bins;
//     std::unordered_map<std::string, size_t>& bin_indices;
//     
//     // Constructor
//     BinWorker(const std::vector<std::string>& feature,
//               const std::vector<int>& target,
//               const std::unordered_map<std::string, std::string>& original_to_merged_category,
//               std::vector<BinInfo>& bins,
//               std::unordered_map<std::string, size_t>& bin_indices)
//         : feature(feature), target(target), original_to_merged_category(original_to_merged_category),
//           bins(bins), bin_indices(bin_indices) {}
//     
//     // Parallel operator
//     void operator()(std::size_t begin, std::size_t end) {
//         for (size_t i = begin; i < end; ++i) {
//             const std::string& original_cat = feature[i];
//             auto it = original_to_merged_category.find(original_cat);
//             if (it == original_to_merged_category.end()) continue;
//             
//             const std::string& merged_cat = it->second;
//             auto bin_it = bin_indices.find(merged_cat);
//             if (bin_it == bin_indices.end()) continue;
//             
//             size_t bin_index = bin_it->second;
//             BinInfo& bin = bins[bin_index];
//             
//             bin.count++;
//             if (target[i] == 1) {
//                 bin.count_pos++;
//             } else {
//                 bin.count_neg++;
//             }
//         }
//     }
// };
// 
// class OptimalBinningCategoricalFETB {
// private:
//     const std::vector<std::string>& feature;
//     const std::vector<int>& target;
//     const size_t min_bins;
//     size_t max_bins;
//     const double bin_cutoff;
//     const size_t max_n_prebins;
//     const double convergence_threshold;
//     const size_t max_iterations;
//     const int nthreads;
//     
//     std::unordered_map<std::string, std::string> original_to_merged_category;
//     std::unordered_map<std::string, std::unordered_set<std::string>> merged_to_original_categories;
//     
//     std::vector<BinInfo> bins;
//     size_t total_pos = 0;
//     size_t total_neg = 0;
//     
//     static constexpr double EPSILON = 1e-10;
//     std::vector<double> log_factorials;
//     
//     bool converged = false;
//     size_t iterations = 0;
//     
//     // Calculate Weight of Evidence (WOE) and Information Value (IV) for a bin
//     void calculateWoeIv(BinInfo& bin) const {
//         if (bin.count_pos == 0 || bin.count_neg == 0 || total_pos == 0 || total_neg == 0) {
//             bin.woe = 0.0;
//             bin.iv = 0.0;
//             return;
//         }
//         
//         double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//         double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//         
//         bin.woe = std::log((dist_pos + EPSILON) / (dist_neg + EPSILON));
//         bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//     
//     // Perform Fisher's Exact Test
//     double fisherExactTest(size_t a, size_t b, size_t c, size_t d) const {
//         size_t n = a + b + c + d;
//         size_t row1 = a + b;
//         size_t row2 = c + d;
//         size_t col1 = a + c;
//         size_t col2 = b + d;
//         
//         double log_p = log_factorials[row1] + log_factorials[row2] + log_factorials[col1] + log_factorials[col2]
//             - log_factorials[n] - log_factorials[a] - log_factorials[b] - log_factorials[c] - log_factorials[d];
//         
//         return std::exp(log_p);
//     }
//     
//     // Merge rare categories based on bin_cutoff
//     void mergeRareCategories() {
//         std::unordered_map<std::string, size_t> category_counts;
//         size_t total_count = feature.size();
//         
//         for (const auto& cat : feature) {
//             category_counts[cat]++;
//         }
//         
//         std::vector<std::pair<std::string, size_t>> sorted_categories(category_counts.begin(), category_counts.end());
//         std::sort(sorted_categories.begin(), sorted_categories.end(),
//                   [](const std::pair<std::string, size_t>& a, const std::pair<std::string, size_t>& b) {
//                       return a.second > b.second;
//                   });
//         
//         for (const auto& [cat, count] : sorted_categories) {
//             double frequency = static_cast<double>(count) / total_count;
//             if (frequency < bin_cutoff) {
//                 original_to_merged_category[cat] = sorted_categories.front().first;
//             } else {
//                 original_to_merged_category[cat] = cat;
//             }
//         }
//         
//         for (const auto& [original_cat, merged_cat] : original_to_merged_category) {
//             merged_to_original_categories[merged_cat].insert(original_cat);
//         }
//     }
//     
//     // Initialize bins
//     void initializeBins() {
//         std::unordered_map<std::string, size_t> bin_indices;
//         bins.clear();
//         
//         for (const auto& [merged_cat, original_cats] : merged_to_original_categories) {
//             bin_indices[merged_cat] = bins.size();
//             bins.emplace_back();
//             bins.back().categories = original_cats;
//         }
//         
//         BinWorker worker(feature, target, original_to_merged_category, bins, bin_indices);
//         parallelFor(0, feature.size(), worker, 1000, nthreads);
//         
//         for (auto& bin : bins) {
//             calculateWoeIv(bin);
//         }
//         
//         std::sort(bins.begin(), bins.end(), [](const BinInfo& a, const BinInfo& b) {
//             return a.woe < b.woe;
//         });
//     }
//     
//     // Merge bins to achieve monotonicity and respect min_bins
//     void mergeBins() {
//         while (bins.size() > min_bins) {
//             bool merged = false;
//             for (size_t i = 0; i < bins.size() - 1; ++i) {
//                 if (bins[i].woe > bins[i + 1].woe) {
//                     mergeTwoBins(i, i + 1);
//                     merged = true;
//                     break;
//                 }
//             }
//             if (!merged) break;
//         }
//     }
//     
//     // Merge two adjacent bins
//     void mergeTwoBins(size_t index1, size_t index2) {
//         BinInfo& bin1 = bins[index1];
//         BinInfo& bin2 = bins[index2];
//         
//         bin1.count += bin2.count;
//         bin1.count_pos += bin2.count_pos;
//         bin1.count_neg += bin2.count_neg;
//         bin1.categories.insert(bin2.categories.begin(), bin2.categories.end());
//         
//         calculateWoeIv(bin1);
//         
//         bins.erase(bins.begin() + index2);
//     }
//     
//     // Join strings with a delimiter
//     static std::string joinStrings(const std::unordered_set<std::string>& strings, const std::string& delimiter) {
//         std::ostringstream oss;
//         size_t i = 0;
//         for (const auto& s : strings) {
//             if (i > 0) oss << delimiter;
//             oss << s;
//             i++;
//         }
//         return oss.str();
//     }
//     
// public:
//     OptimalBinningCategoricalFETB(const std::vector<std::string>& feature,
//                                   const std::vector<int>& target,
//                                   size_t min_bins = 3,
//                                   size_t max_bins = 5,
//                                   double bin_cutoff = 0.05,
//                                   size_t max_n_prebins = 20,
//                                   double convergence_threshold = 1e-6,
//                                   size_t max_iterations = 1000,
//                                   int nthreads = 0)
//         : feature(feature), target(target),
//           min_bins(min_bins), max_bins(max_bins),
//           bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
//           convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//           nthreads(nthreads == 0 ? tbb::task_scheduler_init::automatic : nthreads) {
//         
//         validateInput();
//         
//         // Precalculate logarithms of factorials
//         size_t max_n = feature.size();
//         log_factorials.resize(max_n + 1);
//         log_factorials[0] = 0.0;
//         for (size_t i = 1; i <= max_n; ++i) {
//             log_factorials[i] = log_factorials[i - 1] + std::log(static_cast<double>(i));
//         }
//         
//         for (int val : target) {
//             if (val == 1) {
//                 total_pos++;
//             } else {
//                 total_neg++;
//             }
//         }
//     }
//     
//     // Validate input parameters
//     void validateInput() const {
//         if (feature.empty() || target.empty()) {
//             throw std::invalid_argument("Input vectors cannot be empty");
//         }
//         if (feature.size() != target.size()) {
//             throw std::invalid_argument("Feature and target vectors must have the same length");
//         }
//         if (min_bins < 2) {
//             throw std::invalid_argument("min_bins must be >= 2");
//         }
//         if (max_bins < min_bins) {
//             throw std::invalid_argument("max_bins must be >= min_bins");
//         }
//         if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
//             throw std::invalid_argument("bin_cutoff must be between 0 and 1");
//         }
//         if (max_n_prebins < min_bins) {
//             throw std::invalid_argument("max_n_prebins must be >= min_bins");
//         }
//         for (int val : target) {
//             if (val != 0 && val != 1) {
//                 throw std::invalid_argument("Target vector must contain only binary values (0 and 1)");
//             }
//         }
//     }
//     
//     // Perform optimal binning
//     void fit() {
//         mergeRareCategories();
//         initializeBins();
//         
//         size_t ncat = merged_to_original_categories.size();
//         max_bins = std::min(max_bins, ncat);
//         
//         if (ncat <= max_bins) {
//             converged = true;
//             return;
//         }
//         
//         double prev_total_iv = 0.0;
//         while (bins.size() > max_bins && iterations < max_iterations) {
//             double min_p_value = std::numeric_limits<double>::max();
//             size_t merge_index = 0;
//             
//             for (size_t i = 0; i < bins.size() - 1; ++i) {
//                 size_t a = bins[i].count_pos;
//                 size_t b = bins[i].count_neg;
//                 size_t c = bins[i + 1].count_pos;
//                 size_t d = bins[i + 1].count_neg;
//                 
//                 double p_value = fisherExactTest(a, b, c, d);
//                 
//                 if (p_value < min_p_value) {
//                     min_p_value = p_value;
//                     merge_index = i;
//                 }
//             }
//             
//             mergeTwoBins(merge_index, merge_index + 1);
//             
//             double total_iv = 0.0;
//             for (const auto& bin : bins) {
//                 total_iv += bin.iv;
//             }
//             
//             if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//                 converged = true;
//                 break;
//             }
//             
//             prev_total_iv = total_iv;
//             iterations++;
//         }
//         
//         mergeBins();
//     }
//     
//     // Get the results of optimal binning
//     Rcpp::List getResults() const {
//         std::vector<std::string> bin_labels;
//         std::vector<double> woe_values;
//         std::vector<double> iv_values;
//         std::vector<int> counts;
//         std::vector<int> counts_pos;
//         std::vector<int> counts_neg;
//         
//         bin_labels.reserve(bins.size());
//         woe_values.reserve(bins.size());
//         iv_values.reserve(bins.size());
//         counts.reserve(bins.size());
//         counts_pos.reserve(bins.size());
//         counts_neg.reserve(bins.size());
//         
//         for (const auto& bin : bins) {
//             std::string label = joinStrings(bin.categories, "%;%");
//             bin_labels.emplace_back(label);
//             woe_values.emplace_back(bin.woe);
//             iv_values.emplace_back(bin.iv);
//             counts.emplace_back(static_cast<int>(bin.count));
//             counts_pos.emplace_back(static_cast<int>(bin.count_pos));
//             counts_neg.emplace_back(static_cast<int>(bin.count_neg));
//         }
//         
// return Rcpp::List::create(
//             Rcpp::Named("bin") = bin_labels,
//             Rcpp::Named("woe") = woe_values,
//             Rcpp::Named("iv") = iv_values,
//             Rcpp::Named("count") = counts,
//             Rcpp::Named("count_pos") = counts_pos,
//             Rcpp::Named("count_neg") = counts_neg,
//             Rcpp::Named("converged") = converged,
//             Rcpp::Named("iterations") = static_cast<int>(iterations)
//         );
//     }
// };
// 
// //' Categorical Optimal Binning with Fisher's Exact Test
// //'
// //' Implements optimal binning for categorical variables using Fisher's Exact Test,
// //' calculating Weight of Evidence (WoE) and Information Value (IV).
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //' @param max_iterations Maximum number of iterations (default: 1000).
// //' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bin: Character vector of bin labels (merged categories).
// //'   \item woe: Numeric vector of Weight of Evidence values for each bin.
// //'   \item iv: Numeric vector of Information Value for each bin.
// //'   \item count: Integer vector of total count in each bin.
// //'   \item count_pos: Integer vector of positive class count in each bin.
// //'   \item count_neg: Integer vector of negative class count in each bin.
// //'   \item converged: Logical indicating whether the algorithm converged.
// //'   \item iterations: Integer indicating the number of iterations performed.
// //' }
// //'
// //' @details
// //' The algorithm uses Fisher's Exact Test to iteratively merge bins, maximizing
// //' the statistical significance of the difference between adjacent bins. It ensures
// //' monotonicity in the resulting bins and respects the minimum number of bins specified.
// //'
// //' @examples
// //' \dontrun{
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //' result <- optimal_binning_categorical_fetb(target, feature, min_bins = 2, max_bins = 4)
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_fetb(
//     Rcpp::IntegerVector target,
//     Rcpp::CharacterVector feature,
//     int min_bins = 3,
//     int max_bins = 5,
//     double bin_cutoff = 0.05,
//     int max_n_prebins = 20,
//     double convergence_threshold = 1e-6,
//     int max_iterations = 1000,
//     int nthreads = 0
// ) {
//     // Convert Rcpp vectors to std::vector
//     std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//     std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//     
//     try {
//         // Create and run the binner
//         OptimalBinningCategoricalFETB binner(
//             feature_vec, target_vec,
//             static_cast<size_t>(min_bins),
//             static_cast<size_t>(max_bins),
//             bin_cutoff,
//             static_cast<size_t>(max_n_prebins),
//             convergence_threshold,
//             static_cast<size_t>(max_iterations),
//             nthreads
//         );
//         
//         binner.fit();
//         
//         // Return the results
//         return binner.getResults();
//     } catch (const std::exception& e) {
//         Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
// }
