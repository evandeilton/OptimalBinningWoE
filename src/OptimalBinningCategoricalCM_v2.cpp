#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <memory>

using namespace Rcpp;

// Structure to hold bin information
struct Bin {
  std::vector<std::string> categories;
  int count_pos;
  int count_neg;
  double woe;
  double iv;
  int total_count;
  
  Bin() : count_pos(0), count_neg(0), woe(0), iv(0), total_count(0) {}
};

// Main class for optimal binning
class OptimalBinningCategorical {
private:
  // Input data and parameters
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  // Internal variables
  std::vector<std::shared_ptr<Bin>> bins;
  int total_pos;
  int total_neg;
  std::unordered_map<std::string, int> count_pos_map;
  std::unordered_map<std::string, int> count_neg_map;
  std::unordered_map<std::string, int> total_count_map;
  int unique_categories;
  bool is_increasing;
  bool converged;
  int iterations_run;
  
  // Constants
  static constexpr double EPSILON = 1e-10;
  
public:
  // Constructor
  OptimalBinningCategorical(
    const std::vector<std::string>& feature,
    const std::vector<int>& target,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins,
    std::string bin_separator,
    double convergence_threshold,
    int max_iterations
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
  convergence_threshold(convergence_threshold), max_iterations(max_iterations),
  total_pos(0), total_neg(0), unique_categories(0),
  is_increasing(true), converged(false), iterations_run(0) {}
  
  // Main function to perform binning
  Rcpp::List perform_binning() {
    try {
      validate_inputs();
      initialize_bins();
      handle_rare_categories();
      limit_prebins();
      ensure_min_bins();
      merge_bins();
      enforce_monotonicity();
      calculate_woe_iv_bins();
      return prepare_output();
    } catch (const std::exception& e) {
      Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
    }
  }
  
private:
  // Validate input parameters
  void validate_inputs() {
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target vectors must not be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
      throw std::invalid_argument("Invalid min_bins or max_bins values.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be positive.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be positive.");
    }
    
    // Count total positives and negatives
    for (size_t i = 0; i < target.size(); ++i) {
      int t = target[i];
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target vector must be binary (0 or 1).");
      }
      if (t == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
    
    if (total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target vector must contain both 0 and 1 values.");
    }
    
    // Count unique categories
    std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
    unique_categories = unique_set.size();
    
    // Adjust min_bins and max_bins based on unique categories
    min_bins = std::max(2, std::min(min_bins, unique_categories));
    max_bins = std::min(max_bins, unique_categories);
    if (min_bins > max_bins) {
      min_bins = max_bins;
    }
  }
  
  // Initialize bins with each category as its own bin
  void initialize_bins() {
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int tar = target[i];
      
      total_count_map[cat]++;
      if (tar == 1) {
        count_pos_map[cat]++;
      } else {
        count_neg_map[cat]++;
      }
    }
    
    // Create bins
    for (const auto& item : total_count_map) {
      auto bin = std::make_shared<Bin>();
      bin->categories.push_back(item.first);
      bin->count_pos = count_pos_map[item.first];
      bin->count_neg = count_neg_map[item.first];
      bin->total_count = item.second;
      bins.push_back(bin);
    }
  }
  
  // Handle rare categories by merging them into existing bins
  void handle_rare_categories() {
    int total_count = total_pos + total_neg;
    std::vector<std::shared_ptr<Bin>> updated_bins;
    std::vector<std::shared_ptr<Bin>> rare_bins;
    
    // Separate rare and non-rare bins
    for (auto& bin : bins) {
      double freq = static_cast<double>(bin->total_count) / total_count;
      if (freq < bin_cutoff && unique_categories > 2) {
        rare_bins.push_back(bin);
      } else {
        updated_bins.push_back(bin);
      }
    }
    
    // Merge each rare bin into the most similar existing bin
    for (auto& rare_bin : rare_bins) {
      size_t best_merge_index = 0;
      double min_chi_square = std::numeric_limits<double>::max();
      
      for (size_t i = 0; i < updated_bins.size(); ++i) {
        double chi_sq = compute_chi_square_between_bins(*rare_bin, *updated_bins[i]);
        if (chi_sq < min_chi_square) {
          min_chi_square = chi_sq;
          best_merge_index = i;
        }
      }
      
      merge_two_bins(updated_bins[best_merge_index], rare_bin);
    }
    
    bins = updated_bins;
  }
  
  // Limit the number of prebins to max_n_prebins
  void limit_prebins() {
    while (bins.size() > static_cast<size_t>(max_n_prebins) && can_merge_further()) {
      size_t merge_index = 0;
      int min_total = bins[0]->total_count + ((bins.size() > 1) ? bins[1]->total_count : 0);
      
      for (size_t i = 1; i < bins.size() - 1; ++i) {
        int current_total = bins[i]->total_count + bins[i + 1]->total_count;
        if (current_total < min_total) {
          min_total = current_total;
          merge_index = i;
        }
      }
      
      merge_adjacent_bins(merge_index);
    }
  }
  
  // Ensure min_bins is respected
  void ensure_min_bins() {
    while (bins.size() < static_cast<size_t>(min_bins)) {
      auto max_it = std::max_element(bins.begin(), bins.end(),
                                     [](const std::shared_ptr<Bin>& a, const std::shared_ptr<Bin>& b) {
                                       return a->total_count < b->total_count;
                                     });
      
      if ((*max_it)->categories.size() <= 1) {
        break;
      }
      
      auto bin1 = std::make_shared<Bin>();
      auto bin2 = std::make_shared<Bin>();
      size_t split_point = (*max_it)->categories.size() / 2;
      
      bin1->categories.insert(bin1->categories.end(),
                              (*max_it)->categories.begin(),
                              (*max_it)->categories.begin() + split_point);
      bin2->categories.insert(bin2->categories.end(),
                              (*max_it)->categories.begin() + split_point,
                              (*max_it)->categories.end());
      
      for (const auto& cat : bin1->categories) {
        bin1->count_pos += count_pos_map.at(cat);
        bin1->count_neg += count_neg_map.at(cat);
      }
      bin1->total_count = bin1->count_pos + bin1->count_neg;
      
      for (const auto& cat : bin2->categories) {
        bin2->count_pos += count_pos_map.at(cat);
        bin2->count_neg += count_neg_map.at(cat);
      }
      bin2->total_count = bin2->count_pos + bin2->count_neg;
      
      *max_it = bin1;
      bins.push_back(bin2);
    }
  }
  
  // Merge bins based on Chi-square statistics until max_bins is reached
  void merge_bins() {
    iterations_run = 0;
    bool keep_merging = true;
    while (can_merge_further() && keep_merging && iterations_run < max_iterations) {
      std::vector<double> chi_squares(bins.size() - 1, 0.0);
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        chi_squares[i] = compute_chi_square_between_bins(*bins[i], *bins[i + 1]);
      }
      
      auto min_it = std::min_element(chi_squares.begin(), chi_squares.end());
      size_t min_index = std::distance(chi_squares.begin(), min_it);
      
      double old_chi_square = *min_it;
      merge_adjacent_bins(min_index);
      
      if (!can_merge_further()) {
        break;
      }
      
      double new_chi_square = compute_chi_square_between_bins(*bins[min_index],
                                                              (min_index + 1 < bins.size()) ? *bins[min_index + 1] : *bins[min_index]);
      
      keep_merging = std::abs(new_chi_square - old_chi_square) > convergence_threshold;
      iterations_run++;
    }
    
    converged = (iterations_run < max_iterations) || !can_merge_further();
  }
  
  // Enforce monotonicity of WoE
  void enforce_monotonicity() {
    if (unique_categories <= 2) {
      return;
    }
    
    determine_monotonicity_direction();
    
    bool monotonic = false;
    while (!monotonic && can_merge_further()) {
      monotonic = true;
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        bool violation = is_increasing ? (bins[i]->woe > bins[i + 1]->woe) : (bins[i]->woe < bins[i + 1]->woe);
        if (violation) {
          merge_adjacent_bins(i);
          monotonic = false;
          break;
        }
      }
    }
  }
  
  // Calculate WoE and IV for bins
  void calculate_woe_iv_bins() {
    for (size_t i = 0; i < bins.size(); ++i) {
      auto& bin = bins[i];
      if (bin->count_pos == 0 || bin->count_neg == 0) {
        bin->iv = 0.0;
      } else {
        double dist_pos = static_cast<double>(bin->count_pos) / total_pos;
        double dist_neg = static_cast<double>(bin->count_neg) / total_neg;
        dist_pos = std::max(dist_pos, EPSILON);
        dist_neg = std::max(dist_neg, EPSILON);
        bin->woe = std::log(dist_pos / dist_neg);
        bin->iv = (dist_pos - dist_neg) * bin->woe;
      }
    }
  }
  
  // Prepare the output List
  Rcpp::List prepare_output() {
    std::vector<std::string> bin_names;
    std::vector<double> bin_woe;
    std::vector<double> bin_iv;
    std::vector<int> bin_count;
    std::vector<int> bin_count_pos;
    std::vector<int> bin_count_neg;
    
    for (const auto& bin : bins) {
      bin_names.push_back(join_categories(bin->categories));
      bin_woe.push_back(bin->woe);
      bin_iv.push_back(bin->iv);
      bin_count.push_back(bin->total_count);
      bin_count_pos.push_back(bin->count_pos);
      bin_count_neg.push_back(bin->count_neg);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = bin_woe,
      Rcpp::Named("iv") = bin_iv,
      Rcpp::Named("count") = bin_count,
      Rcpp::Named("count_pos") = bin_count_pos,
      Rcpp::Named("count_neg") = bin_count_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
  
  // Compute Chi-square between two bins
  double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
    int total = bin1.total_count + bin2.total_count;
    int total_pos_bins = bin1.count_pos + bin2.count_pos;
    int total_neg_bins = bin1.count_neg + bin2.count_neg;
    
    double expected1_pos = static_cast<double>(bin1.total_count) * total_pos_bins / total;
    double expected1_neg = static_cast<double>(bin1.total_count) * total_neg_bins / total;
    double expected2_pos = static_cast<double>(bin2.total_count) * total_pos_bins / total;
    double expected2_neg = static_cast<double>(bin2.total_count) * total_neg_bins / total;
    
    expected1_pos = std::max(expected1_pos, EPSILON);
    expected1_neg = std::max(expected1_neg, EPSILON);
    expected2_pos = std::max(expected2_pos, EPSILON);
    expected2_neg = std::max(expected2_neg, EPSILON);
    
    double chi_square = 0.0;
    chi_square += std::pow(bin1.count_pos - expected1_pos, 2) / expected1_pos;
    chi_square += std::pow(bin1.count_neg - expected1_neg, 2) / expected1_neg;
    chi_square += std::pow(bin2.count_pos - expected2_pos, 2) / expected2_pos;
    chi_square += std::pow(bin2.count_neg - expected2_neg, 2) / expected2_neg;
    
    return chi_square;
  }
  
  // Merge two adjacent bins at the specified index
  void merge_adjacent_bins(size_t index) {
    if (index >= bins.size() - 1) return;
    merge_two_bins(bins[index], bins[index + 1]);
    bins.erase(bins.begin() + index + 1);
  }
  
  // Merge two bins
  void merge_two_bins(std::shared_ptr<Bin>& bin1, std::shared_ptr<Bin>& bin2) {
    bin1->categories.insert(bin1->categories.end(),
                            bin2->categories.begin(),
                            bin2->categories.end());
    bin1->count_pos += bin2->count_pos;
    bin1->count_neg += bin2->count_neg;
    bin1->total_count += bin2->total_count;
  }
  
  // Concatenate category names with the specified separator
  std::string join_categories(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    
    std::string result = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      result += bin_separator + categories[i];
    }
    return result;
  }
  
  // Determine the direction of monotonicity
  void determine_monotonicity_direction() {
    if (bins.size() < 2) {
      is_increasing = true;
      return;
    }
    
    double first_woe = bins.front()->woe;
    double last_woe = bins.back()->woe;
    
    is_increasing = (last_woe >= first_woe);
    
    if (std::abs(last_woe - first_woe) < EPSILON) {
      is_increasing = true;
    }
  }
  
  // Check if further merging is allowed
  bool can_merge_further() const {
    return bins.size() > static_cast<size_t>(std::max(min_bins, 2));
  }
};

//' @title Optimal Binning for Categorical Variables
//'
//' @description
//' Implements optimal binning for categorical variables using the Chi-Merge algorithm,
//' calculating Weight of Evidence (WoE) and Information Value (IV) for resulting bins.
//'
//' @param target Integer vector of binary target values (0 or 1).
//' @param feature Character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
//' @param convergence_threshold Threshold for convergence in Chi-square difference (default: 1e-6).
//' @param max_iterations Maximum number of iterations for bin merging (default: 1000).
//'
//' @return A list containing:
//' \itemize{
//'   \item bins: Vector of bin names (concatenated categories).
//'   \item woe: Vector of Weight of Evidence values for each bin.
//'   \item iv: Vector of Information Value for each bin.
//'   \item count: Vector of total counts for each bin.
//'   \item count_pos: Vector of positive class counts for each bin.
//'   \item count_neg: Vector of negative class counts for each bin.
//'   \item converged: Boolean indicating whether the algorithm converged.
//'   \item iterations: Number of iterations run.
//' }
//'
//' @details
//' The algorithm uses Chi-square statistics to merge adjacent bins:
//'
//' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
//'
//' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency
//' for bin i and class j.
//'
//' Weight of Evidence (WoE) for each bin:
//'
//' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
//'
//' Information Value (IV) for each bin:
//'
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//'
//' The algorithm initializes bins for each category, merges rare categories based on
//' bin_cutoff, and then iteratively merges bins with the lowest chi-square
//' until max_bins is reached or no further merging is possible. It determines the
//' direction of monotonicity based on the initial trend and enforces it, allowing
//' deviations if min_bins constraints are triggered.
//'
//' @examples
//' \dontrun{
//' # Example data
//' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
//' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_cm(target, feature, min_bins = 2, max_bins = 4)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_cm(
   Rcpp::IntegerVector target,
   Rcpp::CharacterVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Convert R vectors to C++ vectors
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create OptimalBinningCategorical object and perform binning
 OptimalBinningCategorical obcm(
     feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations
 );
 
 return obcm.perform_binning();
}





// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <unordered_map>
// #include <unordered_set>
// #include <memory>
// 
// using namespace Rcpp;
// 
// // Structure to hold bin information
// struct Bin {
//   std::vector<std::string> categories;
//   int count_pos;
//   int count_neg;
//   double woe;
//   double iv;
//   int total_count;
// 
//   Bin() : count_pos(0), count_neg(0), woe(0), iv(0), total_count(0) {}
// };
// 
// // Main class for optimal binning
// class OptimalBinningCategorical {
// private:
//   // Input data and parameters
//   const std::vector<std::string>& feature;
//   const std::vector<int>& target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   int nthreads;
// 
//   // Internal variables
//   std::vector<std::shared_ptr<Bin>> bins;
//   int total_pos;
//   int total_neg;
//   std::unordered_map<std::string, int> count_pos_map;
//   std::unordered_map<std::string, int> count_neg_map;
//   std::unordered_map<std::string, int> total_count_map;
//   int unique_categories;
//   bool is_increasing;
//   bool converged;
//   int iterations_run;
// 
//   // Constants
//   static constexpr double EPSILON = 1e-10;
// 
// public:
//   // Constructor
//   OptimalBinningCategorical(
//     const std::vector<std::string>& feature,
//     const std::vector<int>& target,
//     int min_bins,
//     int max_bins,
//     double bin_cutoff,
//     int max_n_prebins,
//     std::string bin_separator,
//     double convergence_threshold,
//     int max_iterations,
//     int nthreads
//   ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//   bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
//   convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//   nthreads(nthreads), total_pos(0), total_neg(0), unique_categories(0),
//   is_increasing(true), converged(false), iterations_run(0) {}
// 
//   // Main function to perform binning
//   Rcpp::List perform_binning() {
//     try {
//       validate_inputs();
//       initialize_bins();
//       handle_rare_categories();
//       limit_prebins();
//       ensure_min_bins();
//       merge_bins();
//       enforce_monotonicity();
//       calculate_woe_iv_bins();
//       return prepare_output();
//     } catch (const std::exception& e) {
//       Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//     }
//   }
// 
// private:
//   // Validate input parameters
//   void validate_inputs() {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature and target vectors must not be empty.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must be of the same length.");
//     }
//     if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
//       throw std::invalid_argument("Invalid min_bins or max_bins values.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold must be positive.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations must be positive.");
//     }
// 
//     // Count total positives and negatives
//     for (size_t i = 0; i < target.size(); ++i) {
//       int t = target[i];
//       if (t != 0 && t != 1) {
//         throw std::invalid_argument("Target vector must be binary (0 or 1).");
//       }
//       if (t == 1) {
//         total_pos++;
//       } else {
//         total_neg++;
//       }
//     }
// 
//     if (total_pos == 0 || total_neg == 0) {
//       throw std::invalid_argument("Target vector must contain both 0 and 1 values.");
//     }
// 
//     // Count unique categories
//     std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
//     unique_categories = unique_set.size();
// 
//     // Adjust min_bins and max_bins based on unique categories
//     min_bins = std::max(2, std::min(min_bins, unique_categories));
//     max_bins = std::min(max_bins, unique_categories);
//     if (min_bins > max_bins) {
//       min_bins = max_bins;
//     }
//   }
// 
//   // Initialize bins with each category as its own bin
//   void initialize_bins() {
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int tar = target[i];
// 
//       total_count_map[cat]++;
//       if (tar == 1) {
//         count_pos_map[cat]++;
//       } else {
//         count_neg_map[cat]++;
//       }
//     }
// 
//     // Create bins
//     for (const auto& item : total_count_map) {
//       auto bin = std::make_shared<Bin>();
//       bin->categories.push_back(item.first);
//       bin->count_pos = count_pos_map[item.first];
//       bin->count_neg = count_neg_map[item.first];
//       bin->total_count = item.second;
//       bins.push_back(bin);
//     }
//   }
// 
//   // Handle rare categories by merging them into existing bins
//   void handle_rare_categories() {
//     int total_count = total_pos + total_neg;
//     std::vector<std::shared_ptr<Bin>> updated_bins;
//     std::vector<std::shared_ptr<Bin>> rare_bins;
// 
//     // Separate rare and non-rare bins
//     for (auto& bin : bins) {
//       double freq = static_cast<double>(bin->total_count) / total_count;
//       if (freq < bin_cutoff && unique_categories > 2) {
//         rare_bins.push_back(bin);
//       } else {
//         updated_bins.push_back(bin);
//       }
//     }
// 
//     // Merge each rare bin into the most similar existing bin
//     for (auto& rare_bin : rare_bins) {
//       size_t best_merge_index = 0;
//       double min_chi_square = std::numeric_limits<double>::max();
// 
//       for (size_t i = 0; i < updated_bins.size(); ++i) {
//         double chi_sq = compute_chi_square_between_bins(*rare_bin, *updated_bins[i]);
//         if (chi_sq < min_chi_square) {
//           min_chi_square = chi_sq;
//           best_merge_index = i;
//         }
//       }
// 
//       merge_two_bins(updated_bins[best_merge_index], rare_bin);
//     }
// 
//     bins = updated_bins;
//   }
// 
//   // Limit the number of prebins to max_n_prebins
//   void limit_prebins() {
//     while (bins.size() > static_cast<size_t>(max_n_prebins) && can_merge_further()) {
//       size_t merge_index = 0;
//       int min_total = bins[0]->total_count + ((bins.size() > 1) ? bins[1]->total_count : 0);
// 
//       for (size_t i = 1; i < bins.size() - 1; ++i) {
//         int current_total = bins[i]->total_count + bins[i + 1]->total_count;
//         if (current_total < min_total) {
//           min_total = current_total;
//           merge_index = i;
//         }
//       }
// 
//       merge_adjacent_bins(merge_index);
//     }
//   }
// 
//   // Ensure min_bins is respected
//   void ensure_min_bins() {
//     while (bins.size() < static_cast<size_t>(min_bins)) {
//       auto max_it = std::max_element(bins.begin(), bins.end(),
//                                      [](const std::shared_ptr<Bin>& a, const std::shared_ptr<Bin>& b) {
//                                        return a->total_count < b->total_count;
//                                      });
// 
//       if ((*max_it)->categories.size() <= 1) {
//         break;
//       }
// 
//       auto bin1 = std::make_shared<Bin>();
//       auto bin2 = std::make_shared<Bin>();
//       size_t split_point = (*max_it)->categories.size() / 2;
// 
//       bin1->categories.insert(bin1->categories.end(),
//                               (*max_it)->categories.begin(),
//                               (*max_it)->categories.begin() + split_point);
//       bin2->categories.insert(bin2->categories.end(),
//                               (*max_it)->categories.begin() + split_point,
//                               (*max_it)->categories.end());
// 
//       for (const auto& cat : bin1->categories) {
//         bin1->count_pos += count_pos_map.at(cat);
//         bin1->count_neg += count_neg_map.at(cat);
//       }
//       bin1->total_count = bin1->count_pos + bin1->count_neg;
// 
//       for (const auto& cat : bin2->categories) {
//         bin2->count_pos += count_pos_map.at(cat);
//         bin2->count_neg += count_neg_map.at(cat);
//       }
//       bin2->total_count = bin2->count_pos + bin2->count_neg;
// 
//       *max_it = bin1;
//       bins.push_back(bin2);
//     }
//   }
// 
//   // Merge bins based on Chi-square statistics until max_bins is reached
//   void merge_bins() {
//     iterations_run = 0;
//     bool keep_merging = true;
//     while (can_merge_further() && keep_merging && iterations_run < max_iterations) {
//       std::vector<double> chi_squares(bins.size() - 1, 0.0);
// 
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         chi_squares[i] = compute_chi_square_between_bins(*bins[i], *bins[i + 1]);
//       }
// 
//       auto min_it = std::min_element(chi_squares.begin(), chi_squares.end());
//       size_t min_index = std::distance(chi_squares.begin(), min_it);
// 
//       double old_chi_square = *min_it;
//       merge_adjacent_bins(min_index);
// 
//       if (!can_merge_further()) {
//         break;
//       }
// 
//       double new_chi_square = compute_chi_square_between_bins(*bins[min_index],
//                                                               (min_index + 1 < bins.size()) ? *bins[min_index + 1] : *bins[min_index]);
// 
//       keep_merging = std::abs(new_chi_square - old_chi_square) > convergence_threshold;
//       iterations_run++;
//     }
// 
//     converged = (iterations_run < max_iterations) || !can_merge_further();
//   }
// 
//   // Enforce monotonicity of WoE
//   void enforce_monotonicity() {
//     if (unique_categories <= 2) {
//       return;
//     }
// 
//     determine_monotonicity_direction();
// 
//     bool monotonic = false;
//     while (!monotonic && can_merge_further()) {
//       monotonic = true;
//       for (size_t i = 0; i < bins.size() - 1; ++i) {
//         bool violation = is_increasing ? (bins[i]->woe > bins[i + 1]->woe) : (bins[i]->woe < bins[i + 1]->woe);
//         if (violation) {
//           merge_adjacent_bins(i);
//           monotonic = false;
//           break;
//         }
//       }
//     }
//   }
// 
//   // Calculate WoE and IV for bins
//   void calculate_woe_iv_bins() {
//     for (size_t i = 0; i < bins.size(); ++i) {
//       auto& bin = bins[i];
//       if (bin->count_pos == 0 || bin->count_neg == 0) {
//         bin->iv = 0.0;
//       } else {
//         double dist_pos = static_cast<double>(bin->count_pos) / total_pos;
//         double dist_neg = static_cast<double>(bin->count_neg) / total_neg;
//         dist_pos = std::max(dist_pos, EPSILON);
//         dist_neg = std::max(dist_neg, EPSILON);
//         bin->woe = std::log(dist_pos / dist_neg);
//         bin->iv = (dist_pos - dist_neg) * bin->woe;
//       }
//     }
//   }
// 
//   // Prepare the output List
//   Rcpp::List prepare_output() {
//     std::vector<std::string> bin_names;
//     std::vector<double> bin_woe;
//     std::vector<double> bin_iv;
//     std::vector<int> bin_count;
//     std::vector<int> bin_count_pos;
//     std::vector<int> bin_count_neg;
// 
//     for (const auto& bin : bins) {
//       bin_names.push_back(join_categories(bin->categories));
//       bin_woe.push_back(bin->woe);
//       bin_iv.push_back(bin->iv);
//       bin_count.push_back(bin->total_count);
//       bin_count_pos.push_back(bin->count_pos);
//       bin_count_neg.push_back(bin->count_neg);
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("bins") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// 
//   // Compute Chi-square between two bins
//   double compute_chi_square_between_bins(const Bin& bin1, const Bin& bin2) const {
//     int total = bin1.total_count + bin2.total_count;
//     int total_pos_bins = bin1.count_pos + bin2.count_pos;
//     int total_neg_bins = bin1.count_neg + bin2.count_neg;
// 
//     double expected1_pos = static_cast<double>(bin1.total_count) * total_pos_bins / total;
//     double expected1_neg = static_cast<double>(bin1.total_count) * total_neg_bins / total;
//     double expected2_pos = static_cast<double>(bin2.total_count) * total_pos_bins / total;
//     double expected2_neg = static_cast<double>(bin2.total_count) * total_neg_bins / total;
// 
//     expected1_pos = std::max(expected1_pos, EPSILON);
//     expected1_neg = std::max(expected1_neg, EPSILON);
//     expected2_pos = std::max(expected2_pos, EPSILON);
//     expected2_neg = std::max(expected2_neg, EPSILON);
// 
//     double chi_square = 0.0;
//     chi_square += std::pow(bin1.count_pos - expected1_pos, 2) / expected1_pos;
//     chi_square += std::pow(bin1.count_neg - expected1_neg, 2) / expected1_neg;
//     chi_square += std::pow(bin2.count_pos - expected2_pos, 2) / expected2_pos;
//     chi_square += std::pow(bin2.count_neg - expected2_neg, 2) / expected2_neg;
// 
//     return chi_square;
//   }
// 
//   // Merge two adjacent bins at the specified index
//   void merge_adjacent_bins(size_t index) {
//     if (index >= bins.size() - 1) return;
//     merge_two_bins(bins[index], bins[index + 1]);
//     bins.erase(bins.begin() + index + 1);
//   }
// 
//   // Merge two bins
//   void merge_two_bins(std::shared_ptr<Bin>& bin1, std::shared_ptr<Bin>& bin2) {
//     bin1->categories.insert(bin1->categories.end(),
//                             bin2->categories.begin(),
//                             bin2->categories.end());
//     bin1->count_pos += bin2->count_pos;
//     bin1->count_neg += bin2->count_neg;
//     bin1->total_count += bin2->total_count;
//   }
// 
//   // Concatenate category names with the specified separator
//   std::string join_categories(const std::vector<std::string>& categories) const {
//     if (categories.empty()) return "";
// 
//     std::string result = categories[0];
//     for (size_t i = 1; i < categories.size(); ++i) {
//       result += bin_separator + categories[i];
//     }
//     return result;
//   }
// 
//   // Determine the direction of monotonicity
//   void determine_monotonicity_direction() {
//     if (bins.size() < 2) {
//       is_increasing = true;
//       return;
//     }
// 
//     double first_woe = bins.front()->woe;
//     double last_woe = bins.back()->woe;
// 
//     is_increasing = (last_woe >= first_woe);
// 
//     if (std::abs(last_woe - first_woe) < EPSILON) {
//       is_increasing = true;
//     }
//   }
// 
//   // Check if further merging is allowed
//   bool can_merge_further() const {
//     return bins.size() > static_cast<size_t>(std::max(min_bins, 2));
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables
// //'
// //' @description
// //' Implements optimal binning for categorical variables using the Chi-Merge algorithm,
// //' calculating Weight of Evidence (WoE) and Information Value (IV) for resulting bins.
// //'
// //' @param target Integer vector of binary target values (0 or 1).
// //' @param feature Character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator for concatenating category names in bins (default: "%;%").
// //' @param convergence_threshold Threshold for convergence in Chi-square difference (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for bin merging (default: 1000).
// //'
// //' @return A list containing:
// //' \itemize{
// //'   \item bins: Vector of bin names (concatenated categories).
// //'   \item woe: Vector of Weight of Evidence values for each bin.
// //'   \item iv: Vector of Information Value for each bin.
// //'   \item count: Vector of total counts for each bin.
// //'   \item count_pos: Vector of positive class counts for each bin.
// //'   \item count_neg: Vector of negative class counts for each bin.
// //'   \item converged: Boolean indicating whether the algorithm converged.
// //'   \item iterations: Number of iterations run.
// //' }
// //'
// //' @details
// //' The algorithm uses Chi-square statistics to merge adjacent bins:
// //'
// //' \deqn{\chi^2 = \sum_{i=1}^{2}\sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
// //'
// //' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency
// //' for bin i and class j.
// //'
// //' Weight of Evidence (WoE) for each bin:
// //'
// //' \deqn{WoE = \ln(\frac{P(X|Y=1)}{P(X|Y=0)})}
// //'
// //' Information Value (IV) for each bin:
// //'
// //' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
// //'
// //' The algorithm initializes bins for each category, merges rare categories based on
// //' bin_cutoff, and then iteratively merges bins with the lowest chi-square
// //' until max_bins is reached or no further merging is possible. It determines the
// //' direction of monotonicity based on the initial trend and enforces it, allowing
// //' deviations if min_bins constraints are triggered.
// //'
// //' @examples
// //' \dontrun{
// //' # Example data
// //' target <- c(1, 0, 1, 1, 0, 1, 0, 0, 1, 1)
// //' feature <- c("A", "B", "A", "C", "B", "D", "C", "A", "D", "B")
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_cm(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_cm(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000,
//    int nthreads = 0
// ) {
//  // Convert R vectors to C++ vectors
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//  // Note: nthreads is retained for compatibility but not used in serial version
//  (void)nthreads; // Suppress unused variable warning
// 
//  // Create OptimalBinningCategorical object and perform binning
//  OptimalBinningCategorical obcm(
//      feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations, nthreads
//  );
// 
//  return obcm.perform_binning();
// }
