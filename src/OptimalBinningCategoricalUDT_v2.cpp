#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

class OptimalBinningCategoricalUDT {
private:
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  struct BinInfo {
    std::string name;
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
  };
  
  std::vector<BinInfo> bins_;
  bool converged_;
  int iterations_;
  
  void validate_inputs(const std::vector<std::string>& feature, const std::vector<int>& target) {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    for (int t : target) {
      if (t != 0 && t != 1) {
        throw std::invalid_argument("Target vector must contain only 0 and 1.");
      }
    }
  }
  
  void initial_binning(const std::vector<std::string>& feature, const std::vector<int>& target) {
    std::unordered_map<std::string, BinInfo> bin_map;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      auto& bin = bin_map[feature[i]];
      bin.name = feature[i];
      bin.count++;
      bin.count_pos += target[i];
      bin.count_neg += 1 - target[i];
    }
    
    bins_.clear();
    for (const auto& pair : bin_map) {
      bins_.push_back(pair.second);
    }
  }
  
  void merge_low_frequency_bins() {
    std::sort(bins_.begin(), bins_.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.count < b.count;
    });
    
    int total_count = 0;
    for (const auto& bin : bins_) {
      total_count += bin.count;
    }
    
    double cutoff_count = total_count * bin_cutoff_;
    std::vector<BinInfo> new_bins;
    BinInfo others{"Others", 0, 0, 0, 0, 0};
    
    for (const auto& bin : bins_) {
      if (bin.count >= cutoff_count || new_bins.size() < static_cast<size_t>(min_bins_)) {
        new_bins.push_back(bin);
      } else {
        others.count += bin.count;
        others.count_pos += bin.count_pos;
        others.count_neg += bin.count_neg;
      }
    }
    
    if (others.count > 0) {
      new_bins.push_back(others);
    }
    
    bins_ = new_bins;
  }
  
  void calculate_woe_iv() {
    int total_pos = 0, total_neg = 0;
    for (const auto& bin : bins_) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    double total_iv = 0;
    for (auto& bin : bins_) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      
      bin.woe = std::log((pos_rate + 1e-10) / (neg_rate + 1e-10));
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }
  
  void merge_bins() {
    while (bins_.size() > static_cast<size_t>(max_bins_)) {
      auto min_iv_it = std::min_element(bins_.begin(), bins_.end(),
                                        [](const BinInfo& a, const BinInfo& b) { return a.iv < b.iv; });
      
      if (min_iv_it == bins_.end()) break;
      
      auto next_it = std::next(min_iv_it);
      if (next_it == bins_.end()) next_it = std::prev(min_iv_it);
      
      min_iv_it->count += next_it->count;
      min_iv_it->count_pos += next_it->count_pos;
      min_iv_it->count_neg += next_it->count_neg;
      min_iv_it->name += bin_separator_ + next_it->name;
      
      bins_.erase(next_it);
    }
  }
  
  void split_bins() {
    while (bins_.size() < static_cast<size_t>(min_bins_)) {
      auto max_iv_it = std::max_element(bins_.begin(), bins_.end(),
                                        [](const BinInfo& a, const BinInfo& b) { return a.iv < b.iv; });
      
      if (max_iv_it == bins_.end()) break;
      
      BinInfo new_bin = *max_iv_it;
      max_iv_it->count /= 2;
      max_iv_it->count_pos /= 2;
      max_iv_it->count_neg /= 2;
      new_bin.count -= max_iv_it->count;
      new_bin.count_pos -= max_iv_it->count_pos;
      new_bin.count_neg -= max_iv_it->count_neg;
      new_bin.name += "_split";
      
      bins_.push_back(new_bin);
    }
  }
  
  void ensure_monotonicity() {
    std::sort(bins_.begin(), bins_.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
  }
  
  double calculate_total_iv() {
    double total_iv = 0;
    for (const auto& bin : bins_) {
      total_iv += bin.iv;
    }
    return total_iv;
  }
  
public:
  OptimalBinningCategoricalUDT(
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
  max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
  convergence_threshold_(convergence_threshold), max_iterations_(max_iterations),
  converged_(false), iterations_(0) {}
  
  void fit(const std::vector<std::string>& feature, const std::vector<int>& target) {
    validate_inputs(feature, target);
    initial_binning(feature, target);
    merge_low_frequency_bins();
    
    double prev_total_iv = 0;
    converged_ = false;
    iterations_ = 0;
    
    while (!converged_ && iterations_ < max_iterations_) {
      calculate_woe_iv();
      merge_bins();
      split_bins();
      ensure_monotonicity();
      
      double total_iv = calculate_total_iv();
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold_) {
        converged_ = true;
      }
      
      prev_total_iv = total_iv;
      iterations_++;
    }
  }
  
  Rcpp::List get_woe_bin() const {
    Rcpp::CharacterVector bin_names;
    Rcpp::NumericVector woe_values, iv_values;
    Rcpp::IntegerVector counts, counts_pos, counts_neg;
    
    for (const auto& bin : bins_) {
      bin_names.push_back(bin.name);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bins") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged_,
      Rcpp::Named("iterations") = iterations_
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_udt(
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
  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  OptimalBinningCategoricalUDT binning(
      min_bins, max_bins, bin_cutoff, max_n_prebins,
      bin_separator, convergence_threshold, max_iterations
  );
  
  try {
    binning.fit(feature_vec, target_vec);
    return binning.get_woe_bin();
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}








// #include <Rcpp.h>
// #include <vector>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <numeric>
// 
// // For convenience
// using namespace Rcpp;
// 
// // Define a structure to hold bin information
// struct BinInfo {
//   std::vector<std::string> categories;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
//   
//   // Constructor to initialize counts
//   BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
// };
// 
// // OptimalBinningCategoricalUDT Class
// class OptimalBinningCategoricalUDT {
// private:
//   // Parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Data
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   std::vector<BinInfo> final_bins;
//   bool converged;
//   int iterations_run;
//   
//   // Helper Functions
//   void validate_input_parameters(const size_t unique_categories) {
//     min_bins = std::max(2, std::min(static_cast<int>(unique_categories), min_bins));
//     max_bins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_bins));
//     bin_cutoff = std::max(0.0, std::min(1.0, bin_cutoff));
//     max_n_prebins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_n_prebins));
//   }
//   
//   void validate_input_data() const {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature and target vectors must not be empty.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same length.");
//     }
//     std::unordered_set<int> unique_targets(target.begin(), target.end());
//     if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
//       throw std::invalid_argument("Target variable must be binary (0 and 1).");
//     }
//   }
//   
//   double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     const double epsilon = 1e-10;
//     double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
//     double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
//     return std::log(event_rate / non_event_rate);
//   }
//   
//   double calculate_iv(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     const double epsilon = 1e-10;
//     double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
//     double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
//     return (event_rate - non_event_rate) * std::log(event_rate / non_event_rate);
//   }
//   
//   void compute_initial_bins(std::vector<BinInfo>& initial_bins, int& total_pos, int& total_neg) {
//     std::unordered_map<std::string, BinInfo> temp_bins;
//     total_pos = 0;
//     total_neg = 0;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int tgt = target[i];
//       auto& bin = temp_bins[cat];
//       if (bin.categories.empty()) {
//         bin.categories.push_back(cat);
//       }
//       bin.count++;
//       bin.count_pos += tgt;
//       bin.count_neg += (1 - tgt);
//       
//       total_pos += tgt;
//       total_neg += (1 - tgt);
//     }
//     
//     initial_bins.clear();
//     for (auto& pair : temp_bins) {
//       initial_bins.push_back(std::move(pair.second));
//     }
//   }
//   
//   void calculate_woe_iv(std::vector<BinInfo>& bins, int total_pos, int total_neg) {
//     for (auto& bin : bins) {
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
//     }
//   }
//   
//   bool is_monotonic(const std::vector<BinInfo>& bins) const {
//     if (bins.size() <= 2) return true;
//     bool increasing = bins[1].woe > bins[0].woe;
//     for (size_t i = 2; i < bins.size(); ++i) {
//       if ((increasing && bins[i].woe < bins[i-1].woe) || (!increasing && bins[i].woe > bins[i-1].woe)) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   void merge_bins(std::vector<BinInfo>& bins, int idx1, int idx2, int total_pos, int total_neg) {
//     BinInfo& bin1 = bins[idx1];
//     BinInfo& bin2 = bins[idx2];
//     bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
//     bin1.count += bin2.count;
//     bin1.count_pos += bin2.count_pos;
//     bin1.count_neg += bin2.count_neg;
//     bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
//     bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
//     
//     bins.erase(bins.begin() + idx2);
//   }
//   
//   void split_bin(std::vector<BinInfo>& bins, int idx, int total_pos, int total_neg) {
//     BinInfo& bin = bins[idx];
//     if (bin.categories.size() < 2) return;  // Can't split a bin with only one category
//     
//     size_t split_point = bin.categories.size() / 2;
//     BinInfo new_bin;
//     
//     new_bin.categories.insert(new_bin.categories.end(),
//                               bin.categories.begin() + split_point,
//                               bin.categories.end());
//     bin.categories.erase(bin.categories.begin() + split_point,
//                          bin.categories.end());
//     
//     // Recalculate counts for both bins
//     int new_count_pos = 0, new_count_neg = 0;
//     for (const auto& cat : new_bin.categories) {
//       for (size_t i = 0; i < feature.size(); ++i) {
//         if (feature[i] == cat) {
//           new_count_pos += target[i];
//           new_count_neg += (1 - target[i]);
//         }
//       }
//     }
//     
//     new_bin.count = new_count_pos + new_count_neg;
//     new_bin.count_pos = new_count_pos;
//     new_bin.count_neg = new_count_neg;
//     
//     bin.count -= new_bin.count;
//     bin.count_pos -= new_bin.count_pos;
//     bin.count_neg -= new_bin.count_neg;
//     
//     bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
//     bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
//     
//     new_bin.woe = calculate_woe(new_bin.count_pos, new_bin.count_neg, total_pos, total_neg);
//     new_bin.iv = calculate_iv(new_bin.count_pos, new_bin.count_neg, total_pos, total_neg);
//     
//     bins.insert(bins.begin() + idx + 1, std::move(new_bin));
//   }
//   
// public:
//   // Constructor
//   OptimalBinningCategoricalUDT(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, 
//                                int max_n_prebins_ = 20, std::string bin_separator_ = "%;%",
//                                double convergence_threshold_ = 1e-6, int max_iterations_ = 1000) :
//   min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_), convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//   converged(false), iterations_run(0) {}
//   
//   // Fit Function
//   void fit(const std::vector<std::string>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
//     
//     std::vector<BinInfo> initial_bins;
//     int total_pos = 0, total_neg = 0;
//     compute_initial_bins(initial_bins, total_pos, total_neg);
//     size_t unique_categories = initial_bins.size();
//     
//     validate_input_parameters(unique_categories);
//     validate_input_data();
//     
//     calculate_woe_iv(initial_bins, total_pos, total_neg);
//     
//     // If ncat <= max_bins, no optimization needed
//     if (unique_categories <= static_cast<size_t>(max_bins)) {
//       final_bins = std::move(initial_bins);
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.count < b.count;
//     });
//     
//     while (initial_bins.size() > static_cast<size_t>(max_n_prebins)) {
//       merge_bins(initial_bins, 0, 1, total_pos, total_neg);
//       std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//         return a.count < b.count;
//       });
//     }
//     
//     final_bins = std::move(initial_bins);
//     
//     std::sort(final_bins.begin(), final_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.woe < b.woe;
//     });
//     
//     double prev_total_iv = 0.0;
//     for (const auto& bin : final_bins) {
//       prev_total_iv += bin.iv;
//     }
//     
//     converged = false;
//     iterations_run = 0;
//     
//     while (iterations_run < max_iterations) {
//       if (final_bins.size() <= static_cast<size_t>(min_bins)) {
//         // Split the bin with the highest IV if we're below min_bins
//         while (final_bins.size() < static_cast<size_t>(min_bins)) {
//           auto max_iv_bin = std::max_element(final_bins.begin(), final_bins.end(),
//                                              [](const BinInfo& a, const BinInfo& b) { return a.iv < b.iv; });
//           int split_idx = std::distance(final_bins.begin(), max_iv_bin);
//           split_bin(final_bins, split_idx, total_pos, total_neg);
//         }
//       } else if (final_bins.size() > static_cast<size_t>(max_bins)) {
//         // Merge bins if we're above max_bins
//         double min_iv_diff = std::numeric_limits<double>::max();
//         int merge_idx = -1;
//         for (size_t i = 0; i < final_bins.size() - 1; ++i) {
//           double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
//           if (iv_diff < min_iv_diff) {
//             min_iv_diff = iv_diff;
//             merge_idx = static_cast<int>(i);
//           }
//         }
//         
//         if (merge_idx == -1) break;
//         
//         merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
//       } else if (!is_monotonic(final_bins)) {
//         // Try to improve monotonicity
//         double min_iv_diff = std::numeric_limits<double>::max();
//         int merge_idx = -1;
//         for (size_t i = 0; i < final_bins.size() - 1; ++i) {
//           double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
//           if (iv_diff < min_iv_diff) {
//             min_iv_diff = iv_diff;
//             merge_idx = static_cast<int>(i);
//           }
//         }
//         
//         if (merge_idx == -1) break;
//         
//         merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
//       } else {
//         // We've reached a valid state
//         break;
//       }
//       
//       double total_iv = 0.0;
//       for (const auto& bin : final_bins) {
//         total_iv += bin.iv;
//       }
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations_run++;
//     }
//     
//     if (iterations_run >= max_iterations) {
//       Rcpp::warning("Maximum iterations reached. Optimal binning may not have converged.");
//     }
//   }
//   
//   // Getter for output
//   Rcpp::List get_woe_bin() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_vals;
//     std::vector<double> iv_vals;
//     std::vector<int> counts;
//     std::vector<int> count_pos;
//     std::vector<int> count_neg;
//     
//     for (const auto& bin : final_bins) {
//       std::string label = std::accumulate(std::next(bin.categories.begin()), bin.categories.end(),
//                                           bin.categories[0],
//                                                         [this](std::string a, std::string b) {
//                                                           return std::move(a) + this->bin_separator + b;
//                                                         });
//       bin_labels.push_back(label);
//       woe_vals.push_back(bin.woe);
//       iv_vals.push_back(bin.iv);
//       counts.push_back(bin.count);
//       count_pos.push_back(bin.count_pos);
//       count_neg.push_back(bin.count_neg);
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_vals,
//       Rcpp::Named("iv") = iv_vals,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = count_pos,
//       Rcpp::Named("count_neg") = count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Unsupervised Decision Tree (UDT)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using an Unsupervised Decision Tree (UDT) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator used for merging category names (default: "%;%").
// //' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //'
// //' @return A list containing bin information:
// //' \item{bins}{A character vector of bin labels}
// //' \item{woe}{A numeric vector of WOE values for each bin}
// //' \item{iv}{A numeric vector of IV values for each bin}
// //' \item{count}{An integer vector of total counts for each bin}
// //' \item{count_pos}{An integer vector of positive target counts for each bin}
// //' \item{count_neg}{An integer vector of negative target counts for each bin}
// //' \item{converged}{A logical indicating whether the algorithm converged}
// //' \item{iterations}{An integer indicating the number of iterations run}
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' 1. Input validation and preprocessing
// //' 2. Initial binning based on unique categories
// //' 3. Merge bins to respect max_n_prebins
// //' 4. Iterative merging or splitting of bins to respect min_bins and max_bins
// //' 5. Ensure monotonicity of WOE values across bins (if possible)
// //' 6. Respect min_bins and max_bins constraints
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //'
// //' WOE = ln((Distribution of Good) / (Distribution of Bad))
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //'
// //' IV = (Distribution of Good - Distribution of Bad) * WOE
// //'
// //' The algorithm aims to find an optimal binning solution while respecting the specified constraints.
// //' It uses a convergence threshold and maximum number of iterations to ensure stability and prevent infinite loops.
// //'
// //' @references
// //' \itemize{
// //'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_udt(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_udt(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Convert R vectors to C++ vectors
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  // Create OptimalBinningCategoricalUDT object
//  OptimalBinningCategoricalUDT binning(
//      min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations
//  );
//  
//  try {
//    // Perform optimal binning
//    binning.fit(feature_vec, target_vec);
//    
//    // Get results
//    return binning.get_woe_bin();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }









// #include <Rcpp.h>
// #include <vector>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <numeric>
// 
// // For convenience
// using namespace Rcpp;
// 
// // Define a structure to hold bin information
// struct BinInfo {
//   std::vector<std::string> categories;
//   double woe;
//   double iv;
//   int count;
//   int count_pos;
//   int count_neg;
//   
//   // Constructor to initialize counts
//   BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
// };
// 
// // OptimalBinningCategoricalUDT Class
// class OptimalBinningCategoricalUDT {
// private:
//   // Parameters
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   // Data
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   std::vector<BinInfo> final_bins;
//   bool converged;
//   int iterations_run;
//   
//   // Helper Functions
//   void validate_input_parameters(const size_t unique_categories) {
//     min_bins = std::max(2, std::min(static_cast<int>(unique_categories), min_bins));
//     max_bins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_bins));
//     bin_cutoff = std::max(0.0, std::min(1.0, bin_cutoff));
//     max_n_prebins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_n_prebins));
//   }
//   
//   void validate_input_data() const {
//     if (feature.empty() || target.empty()) {
//       throw std::invalid_argument("Feature and target vectors must not be empty.");
//     }
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same length.");
//     }
//     std::unordered_set<int> unique_targets(target.begin(), target.end());
//     if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
//       throw std::invalid_argument("Target variable must be binary (0 and 1).");
//     }
//   }
//   
//   double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     const double epsilon = 1e-10;
//     double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
//     double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
//     return std::log(event_rate / non_event_rate);
//   }
//   
//   double calculate_iv(int count_pos, int count_neg, int total_pos, int total_neg) const {
//     const double epsilon = 1e-10;
//     double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
//     double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
//     return (event_rate - non_event_rate) * std::log(event_rate / non_event_rate);
//   }
//   
//   void compute_initial_bins(std::vector<BinInfo>& initial_bins, int& total_pos, int& total_neg) {
//     std::unordered_map<std::string, BinInfo> temp_bins;
//     total_pos = 0;
//     total_neg = 0;
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int tgt = target[i];
//       auto& bin = temp_bins[cat];
//       if (bin.categories.empty()) {
//         bin.categories.push_back(cat);
//       }
//       bin.count++;
//       bin.count_pos += tgt;
//       bin.count_neg += (1 - tgt);
//       
//       total_pos += tgt;
//       total_neg += (1 - tgt);
//     }
//     
//     initial_bins.clear();
//     for (auto& pair : temp_bins) {
//       initial_bins.push_back(std::move(pair.second));
//     }
//   }
//   
//   void calculate_woe_iv(std::vector<BinInfo>& bins, int total_pos, int total_neg) {
//     for (auto& bin : bins) {
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
//       bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
//     }
//   }
//   
//   bool is_monotonic(const std::vector<BinInfo>& bins) const {
//     if (bins.size() <= 2) return true;
//     bool increasing = bins[1].woe > bins[0].woe;
//     for (size_t i = 2; i < bins.size(); ++i) {
//       if ((increasing && bins[i].woe < bins[i-1].woe) || (!increasing && bins[i].woe > bins[i-1].woe)) {
//         return false;
//       }
//     }
//     return true;
//   }
//   
//   void merge_bins(std::vector<BinInfo>& bins, int idx1, int idx2, int total_pos, int total_neg) {
//     BinInfo& bin1 = bins[idx1];
//     BinInfo& bin2 = bins[idx2];
//     bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
//     bin1.count += bin2.count;
//     bin1.count_pos += bin2.count_pos;
//     bin1.count_neg += bin2.count_neg;
//     bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
//     bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
//     
//     bins.erase(bins.begin() + idx2);
//   }
//   
// public:
//   // Constructor
//   OptimalBinningCategoricalUDT(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, 
//                                int max_n_prebins_ = 20, std::string bin_separator_ = "%;%",
//                                double convergence_threshold_ = 1e-6, int max_iterations_ = 1000) :
//   min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//   bin_separator(bin_separator_), convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//   converged(false), iterations_run(0) {}
//   
//   // Fit Function
//   void fit(const std::vector<std::string>& feature_, const std::vector<int>& target_) {
//     feature = feature_;
//     target = target_;
//     
//     std::vector<BinInfo> initial_bins;
//     int total_pos = 0, total_neg = 0;
//     compute_initial_bins(initial_bins, total_pos, total_neg);
//     size_t unique_categories = initial_bins.size();
//     
//     validate_input_parameters(unique_categories);
//     validate_input_data();
//     
//     calculate_woe_iv(initial_bins, total_pos, total_neg);
//     
//     // If ncat <= max_bins, no optimization needed
//     if (unique_categories <= static_cast<size_t>(max_bins)) {
//       final_bins = std::move(initial_bins);
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.count < b.count;
//     });
//     
//     while (initial_bins.size() > static_cast<size_t>(max_n_prebins)) {
//       merge_bins(initial_bins, 0, 1, total_pos, total_neg);
//       std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//         return a.count < b.count;
//       });
//     }
//     
//     final_bins = std::move(initial_bins);
//     
//     std::sort(final_bins.begin(), final_bins.end(), [](const BinInfo& a, const BinInfo& b) {
//       return a.woe < b.woe;
//     });
//     
//     double prev_total_iv = 0.0;
//     for (const auto& bin : final_bins) {
//       prev_total_iv += bin.iv;
//     }
//     
//     converged = false;
//     iterations_run = 0;
//     
//     while (final_bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
//       double min_iv_diff = std::numeric_limits<double>::max();
//       int merge_idx = -1;
//       for (size_t i = 0; i < final_bins.size() - 1; ++i) {
//         double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
//         if (iv_diff < min_iv_diff) {
//           min_iv_diff = iv_diff;
//           merge_idx = static_cast<int>(i);
//         }
//       }
//       
//       if (merge_idx == -1) break;
//       
//       merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
//       
//       double total_iv = 0.0;
//       for (const auto& bin : final_bins) {
//         total_iv += bin.iv;
//       }
//       
//       if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       
//       prev_total_iv = total_iv;
//       iterations_run++;
//     }
//     
//     // Ensure monotonicity if possible
//     while (!is_monotonic(final_bins) && final_bins.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
//       double min_iv_diff = std::numeric_limits<double>::max();
//       int merge_idx = -1;
//       for (size_t i = 0; i < final_bins.size() - 1; ++i) {
//         double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
//         if (iv_diff < min_iv_diff) {
//           min_iv_diff = iv_diff;
//           merge_idx = static_cast<int>(i);
//         }
//       }
//       
//       if (merge_idx == -1) break;
//       
//       merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
//       iterations_run++;
//     }
//     
//     if (iterations_run >= max_iterations) {
//       Rcpp::warning("Maximum iterations reached. Optimal binning may not have converged.");
//     }
//   }
//   
//   // Getter for output
//   Rcpp::List get_woe_bin() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_vals;
//     std::vector<double> iv_vals;
//     std::vector<int> counts;
//     std::vector<int> count_pos;
//     std::vector<int> count_neg;
//     
//     for (const auto& bin : final_bins) {
//       std::string label = std::accumulate(std::next(bin.categories.begin()), bin.categories.end(),
//                                           bin.categories[0],
//                                                         [this](std::string a, std::string b) {
//                                                           return std::move(a) + this->bin_separator + b;
//                                                         });
//       bin_labels.push_back(label);
//       woe_vals.push_back(bin.woe);
//       iv_vals.push_back(bin.iv);
//       counts.push_back(bin.count);
//       count_pos.push_back(bin.count_pos);
//       count_neg.push_back(bin.count_neg);
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_vals,
//       Rcpp::Named("iv") = iv_vals,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = count_pos,
//       Rcpp::Named("count_neg") = count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Unsupervised Decision Tree (UDT)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using an Unsupervised Decision Tree (UDT) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator used for merging category names (default: "%;%").
// //' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
// //'
// //' @return A list containing bin information:
// //' \item{bins}{A character vector of bin labels}
// //' \item{woe}{A numeric vector of WOE values for each bin}
// //' \item{iv}{A numeric vector of IV values for each bin}
// //' \item{count}{An integer vector of total counts for each bin}
// //' \item{count_pos}{An integer vector of positive target counts for each bin}
// //' \item{count_neg}{An integer vector of negative target counts for each bin}
// //' \item{converged}{A logical indicating whether the algorithm converged}
// //' \item{iterations}{An integer indicating the number of iterations run}
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' 1. Input validation and preprocessing
// //' 2. Initial binning based on unique categories
// //' 3. Merge bins to respect max_n_prebins
// //' 4. Iterative merging of bins based on minimum IV difference
// //' 5. Ensure monotonicity of WOE values across bins (if possible)
// //' 6. Respect min_bins and max_bins constraints
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //'
// //' WOE = ln((Distribution of Good) / (Distribution of Bad))
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //'
// //' IV = (Distribution of Good - Distribution of Bad) * WOE
// //'
// //' The algorithm aims to find an optimal binning solution while respecting the specified constraints.
// //' It uses a convergence threshold and maximum number of iterations to ensure stability and prevent infinite loops.
// //'
// //' @references
// //' \itemize{
// //'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_udt(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_udt(
//    Rcpp::IntegerVector target,
//    Rcpp::CharacterVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    std::string bin_separator = "%;%",
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  // Convert R vectors to C++ vectors
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  
//  // Create OptimalBinningCategoricalUDT object
//  OptimalBinningCategoricalUDT binning(
//      min_bins, max_bins, bin_cutoff, max_n_prebins,
//      bin_separator, convergence_threshold, max_iterations
//  );
//  
//  try {
//    // Perform optimal binning
//    binning.fit(feature_vec, target_vec);
//    
//    // Get results
//    return binning.get_woe_bin();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
