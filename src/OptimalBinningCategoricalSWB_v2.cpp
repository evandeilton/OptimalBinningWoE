#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

using namespace Rcpp;

class OptimalBinningCategoricalSWB {
private:
  struct BinStats {
    std::vector<std::string> categories;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    
    BinStats() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
  };
  
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<BinStats> bins;
  int total_pos;
  int total_neg;
  bool converged;
  int iterations_run;
  
  double calculate_woe(int pos, int neg) const {
    if (pos == 0 || neg == 0) return 0.0;  // Avoid division by zero and log(0)
    double pos_rate = static_cast<double>(pos) / total_pos;
    double neg_rate = static_cast<double>(neg) / total_neg;
    return std::log(pos_rate / neg_rate);
  }
  
  double calculate_iv(const std::vector<BinStats>& current_bins) const {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      if (pos_rate > 0 && neg_rate > 0) {
        iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
      }
    }
    return iv;
  }
  
  bool is_monotonic(const std::vector<BinStats>& current_bins) const {
    if (current_bins.size() <= 2) return true;  // Always monotonic with 2 or fewer bins
    bool increasing = true;
    bool decreasing = true;
    
    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woe < current_bins[i - 1].woe) {
        increasing = false;
      }
      if (current_bins[i].woe > current_bins[i - 1].woe) {
        decreasing = false;
      }
      if (!increasing && !decreasing) break;
    }
    
    return increasing || decreasing;
  }
  
  void initialize_bins() {
    std::unordered_map<std::string, BinStats> initial_bins;
    total_pos = 0;
    total_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int target_val = target[i];
      
      auto& bin = initial_bins[cat];
      if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
        bin.categories.push_back(cat);
      }
      bin.count++;
      bin.count_pos += target_val;
      bin.count_neg += 1 - target_val;
      
      total_pos += target_val;
      total_neg += 1 - target_val;
    }
    
    double count_threshold = bin_cutoff * feature.size();
    
    // Now separate bins
    std::vector<BinStats> temp_bins;
    BinStats low_freq_bin;
    
    for (auto& pair : initial_bins) {
      if (pair.second.count >= count_threshold) {
        pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
        temp_bins.push_back(std::move(pair.second));
      } else {
        // Add to low_freq_bin
        low_freq_bin.count += pair.second.count;
        low_freq_bin.count_pos += pair.second.count_pos;
        low_freq_bin.count_neg += pair.second.count_neg;
        low_freq_bin.categories.insert(low_freq_bin.categories.end(),
                                       pair.second.categories.begin(),
                                       pair.second.categories.end());
      }
    }
    
    if (low_freq_bin.count > 0) {
      low_freq_bin.woe = calculate_woe(low_freq_bin.count_pos, low_freq_bin.count_neg);
      temp_bins.push_back(std::move(low_freq_bin));
    }
    
    bins = std::move(temp_bins);
    
    // Now sort bins by WOE
    std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
      return a.woe < b.woe;
    });
    
    // Respect max_n_prebins while ensuring min_bins
    while (bins.size() > max_n_prebins && bins.size() > min_bins) {
      merge_adjacent_bins();
    }
  }
  
  void merge_adjacent_bins() {
    if (bins.size() <= min_bins) return;  // Cannot merge if bins.size() <= min_bins
    
    double min_iv_loss = std::numeric_limits<double>::max();
    size_t merge_index = 0;
    
    double original_iv = calculate_iv(bins);
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      BinStats merged_bin = bins[i];
      merged_bin.count += bins[i + 1].count;
      merged_bin.count_pos += bins[i + 1].count_pos;
      merged_bin.count_neg += bins[i + 1].count_neg;
      merged_bin.categories.insert(merged_bin.categories.end(),
                                   bins[i + 1].categories.begin(),
                                   bins[i + 1].categories.end());
      merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
      
      std::vector<BinStats> temp_bins = bins;
      temp_bins[i] = merged_bin;
      temp_bins.erase(temp_bins.begin() + i + 1);
      
      double new_iv = calculate_iv(temp_bins);
      double iv_loss = original_iv - new_iv;
      
      if (iv_loss < min_iv_loss) {
        min_iv_loss = iv_loss;
        merge_index = i;
      }
    }
    
    bins[merge_index].count += bins[merge_index + 1].count;
    bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
    bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
    bins[merge_index].categories.insert(bins[merge_index].categories.end(),
                                        bins[merge_index + 1].categories.begin(),
                                        bins[merge_index + 1].categories.end());
    bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
    bins.erase(bins.begin() + merge_index + 1);
  }
  
  void optimize_bins() {
    double prev_iv = calculate_iv(bins);
    converged = false;
    iterations_run = 0;
    
    while (iterations_run < max_iterations) {
      if (is_monotonic(bins) && bins.size() <= max_bins && bins.size() >= min_bins) {
        converged = true;
        break;
      }
      
      if (bins.size() > min_bins) {
        merge_adjacent_bins();
      } else if (bins.size() < min_bins) {
        // Split the bin with the highest IV
        split_highest_iv_bin();
      } else {
        // We have exactly min_bins, but not monotonic. Try to improve monotonicity.
        improve_monotonicity();
      }
      
      double current_iv = calculate_iv(bins);
      if (std::abs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      prev_iv = current_iv;
      iterations_run++;
    }
    
    // Ensure we have at least min_bins
    while (bins.size() < min_bins) {
      split_highest_iv_bin();
    }
    
    // Ensure we have at most max_bins
    while (bins.size() > max_bins) {
      merge_adjacent_bins();
    }
    
    double total_iv = calculate_iv(bins);
    
    for (auto& bin : bins) {
      double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
      double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }
  
  void split_highest_iv_bin() {
    if (bins.empty()) return;
    
    size_t highest_iv_index = 0;
    double highest_iv = bins[0].iv;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].iv > highest_iv) {
        highest_iv = bins[i].iv;
        highest_iv_index = i;
      }
    }
    
    BinStats& bin_to_split = bins[highest_iv_index];
    if (bin_to_split.categories.size() < 2) return;  // Can't split a bin with only one category
    
    size_t split_point = bin_to_split.categories.size() / 2;
    BinStats new_bin;
    
    new_bin.categories.insert(new_bin.categories.end(),
                              bin_to_split.categories.begin() + split_point,
                              bin_to_split.categories.end());
    bin_to_split.categories.erase(bin_to_split.categories.begin() + split_point,
                                  bin_to_split.categories.end());
    
    // Recalculate counts and WOE for both bins
    bin_to_split.count = 0;
    bin_to_split.count_pos = 0;
    bin_to_split.count_neg = 0;
    new_bin.count = 0;
    new_bin.count_pos = 0;
    new_bin.count_neg = 0;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int target_val = target[i];
      
      if (std::find(bin_to_split.categories.begin(), bin_to_split.categories.end(), cat) != bin_to_split.categories.end()) {
        bin_to_split.count++;
        bin_to_split.count_pos += target_val;
        bin_to_split.count_neg += 1 - target_val;
      } else if (std::find(new_bin.categories.begin(), new_bin.categories.end(), cat) != new_bin.categories.end()) {
        new_bin.count++;
        new_bin.count_pos += target_val;
        new_bin.count_neg += 1 - target_val;
      }
    }
    
    bin_to_split.woe = calculate_woe(bin_to_split.count_pos, bin_to_split.count_neg);
    new_bin.woe = calculate_woe(new_bin.count_pos, new_bin.count_neg);
    
    bins.insert(bins.begin() + highest_iv_index + 1, std::move(new_bin));
  }
  
  void improve_monotonicity() {
    // This is a simple implementation. More sophisticated approaches could be used.
    for (size_t i = 1; i < bins.size() - 1; ++i) {
      if ((bins[i].woe < bins[i-1].woe && bins[i].woe < bins[i+1].woe) ||
          (bins[i].woe > bins[i-1].woe && bins[i].woe > bins[i+1].woe)) {
        // This bin breaks monotonicity. Merge it with the neighbor that results in higher IV.
        double iv_merge_left = calculate_iv({bins[i-1], bins[i], bins[i+1]});
        double iv_merge_right = calculate_iv({bins[i-1], bins[i+1]});
        
        if (iv_merge_left > iv_merge_right) {
          merge_bins(i-1, i);
        } else {
          merge_bins(i, i+1);
        }
        break;  // Only make one change per iteration
      }
    }
  }
  
  void merge_bins(size_t index1, size_t index2) {
    bins[index1].count += bins[index2].count;
    bins[index1].count_pos += bins[index2].count_pos;
    bins[index1].count_neg += bins[index2].count_neg;
    bins[index1].categories.insert(bins[index1].categories.end(),
                                   bins[index2].categories.begin(),
                                   bins[index2].categories.end());
    bins[index1].woe = calculate_woe(bins[index1].count_pos, bins[index1].count_neg);
    bins.erase(bins.begin() + index2);
  }
  
  static std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
    std::ostringstream result;
    for (size_t i = 0; i < categories.size(); ++i) {
      if (i > 0) result << delimiter;
      result << categories[i];
    }
    return result.str();
  }
  
public:
  OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
                               const std::vector<int>& target,
                               int min_bins = 3,
                               int max_bins = 5,
                               double bin_cutoff = 0.05,
                               int max_n_prebins = 20,
                               std::string bin_separator = "%;%",
                               double convergence_threshold = 1e-6,
                               int max_iterations = 1000)
    : feature(feature),
      target(target),
      min_bins(min_bins),
      max_bins(max_bins),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins),
      bin_separator(bin_separator),
      convergence_threshold(convergence_threshold),
      max_iterations(max_iterations),
      converged(false),
      iterations_run(0) {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length");
    }
    if (min_bins < 2 || max_bins < min_bins) {Rcpp::stop("Invalid bin constraints");
    }
    int ncat = std::unordered_set<std::string>(feature.begin(), feature.end()).size();
    min_bins = std::max(2, std::min(min_bins, ncat));
    max_bins = std::max(min_bins, std::min(max_bins, ncat));
    if (max_n_prebins < min_bins) {
      Rcpp::stop("max_n_prebins cannot be less than min_bins");
    }
  }
  
  void fit() {
    initialize_bins();
    optimize_bins();
  }
  
  Rcpp::List get_results() const {
    std::vector<std::string> bin_categories;
    std::vector<double> woes;
    std::vector<double> ivs;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    
    for (const auto& bin : bins) {
      std::string bin_name = join_categories(bin.categories, bin_separator);
      bin_categories.push_back(bin_name);
      woes.push_back(bin.woe);
      ivs.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_categories,
      Rcpp::Named("woe") = woes,
      Rcpp::Named("iv") = ivs,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run);
  }
};

//' @title Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)
//'
//' @description
//' This function performs optimal binning for categorical variables using a Sliding Window Binning (SWB) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator used for merging category names (default: "%;%").
//' @param convergence_threshold Threshold for convergence of IV (default: 1e-6).
//' @param max_iterations Maximum number of iterations for optimization (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bin}{A character vector of bin labels}
//' \item{woe}{A numeric vector of WOE values for each bin}
//' \item{iv}{A numeric vector of IV values for each bin}
//' \item{count}{An integer vector of total counts for each bin}
//' \item{count_pos}{An integer vector of positive class counts for each bin}
//' \item{count_neg}{An integer vector of negative class counts for each bin}
//' \item{converged}{A logical value indicating whether the algorithm converged}
//' \item{iterations}{An integer value indicating the number of iterations run}
//'
//' @details
//' The algorithm performs the following steps:
//' \enumerate{
//'   \item Initialize bins for each unique category, merging low-frequency categories based on bin_cutoff
//'   \item Sort bins by their WOE values
//'   \item Merge adjacent bins iteratively, minimizing information loss
//'   \item Optimize the number of bins while maintaining monotonicity
//'   \item Calculate final WOE and IV values for each bin
//' }
//'
//' The Weight of Evidence (WOE) is calculated as:
//' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
//'
//' The Information Value (IV) for each bin is calculated as:
//' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
//'
//' @references
//' \itemize{
//'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
//'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_swb(target, feature)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
                                           Rcpp::StringVector feature,
                                           int min_bins = 3,
                                           int max_bins = 5,
                                           double bin_cutoff = 0.05,
                                           int max_n_prebins = 20,
                                           std::string bin_separator = "%;%",
                                           double convergence_threshold = 1e-6,
                                           int max_iterations = 1000) {
  try {
    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
    
    OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins,
                                        bin_cutoff, max_n_prebins, bin_separator,
                                        convergence_threshold, max_iterations);
    binner.fit();
    return binner.get_results();
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}










// #include <Rcpp.h>
// #include <unordered_map>
// #include <unordered_set>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// class OptimalBinningCategoricalSWB {
// private:
//   struct BinStats {
//     std::vector<std::string> categories;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
// 
//     BinStats() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
//   };
// 
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
// 
//   std::vector<BinStats> bins;
//   int total_pos;
//   int total_neg;
//   bool converged;
//   int iterations_run;
// 
//   double calculate_woe(int pos, int neg) const {
//     if (pos == 0 || neg == 0) return 0.0;  // Avoid division by zero and log(0)
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     return std::log(pos_rate / neg_rate);
//   }
// 
//   double calculate_iv(const std::vector<BinStats>& current_bins) const {
//     double iv = 0.0;
//     for (const auto& bin : current_bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       if (pos_rate > 0 && neg_rate > 0) {
//         iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
//       }
//     }
//     return iv;
//   }
// 
//   bool is_monotonic(const std::vector<BinStats>& current_bins) const {
//     if (current_bins.size() <= 2) return true;  // Always monotonic with 2 or fewer bins
//     bool increasing = true;
//     bool decreasing = true;
// 
//     for (size_t i = 1; i < current_bins.size(); ++i) {
//       if (current_bins[i].woe < current_bins[i - 1].woe) {
//         increasing = false;
//       }
//       if (current_bins[i].woe > current_bins[i - 1].woe) {
//         decreasing = false;
//       }
//       if (!increasing && !decreasing) break;
//     }
// 
//     return increasing || decreasing;
//   }
// 
//   void initialize_bins() {
//     std::unordered_map<std::string, BinStats> initial_bins;
//     total_pos = 0;
//     total_neg = 0;
// 
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int target_val = target[i];
// 
//       auto& bin = initial_bins[cat];
//       if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
//         bin.categories.push_back(cat);
//       }
//       bin.count++;
//       bin.count_pos += target_val;
//       bin.count_neg += 1 - target_val;
// 
//       total_pos += target_val;
//       total_neg += 1 - target_val;
//     }
// 
//     double count_threshold = bin_cutoff * feature.size();
// 
//     // Now separate bins
//     std::vector<BinStats> temp_bins;
//     BinStats low_freq_bin;
// 
//     for (auto& pair : initial_bins) {
//       if (pair.second.count >= count_threshold) {
//         pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
//         temp_bins.push_back(std::move(pair.second));
//       } else {
//         // Add to low_freq_bin
//         low_freq_bin.count += pair.second.count;
//         low_freq_bin.count_pos += pair.second.count_pos;
//         low_freq_bin.count_neg += pair.second.count_neg;
//         low_freq_bin.categories.insert(low_freq_bin.categories.end(),
//                                        pair.second.categories.begin(),
//                                        pair.second.categories.end());
//       }
//     }
// 
//     if (low_freq_bin.count > 0) {
//       low_freq_bin.woe = calculate_woe(low_freq_bin.count_pos, low_freq_bin.count_neg);
//       temp_bins.push_back(std::move(low_freq_bin));
//     }
// 
//     bins = std::move(temp_bins);
// 
//     // Now sort bins by WOE
//     std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
//       return a.woe < b.woe;
//     });
// 
//     while (bins.size() > max_n_prebins && bins.size() > min_bins) {
//       merge_adjacent_bins();
//     }
//   }
// 
//   void merge_adjacent_bins() {
//     if (bins.size() <= min_bins) return;  // Cannot merge if bins.size() <= min_bins
// 
//     double min_iv_loss = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
// 
//     double original_iv = calculate_iv(bins);
// 
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       BinStats merged_bin = bins[i];
//       merged_bin.count += bins[i + 1].count;
//       merged_bin.count_pos += bins[i + 1].count_pos;
//       merged_bin.count_neg += bins[i + 1].count_neg;
//       merged_bin.categories.insert(merged_bin.categories.end(),
//                                    bins[i + 1].categories.begin(),
//                                    bins[i + 1].categories.end());
//       merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
// 
//       std::vector<BinStats> temp_bins = bins;
//       temp_bins[i] = merged_bin;
//       temp_bins.erase(temp_bins.begin() + i + 1);
// 
//       double new_iv = calculate_iv(temp_bins);
//       double iv_loss = original_iv - new_iv;
// 
//       if (iv_loss < min_iv_loss) {
//         min_iv_loss = iv_loss;
//         merge_index = i;
//       }
//     }
// 
//     bins[merge_index].count += bins[merge_index + 1].count;
//     bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
//     bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
//     bins[merge_index].categories.insert(bins[merge_index].categories.end(),
//                                         bins[merge_index + 1].categories.begin(),
//                                         bins[merge_index + 1].categories.end());
//     bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
//     bins.erase(bins.begin() + merge_index + 1);
//   }
// 
//   void optimize_bins() {
//     double prev_iv = calculate_iv(bins);
//     converged = false;
//     iterations_run = 0;
// 
//     while (bins.size() > min_bins && iterations_run < max_iterations) {
//       if (is_monotonic(bins) && bins.size() <= max_bins) {
//         converged = true;
//         break;
//       }
//       merge_adjacent_bins();
// 
//       double current_iv = calculate_iv(bins);
//       if (std::abs(current_iv - prev_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       prev_iv = current_iv;
//       iterations_run++;
//     }
// 
//     // Force monotonicity if possible, respecting min_bins
//     while (!is_monotonic(bins) && bins.size() > min_bins) {
//       merge_adjacent_bins();
//       iterations_run++;
//     }
// 
//     double total_iv = calculate_iv(bins);
// 
//     for (auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//     }
//   }
// 
//   static std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
//     std::ostringstream result;
//     for (size_t i = 0; i < categories.size(); ++i) {
//       if (i > 0) result << delimiter;
//       result << categories[i];
//     }
//     return result.str();
//   }
// 
// public:
//   OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20,
//                                std::string bin_separator = "%;%",
//                                double convergence_threshold = 1e-6,
//                                int max_iterations = 1000)
//     : feature(feature),
//       target(target),
//       min_bins(min_bins),
//       max_bins(max_bins),
//       bin_cutoff(bin_cutoff),
//       max_n_prebins(max_n_prebins),
//       bin_separator(bin_separator),
//       convergence_threshold(convergence_threshold),
//       max_iterations(max_iterations),
//       converged(false),
//       iterations_run(0) {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2 || max_bins < min_bins) {
//       Rcpp::stop("Invalid bin constraints");
//     }
//     int ncat = std::unordered_set<std::string>(feature.begin(), feature.end()).size();
//     min_bins = std::max(2, std::min(min_bins, ncat));
//     max_bins = std::max(min_bins, std::min(max_bins, ncat));
//     if (max_n_prebins < min_bins) {
//       Rcpp::stop("max_n_prebins cannot be less than min_bins");
//     }
//   }
// 
//   void fit() {
//     initialize_bins();
//     optimize_bins();
//   }
// 
//   Rcpp::List get_results() const {
//     std::vector<std::string> bin_categories;
//     std::vector<double> woes;
//     std::vector<double> ivs;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
// 
//     for (const auto& bin : bins) {
//       std::string bin_name = join_categories(bin.categories, bin_separator);
//       bin_categories.push_back(bin_name);
//       woes.push_back(bin.woe);
//       ivs.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//     }
// 
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_categories,
//       Rcpp::Named("woe") = woes,
//       Rcpp::Named("iv") = ivs,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run);
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Sliding Window Binning (SWB) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator used for merging category names (default: "%;%").
// //' @param convergence_threshold Threshold for convergence of IV (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for optimization (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bin}{A character vector of bin labels}
// //' \item{woe}{A numeric vector of WOE values for each bin}
// //' \item{iv}{A numeric vector of IV values for each bin}
// //' \item{count}{An integer vector of total counts for each bin}
// //' \item{count_pos}{An integer vector of positive class counts for each bin}
// //' \item{count_neg}{An integer vector of negative class counts for each bin}
// //' \item{converged}{A logical value indicating whether the algorithm converged}
// //' \item{iterations}{An integer value indicating the number of iterations run}
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Initialize bins for each unique category, merging low-frequency categories based on bin_cutoff
// //'   \item Sort bins by their WOE values
// //'   \item Merge adjacent bins iteratively, minimizing information loss
// //'   \item Optimize the number of bins while maintaining monotonicity
// //'   \item Calculate final WOE and IV values for each bin
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
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
// //' result <- optimal_binning_categorical_swb(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
//                                           Rcpp::StringVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           std::string bin_separator = "%;%",
//                                           double convergence_threshold = 1e-6,
//                                           int max_iterations = 1000) {
//  try {
//    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
// 
//    OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins,
//                                        bin_cutoff, max_n_prebins, bin_separator,
//                                        convergence_threshold, max_iterations);
//    binner.fit();
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
























// #include <Rcpp.h>
// #include <unordered_map>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <sstream>
// 
// using namespace Rcpp;
// 
// class OptimalBinningCategoricalSWB {
// private:
//   struct BinStats {
//     std::vector<std::string> categories;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//     
//     BinStats() : count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
//   };
//   
//   std::vector<std::string> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   std::vector<BinStats> bins;
//   int total_pos;
//   int total_neg;
//   bool converged;
//   int iterations_run;
//   
//   double calculate_woe(int pos, int neg) const {
//     if (pos == 0 || neg == 0) return 0.0;  // Avoid division by zero and log(0)
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     return std::log(pos_rate / neg_rate);
//   }
//   
//   double calculate_iv(const std::vector<BinStats>& current_bins) const {
//     double iv = 0.0;
//     for (const auto& bin : current_bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       if (pos_rate > 0 && neg_rate > 0) {
//         iv += (pos_rate - neg_rate) * std::log(pos_rate / neg_rate);
//       }
//     }
//     return iv;
//   }
//   
//   bool is_monotonic(const std::vector<BinStats>& current_bins) const {
//     if (current_bins.size() <= 2) return true;  // Always monotonic with 2 or fewer bins
//     bool increasing = true;
//     bool decreasing = true;
//     
//     for (size_t i = 1; i < current_bins.size(); ++i) {
//       if (current_bins[i].woe < current_bins[i-1].woe) {
//         increasing = false;
//       }
//       if (current_bins[i].woe > current_bins[i-1].woe) {
//         decreasing = false;
//       }
//       if (!increasing && !decreasing) break;
//     }
//     
//     return increasing || decreasing;
//   }
//   
//   void initialize_bins() {
//     std::unordered_map<std::string, BinStats> initial_bins;
//     total_pos = 0;
//     total_neg = 0;
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       const std::string& cat = feature[i];
//       int target_val = target[i];
//       
//       auto& bin = initial_bins[cat];
//       if (std::find(bin.categories.begin(), bin.categories.end(), cat) == bin.categories.end()) {
//         bin.categories.push_back(cat);
//       }
//       bin.count++;
//       bin.count_pos += target_val;
//       bin.count_neg += 1 - target_val;
//       
//       total_pos += target_val;
//       total_neg += 1 - target_val;
//     }
//     
//     bins.reserve(initial_bins.size());
//     for (auto& pair : initial_bins) {
//       pair.second.woe = calculate_woe(pair.second.count_pos, pair.second.count_neg);
//       bins.push_back(std::move(pair.second));
//     }
//     
//     std::sort(bins.begin(), bins.end(), [](const BinStats& a, const BinStats& b) {
//       return a.woe < b.woe;
//     });
//     
//     while (bins.size() > max_n_prebins) {
//       merge_adjacent_bins();
//     }
//   }
//   
//   void merge_adjacent_bins() {
//     if (bins.size() <= 2) return;  // Cannot merge if there are 2 or fewer bins
//     
//     double min_iv_loss = std::numeric_limits<double>::max();
//     size_t merge_index = 0;
//     
//     for (size_t i = 0; i < bins.size() - 1; ++i) {
//       BinStats merged_bin = bins[i];
//       merged_bin.count += bins[i+1].count;
//       merged_bin.count_pos += bins[i+1].count_pos;
//       merged_bin.count_neg += bins[i+1].count_neg;
//       merged_bin.categories.insert(merged_bin.categories.end(),
//                                    bins[i+1].categories.begin(),
//                                    bins[i+1].categories.end());
//       merged_bin.woe = calculate_woe(merged_bin.count_pos, merged_bin.count_neg);
//       
//       std::vector<BinStats> temp_bins = bins;
//       temp_bins[i] = merged_bin;
//       temp_bins.erase(temp_bins.begin() + i + 1);
//       
//       double new_iv = calculate_iv(temp_bins);
//       double iv_loss = calculate_iv(bins) - new_iv;
//       
//       if (iv_loss < min_iv_loss) {
//         min_iv_loss = iv_loss;
//         merge_index = i;
//       }
//     }
//     
//     bins[merge_index].count += bins[merge_index+1].count;
//     bins[merge_index].count_pos += bins[merge_index+1].count_pos;
//     bins[merge_index].count_neg += bins[merge_index+1].count_neg;
//     bins[merge_index].categories.insert(bins[merge_index].categories.end(),
//                                         bins[merge_index+1].categories.begin(),
//                                         bins[merge_index+1].categories.end());
//     bins[merge_index].woe = calculate_woe(bins[merge_index].count_pos, bins[merge_index].count_neg);
//     bins.erase(bins.begin() + merge_index + 1);
//   }
//   
//   void optimize_bins() {
//     double prev_iv = calculate_iv(bins);
//     converged = false;
//     iterations_run = 0;
//     
//     while (bins.size() > min_bins && iterations_run < max_iterations) {
//       if (is_monotonic(bins) && bins.size() <= max_bins) {
//         converged = true;
//         break;
//       }
//       merge_adjacent_bins();
//       
//       double current_iv = calculate_iv(bins);
//       if (std::abs(current_iv - prev_iv) < convergence_threshold) {
//         converged = true;
//         break;
//       }
//       prev_iv = current_iv;
//       iterations_run++;
//     }
//     
//     // Force monotonicity if possible, respecting min_bins
//     while (!is_monotonic(bins) && bins.size() > min_bins) {
//       merge_adjacent_bins();
//       iterations_run++;
//     }
//     
//     double total_iv = calculate_iv(bins);
//     
//     for (auto& bin : bins) {
//       double pos_rate = static_cast<double>(bin.count_pos) / total_pos;
//       double neg_rate = static_cast<double>(bin.count_neg) / total_neg;
//       bin.iv = (pos_rate - neg_rate) * bin.woe;
//     }
//   }
//   
//   static std::string join_categories(const std::vector<std::string>& categories, const std::string& delimiter) {
//     std::ostringstream result;
//     for (size_t i = 0; i < categories.size(); ++i) {
//       if (i > 0) result << delimiter;
//       result << categories[i];
//     }
//     return result.str();
//   }
//   
// public:
//   OptimalBinningCategoricalSWB(const std::vector<std::string>& feature,
//                                const std::vector<int>& target,
//                                int min_bins = 3,
//                                int max_bins = 5,
//                                double bin_cutoff = 0.05,
//                                int max_n_prebins = 20,
//                                std::string bin_separator = "%;%",
//                                double convergence_threshold = 1e-6,
//                                int max_iterations = 1000)
//     : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
//       bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
//       convergence_threshold(convergence_threshold), max_iterations(max_iterations),
//       converged(false), iterations_run(0) {
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length");
//     }
//     if (min_bins < 2 || max_bins < min_bins) {
//       Rcpp::stop("Invalid bin constraints");
//     }
//     int ncat = std::unordered_set<std::string>(feature.begin(), feature.end()).size();
//     min_bins = std::max(2, std::min(min_bins, ncat));
//     max_bins = std::max(min_bins, std::min(max_bins, ncat));
//   }
//   
//   void fit() {
//     initialize_bins();
//     optimize_bins();
//   }
//   
//   Rcpp::List get_results() const {
//     std::vector<std::string> bin_categories;
//     std::vector<double> woes;
//     std::vector<double> ivs;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     
//     for (const auto& bin : bins) {
//       std::string bin_name = join_categories(bin.categories, bin_separator);
//       bin_categories.push_back(bin_name);
//       woes.push_back(bin.woe);
//       ivs.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_categories,
//       Rcpp::Named("woe") = woes,
//       Rcpp::Named("iv") = ivs,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title Optimal Binning for Categorical Variables using Sliding Window Binning (SWB)
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Sliding Window Binning (SWB) approach,
// //' which combines Weight of Evidence (WOE) and Information Value (IV) methods with monotonicity constraints.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
// //' @param bin_separator Separator used for merging category names (default: "%;%").
// //' @param convergence_threshold Threshold for convergence of IV (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for optimization (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \item{bins}{A character vector of bin labels}
// //' \item{woe}{A numeric vector of WOE values for each bin}
// //' \item{iv}{A numeric vector of IV values for each bin}
// //' \item{count}{An integer vector of total counts for each bin}
// //' \item{count_pos}{An integer vector of positive class counts for each bin}
// //' \item{count_neg}{An integer vector of negative class counts for each bin}
// //' \item{converged}{A logical value indicating whether the algorithm converged}
// //' \item{iterations}{An integer value indicating the number of iterations run}
// //'
// //' @details
// //' The algorithm performs the following steps:
// //' \enumerate{
// //'   \item Initialize bins for each unique category
// //'   \item Sort bins by their WOE values
// //'   \item Merge adjacent bins iteratively, minimizing information loss
// //'   \item Optimize the number of bins while maintaining monotonicity
// //'   \item Calculate final WOE and IV values for each bin
// //' }
// //'
// //' The Weight of Evidence (WOE) is calculated as:
// //' \deqn{WOE = \ln\left(\frac{\text{Proportion of Events}}{\text{Proportion of Non-Events}}\right)}
// //'
// //' The Information Value (IV) for each bin is calculated as:
// //' \deqn{IV = (\text{Proportion of Events} - \text{Proportion of Non-Events}) \times WOE}
// //'
// //' @references
// //' \itemize{
// //'   \item Saleem, S. M., & Jain, A. K. (2017). A comprehensive review of supervised binning techniques for credit scoring. Journal of Risk Model Validation, 11(3), 1-35.
// //'   \item Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). Credit scoring and its applications. SIAM.
// //' }
// //'
// //' @author Lopes, J. E.
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_swb(target, feature)
// //'
// //' # View results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_swb(Rcpp::IntegerVector target,
//                                           Rcpp::StringVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           std::string bin_separator = "%;%",
//                                           double convergence_threshold = 1e-6,
//                                           int max_iterations = 1000) {
//  try {
//    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//    
//    OptimalBinningCategoricalSWB binner(feature_vec, target_vec, min_bins, max_bins, 
//                                        bin_cutoff, max_n_prebins, bin_separator,
//                                        convergence_threshold, max_iterations);
//    binner.fit();
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
