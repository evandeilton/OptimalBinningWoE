// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
 
using namespace Rcpp;

class OptimalBinningCategoricalMILP {
private:
  struct Bin {
    std::vector<std::string> categories;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<int> target;
  std::vector<std::string> feature;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::vector<Bin> bins;
  int total_pos;
  int total_neg;
  bool converged;
  int iterations_run;
  
public:
  OptimalBinningCategoricalMILP(
    const std::vector<int>& target,
    const std::vector<std::string>& feature,
    int min_bins,
    int max_bins,
    double bin_cutoff,
    int max_n_prebins,
    const std::string& bin_separator,
    double convergence_threshold,
    int max_iterations
  );
  
  Rcpp::List fit();
  
private:
  void validate_input();
  void initialize_bins();
  void merge_bins();
  void calculate_woe_iv(Bin& bin);
  bool is_monotonic() const;
  void handle_zero_counts();
  std::string join_categories(const std::vector<std::string>& categories) const;
  
  inline double safe_log(double value) const {
    // Safe log: avoids log(0) by adding a small epsilon
    const double EPS = 1e-12;
    return std::log((value <= 0.0) ? EPS : value);
  }
};

OptimalBinningCategoricalMILP::OptimalBinningCategoricalMILP(
  const std::vector<int>& target,
  const std::vector<std::string>& feature,
  int min_bins,
  int max_bins,
  double bin_cutoff,
  int max_n_prebins,
  const std::string& bin_separator,
  double convergence_threshold,
  int max_iterations
) : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
convergence_threshold(convergence_threshold), max_iterations(max_iterations),
total_pos(0), total_neg(0), converged(false), iterations_run(0) {}

void OptimalBinningCategoricalMILP::validate_input() {
  if (target.size() != feature.size()) {
    throw std::invalid_argument("Length of target and feature vectors must be the same.");
  }
  if (target.empty() || feature.empty()) {
    throw std::invalid_argument("Target and feature vectors must not be empty.");
  }
  if (min_bins < 2) {
    throw std::invalid_argument("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    throw std::invalid_argument("max_bins must be greater or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
  }
  if (convergence_threshold <= 0.0) {
    throw std::invalid_argument("convergence_threshold must be positive.");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive.");
  }
}

void OptimalBinningCategoricalMILP::handle_zero_counts() {
  if (total_pos == 0 || total_neg == 0) {
    throw std::runtime_error("Target variable must have at least one positive and one negative case.");
  }
}

void OptimalBinningCategoricalMILP::initialize_bins() {
  std::unordered_map<std::string, Bin> bin_map;
  bin_map.reserve(feature.size());
  
  total_pos = 0;
  total_neg = 0;
  
  for (size_t i = 0; i < target.size(); ++i) {
    const std::string& cat = feature[i];
    int tar = target[i];
    
    if (tar != 0 && tar != 1) {
      throw std::invalid_argument("Target variable must be binary (0 or 1).");
    }
    
    if (bin_map.find(cat) == bin_map.end()) {
      bin_map[cat] = Bin{{cat}, 0, 0, 0.0, 0.0};
    }
    
    if (tar == 1) {
      bin_map[cat].count_pos++;
      total_pos++;
    } else {
      bin_map[cat].count_neg++;
      total_neg++;
    }
  }
  
  handle_zero_counts();
  
  bins.reserve(bin_map.size());
  for (auto& kv : bin_map) {
    bins.push_back(std::move(kv.second));
  }
  
  // Calculate initial WoE and IV for each bin
  for (auto& bin : bins) {
    calculate_woe_iv(bin);
  }
}

void OptimalBinningCategoricalMILP::calculate_woe_iv(Bin& bin) {
  const double EPS = 1e-6; // small epsilon to handle zeros
  double dist_pos = static_cast<double>(bin.count_pos) / (total_pos <= 0 ? 1 : total_pos);
  double dist_neg = static_cast<double>(bin.count_neg) / (total_neg <= 0 ? 1 : total_neg);
  
  if (dist_pos < EPS && dist_neg < EPS) {
    // Both are effectively zero
    bin.woe = 0.0;
    bin.iv = 0.0;
  } else if (dist_pos < EPS) {
    // Very low dist_pos, assign large negative WoE
    bin.woe = safe_log(EPS / dist_neg);
    bin.iv = ((0.0 - dist_neg) * bin.woe);
  } else if (dist_neg < EPS) {
    // Very low dist_neg, assign large positive WoE
    bin.woe = safe_log(dist_pos / EPS);
    bin.iv = ((dist_pos - 0.0) * bin.woe);
  } else {
    bin.woe = safe_log(dist_pos) - safe_log(dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
}

bool OptimalBinningCategoricalMILP::is_monotonic() const {
  if (bins.size() <= 2) {
    return true;  // Monotonic if <= 2 bins
  }
  
  bool increasing = true;
  bool decreasing = true;
  
  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i-1].woe) {
      increasing = false;
    }
    if (bins[i].woe > bins[i-1].woe) {
      decreasing = false;
    }
    if (!increasing && !decreasing) {
      return false;
    }
  }
  
  return increasing || decreasing;
}

void OptimalBinningCategoricalMILP::merge_bins() {
  const size_t min_bins_size = static_cast<size_t>(min_bins);
  const size_t max_bins_size = static_cast<size_t>(std::min(max_bins, static_cast<int>(bins.size())));
  const size_t max_n_prebins_size = static_cast<size_t>(std::max(max_n_prebins, min_bins));
  
  // Reduce pre-bins if necessary
  if (bins.size() > max_n_prebins_size) {
    // Sort by IV ascending
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return a.iv < b.iv;
    });
    
    while (bins.size() > max_n_prebins_size && bins.size() > 1) {
      // Merge first two least informative bins
      bins[0].categories.insert(
          bins[0].categories.end(),
          bins[1].categories.begin(),
          bins[1].categories.end()
      );
      bins[0].count_pos += bins[1].count_pos;
      bins[0].count_neg += bins[1].count_neg;
      calculate_woe_iv(bins[0]);
      bins.erase(bins.begin() + 1);
    }
  }
  
  // Iterative merging
  bool merging = true;
  double prev_total_iv = 0.0;
  double total_count = static_cast<double>(total_pos + total_neg);
  
  while (merging && iterations_run < max_iterations) {
    merging = false;
    iterations_run++;
    
    // Merge bins below bin_cutoff if possible
    std::vector<size_t> low_count_bins;
    low_count_bins.reserve(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) {
      double bin_proportion = (bins[i].count_pos + bins[i].count_neg) / (total_count <= 0 ? 1.0 : total_count);
      if (bin_proportion < bin_cutoff && bins.size() > min_bins_size) {
        low_count_bins.push_back(i);
      }
    }
    
    // Merge low count bins
    // Process in reverse order to avoid indexing issues after erase
    for (auto it = low_count_bins.rbegin(); it != low_count_bins.rend(); ++it) {
      size_t idx = *it;
      if (bins.size() <= 1) break; 
      
      if (idx == 0) {
        // Merge with next bin
        if (bins.size() > 1) {
          bins[1].categories.insert(
              bins[1].categories.end(),
              bins[0].categories.begin(),
              bins[0].categories.end()
          );
          bins[1].count_pos += bins[0].count_pos;
          bins[1].count_neg += bins[0].count_neg;
          calculate_woe_iv(bins[1]);
          bins.erase(bins.begin());
          merging = true;
        }
      } else if (idx < bins.size() - 1) {
        // Merge with previous bin
        bins[idx - 1].categories.insert(
            bins[idx - 1].categories.end(),
            bins[idx].categories.begin(),
            bins[idx].categories.end()
        );
        bins[idx - 1].count_pos += bins[idx].count_pos;
        bins[idx - 1].count_neg += bins[idx].count_neg;
        calculate_woe_iv(bins[idx - 1]);
        bins.erase(bins.begin() + idx);
        merging = true;
      } else {
        // idx == bins.size() - 1, merge with previous
        bins[idx - 1].categories.insert(
            bins[idx - 1].categories.end(),
            bins[idx].categories.begin(),
            bins[idx].categories.end()
        );
        bins[idx - 1].count_pos += bins[idx].count_pos;
        bins[idx - 1].count_neg += bins[idx].count_neg;
        calculate_woe_iv(bins[idx - 1]);
        bins.erase(bins.begin() + idx);
        merging = true;
      }
      
      if (bins.size() <= min_bins_size) break;
    }
    
    // If too many bins, merge least informative bins
    if (bins.size() > max_bins_size && bins.size() > 1) {
      std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
        return a.iv < b.iv;
      });
      // Merge first two bins
      bins[1].categories.insert(
          bins[1].categories.end(),
          bins[0].categories.begin(),
          bins[0].categories.end()
      );
      bins[1].count_pos += bins[0].count_pos;
      bins[1].count_neg += bins[0].count_neg;
      calculate_woe_iv(bins[1]);
      bins.erase(bins.begin());
      merging = true;
    }
    
    // Ensure monotonicity if possible
    if (bins.size() > min_bins_size && !is_monotonic() && bins.size() > 1) {
      // Merge the bin with the smallest absolute IV with its neighbor
      double min_iv = std::numeric_limits<double>::max();
      size_t merge_idx = 0;
      for (size_t i = 0; i < bins.size(); ++i) {
        double abs_iv = std::abs(bins[i].iv);
        if (abs_iv < min_iv) {
          min_iv = abs_iv;
          merge_idx = i;
        }
      }
      if (merge_idx == 0 && bins.size() > 1) {
        // Merge with next bin
        bins[1].categories.insert(
            bins[1].categories.end(),
            bins[0].categories.begin(),
            bins[0].categories.end()
        );
        bins[1].count_pos += bins[0].count_pos;
        bins[1].count_neg += bins[0].count_neg;
        calculate_woe_iv(bins[1]);
        bins.erase(bins.begin());
      } else if (merge_idx > 0) {
        // Merge with previous bin
        bins[merge_idx - 1].categories.insert(
            bins[merge_idx - 1].categories.end(),
            bins[merge_idx].categories.begin(),
            bins[merge_idx].categories.end()
        );
        bins[merge_idx - 1].count_pos += bins[merge_idx].count_pos;
        bins[merge_idx - 1].count_neg += bins[merge_idx].count_neg;
        calculate_woe_iv(bins[merge_idx - 1]);
        bins.erase(bins.begin() + merge_idx);
      }
      merging = true;
    }
    
    // Check for convergence
    double total_iv = 0.0;
    for (const auto& bin : bins) {
      total_iv += bin.iv;
    }
    if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
      converged = true;
      break;
    }
    prev_total_iv = total_iv;
  }
}

std::string OptimalBinningCategoricalMILP::join_categories(const std::vector<std::string>& categories) const {
  // Efficient concatenation
  if (categories.empty()) return "";
  size_t total_length = 0;
  for (const auto& c : categories) total_length += c.size() + bin_separator.size();
  total_length = (total_length > bin_separator.size()) ? total_length - bin_separator.size() : total_length;
  
  std::string result;
  result.reserve(total_length);
  for (size_t i = 0; i < categories.size(); ++i) {
    if (i > 0) result += bin_separator;
    result += categories[i];
  }
  return result;
}

Rcpp::List OptimalBinningCategoricalMILP::fit() {
  try {
    validate_input();
    initialize_bins();
    
    // If number of unique categories <= max_bins, no need for optimization
    if (bins.size() <= static_cast<size_t>(max_bins)) {
      converged = true;
      iterations_run = 0;
    } else {
      merge_bins();
    }
    
    // Prepare output
    size_t num_bins = bins.size();
    Rcpp::CharacterVector bin_names(num_bins);
    Rcpp::NumericVector bin_woe(num_bins);
    Rcpp::NumericVector bin_iv(num_bins);
    Rcpp::IntegerVector bin_count(num_bins);
    Rcpp::IntegerVector bin_count_pos(num_bins);
    Rcpp::IntegerVector bin_count_neg(num_bins);
    
    for (size_t i = 0; i < num_bins; ++i) {
      const Bin& bin = bins[i];
      bin_names[i] = join_categories(bin.categories);
      bin_woe[i] = bin.woe;
      bin_iv[i] = bin.iv;
      bin_count[i] = bin.count_pos + bin.count_neg;
      bin_count_pos[i] = bin.count_pos;
      bin_count_neg[i] = bin.count_neg;
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
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
  }
}


//' @title Optimal Binning for Categorical Variables using MILP
//'
//' @description
//' This function performs optimal binning for categorical variables using a Mixed Integer Linear Programming (MILP) inspired approach. It creates optimal bins for a categorical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//' @param bin_separator Separator used to join categories within a bin (default: "%;%").
//' @param convergence_threshold Threshold for convergence of total Information Value (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item bins: Character vector of bin categories.
//'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector of Information Value (IV) for each bin.
//'   \item count: Integer vector of total observations in each bin.
//'   \item count_pos: Integer vector of positive target observations in each bin.
//'   \item count_neg: Integer vector of negative target observations in each bin.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations run.
//' }
//'
//' @details
//' The Optimal Binning algorithm for categorical variables using a MILP-inspired approach works as follows:
//' 1. Validate input and initialize bins for each unique category.
//' 2. If the number of unique categories is less than or equal to max_bins, no optimization is performed.
//' 3. Otherwise, merge bins iteratively based on the following criteria:
//'    a. Merge bins with counts below the bin_cutoff.
//'    b. Ensure the number of bins is between min_bins and max_bins.
//'    c. Attempt to achieve monotonicity in Weight of Evidence (WoE) values.
//' 4. The algorithm stops when convergence is reached or max_iterations is hit.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//'
//' @references
//' \itemize{
//'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- sample(LETTERS[1:10], n, replace = TRUE)
//'
//' # Run optimal binning
//' result <- optimal_binning_categorical_milp(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_milp(
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
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 
 OptimalBinningCategoricalMILP obcm(
     target_vec,
     feature_vec,
     min_bins,
     max_bins,
     bin_cutoff,
     max_n_prebins,
     bin_separator,
     convergence_threshold,
     max_iterations
 );
 
 return obcm.fit();
}



// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// 
// // Enable C++11 features
// // [[Rcpp::plugins(cpp11)]]
// 
// class OptimalBinningCategoricalMILP {
// private:
//   std::vector<int> target;
//   std::vector<std::string> feature;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   std::string bin_separator;
//   double convergence_threshold;
//   int max_iterations;
//   
//   struct Bin {
//     std::vector<std::string> categories;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> bins;
//   int total_pos;
//   int total_neg;
//   bool converged;
//   int iterations_run;
//   
// public:
//   OptimalBinningCategoricalMILP(
//     const std::vector<int>& target,
//     const std::vector<std::string>& feature,
//     int min_bins,
//     int max_bins,
//     double bin_cutoff,
//     int max_n_prebins,
//     const std::string& bin_separator,
//     double convergence_threshold,
//     int max_iterations
//   );
//   
//   Rcpp::List fit();
//   
// private:
//   void validate_input();
//   void initialize_bins();
//   void merge_bins();
//   void calculate_woe_iv(Bin& bin);
//   bool is_monotonic();
//   void handle_zero_counts();
//   std::string join_categories(const std::vector<std::string>& categories);
// };
// 
// OptimalBinningCategoricalMILP::OptimalBinningCategoricalMILP(
//   const std::vector<int>& target,
//   const std::vector<std::string>& feature,
//   int min_bins,
//   int max_bins,
//   double bin_cutoff,
//   int max_n_prebins,
//   const std::string& bin_separator,
//   double convergence_threshold,
//   int max_iterations
// ) : target(target), feature(feature), min_bins(min_bins), max_bins(max_bins),
// bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins), bin_separator(bin_separator),
// convergence_threshold(convergence_threshold), max_iterations(max_iterations),
// total_pos(0), total_neg(0), converged(false), iterations_run(0) {}
// 
// void OptimalBinningCategoricalMILP::validate_input() {
//   if (target.size() != feature.size()) {
//     throw std::invalid_argument("Length of target and feature vectors must be the same.");
//   }
//   if (target.empty() || feature.empty()) {
//     throw std::invalid_argument("Target and feature vectors must not be empty.");
//   }
//   if (min_bins < 2) {
//     throw std::invalid_argument("min_bins must be at least 2.");
//   }
//   if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
//     throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//   }
//   if (convergence_threshold <= 0.0) {
//     throw std::invalid_argument("convergence_threshold must be positive.");
//   }
//   if (max_iterations <= 0) {
//     throw std::invalid_argument("max_iterations must be positive.");
//   }
// }
// 
// void OptimalBinningCategoricalMILP::handle_zero_counts() {
//   if (total_pos == 0 || total_neg == 0) {
//     throw std::runtime_error("Target variable must have at least one positive and one negative case.");
//   }
// }
// 
// void OptimalBinningCategoricalMILP::initialize_bins() {
//   std::unordered_map<std::string, Bin> bin_map;
//   total_pos = 0;
//   total_neg = 0;
//   
//   for (size_t i = 0; i < target.size(); ++i) {
//     const std::string& cat = feature[i];
//     int tar = target[i];
//     
//     if (tar != 0 && tar != 1) {
//       throw std::invalid_argument("Target variable must be binary (0 or 1).");
//     }
//     
//     if (bin_map.find(cat) == bin_map.end()) {
//       bin_map[cat] = Bin{std::vector<std::string>{cat}, 0, 0, 0.0, 0.0};
//     }
//     
//     if (tar == 1) {
//       bin_map[cat].count_pos++;
//       total_pos++;
//     } else {
//       bin_map[cat].count_neg++;
//       total_neg++;
//     }
//   }
//   
//   handle_zero_counts();
//   
//   // Convert map to vector
//   bins.reserve(bin_map.size());
//   for (auto& kv : bin_map) {
//     bins.push_back(kv.second);
//   }
//   
//   // Calculate initial WoE and IV
//   for (auto& bin : bins) {
//     calculate_woe_iv(bin);
//   }
// }
// 
// void OptimalBinningCategoricalMILP::calculate_woe_iv(Bin& bin) {
//   double dist_pos = static_cast<double>(bin.count_pos) / total_pos;
//   double dist_neg = static_cast<double>(bin.count_neg) / total_neg;
//   
//   // Handle zero distributions to maintain numerical stability
//   if (dist_pos == 0.0 && dist_neg == 0.0) {
//     bin.woe = 0.0;
//     bin.iv = 0.0;
//   }
//   else if (dist_pos == 0.0) {
//     bin.woe = std::log(0.0001 / dist_neg); // Assign a large negative WoE
//     bin.iv = (0.0 - dist_neg) * bin.woe;
//   }
//   else if (dist_neg == 0.0) {
//     bin.woe = std::log(dist_pos / 0.0001); // Assign a large positive WoE
//     bin.iv = (dist_pos - 0.0) * bin.woe;
//   }
//   else {
//     bin.woe = std::log(dist_pos / dist_neg);
//     bin.iv = (dist_pos - dist_neg) * bin.woe;
//   }
// }
// 
// bool OptimalBinningCategoricalMILP::is_monotonic() {
//   if (bins.size() <= 2) {
//     return true;  // Always consider monotonic for 2 or fewer bins
//   }
//   
//   bool increasing = true;
//   bool decreasing = true;
//   
//   for (size_t i = 1; i < bins.size(); ++i) {
//     if (bins[i].woe < bins[i-1].woe) {
//       increasing = false;
//     }
//     if (bins[i].woe > bins[i-1].woe) {
//       decreasing = false;
//     }
//   }
//   
//   return increasing || decreasing;
// }
// 
// void OptimalBinningCategoricalMILP::merge_bins() {
//   // Ensure max_n_prebins is not less than min_bins
//   size_t max_n_prebins_size = static_cast<size_t>(std::max(max_n_prebins, min_bins));
//   size_t min_bins_size = static_cast<size_t>(min_bins);
//   size_t max_bins_size = static_cast<size_t>(std::min(max_bins, static_cast<int>(bins.size())));
//   
//   // Limit the initial number of pre-bins
//   if (bins.size() > max_n_prebins_size) {
//     // Sort bins by IV in ascending order to merge the least informative first
//     std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
//       return a.iv < b.iv;
//     });
//     
//     while (bins.size() > max_n_prebins_size) {
//       // Merge the two bins with the smallest IV
//       if (bins.size() < 2) break;
//       
//       size_t merge_idx1 = 0;
//       size_t merge_idx2 = 1;
//       
//       // Merge bins[0] and bins[1]
//       bins[merge_idx1].categories.insert(
//           bins[merge_idx1].categories.end(),
//           bins[merge_idx2].categories.begin(),
//           bins[merge_idx2].categories.end()
//       );
//       bins[merge_idx1].count_pos += bins[merge_idx2].count_pos;
//       bins[merge_idx1].count_neg += bins[merge_idx2].count_neg;
//       calculate_woe_iv(bins[merge_idx1]);
//       
//       bins.erase(bins.begin() + merge_idx2);
//     }
//   }
//   
//   // Further merge to satisfy max_bins and bin_cutoff constraints
//   bool merging = true;
//   double prev_total_iv = 0.0;
//   while (merging && iterations_run < max_iterations) {
//     merging = false;
//     iterations_run++;
//     
//     // Check for bins below bin_cutoff
//     std::vector<size_t> low_count_bins;
//     for (size_t i = 0; i < bins.size(); ++i) {
//       double bin_proportion = static_cast<double>(bins[i].count_pos + bins[i].count_neg) / (total_pos + total_neg);
//       if (bin_proportion < bin_cutoff && bins.size() > min_bins_size) {
//         low_count_bins.push_back(i);
//       }
//     }
//     
//     for (auto it = low_count_bins.rbegin(); it != low_count_bins.rend(); ++it) {
//       size_t idx = *it;
//       if (idx == 0 && bins.size() > 1) {
//         // Merge with next bin
//         bins[1].categories.insert(
//             bins[1].categories.end(),
//             bins[0].categories.begin(),
//             bins[0].categories.end()
//         );
//         bins[1].count_pos += bins[0].count_pos;
//         bins[1].count_neg += bins[0].count_neg;
//         calculate_woe_iv(bins[1]);
//         bins.erase(bins.begin());
//         merging = true;
//       }
//       else if (idx < bins.size() - 1 && bins.size() > 1) {
//         // Merge with previous bin
//         bins[idx - 1].categories.insert(
//             bins[idx - 1].categories.end(),
//             bins[idx].categories.begin(),
//             bins[idx].categories.end()
//         );
//         bins[idx - 1].count_pos += bins[idx].count_pos;
//         bins[idx - 1].count_neg += bins[idx].count_neg;
//         calculate_woe_iv(bins[idx - 1]);
//         bins.erase(bins.begin() + idx);
//         merging = true;
//       }
//       else if (bins.size() > 1) {
//         // Merge with previous bin
//         bins[idx - 1].categories.insert(
//             bins[idx - 1].categories.end(),
//             bins[idx].categories.begin(),
//             bins[idx].categories.end()
//         );
//         bins[idx - 1].count_pos += bins[idx].count_pos;
//         bins[idx - 1].count_neg += bins[idx].count_neg;
//         calculate_woe_iv(bins[idx - 1]);
//         bins.erase(bins.begin() + idx);
//         merging = true;
//       }
//       if (bins.size() <= min_bins_size) break;
//     }
//     
//     if (bins.size() > max_bins_size) {
//       // Merge bins with the smallest IV to reduce the number of bins
//       std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
//         return a.iv < b.iv;
//       });
//       if (bins.size() >= 2) {
//         // Merge the two least informative bins
//         bins[1].categories.insert(
//             bins[1].categories.end(),
//             bins[0].categories.begin(),
//             bins[0].categories.end()
//         );
//         bins[1].count_pos += bins[0].count_pos;
//         bins[1].count_neg += bins[0].count_neg;
//         calculate_woe_iv(bins[1]);
//         bins.erase(bins.begin());
//         merging = true;
//       }
//     }
//     
//     // Ensure monotonicity if possible
//     if (bins.size() > min_bins_size && !is_monotonic()) {
//       // Find the bin with the smallest absolute IV and merge it with its neighbor
//       double min_iv = std::numeric_limits<double>::max();
//       size_t merge_idx = 0;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (std::abs(bins[i].iv) < min_iv) {
//           min_iv = std::abs(bins[i].iv);
//           merge_idx = i;
//         }
//       }
//       if (merge_idx == 0 && bins.size() > 1) {
//         // Merge with next bin
//         bins[1].categories.insert(
//             bins[1].categories.end(),
//             bins[0].categories.begin(),
//             bins[0].categories.end()
//         );
//         bins[1].count_pos += bins[0].count_pos;
//         bins[1].count_neg += bins[0].count_neg;
//         calculate_woe_iv(bins[1]);
//         bins.erase(bins.begin());
//       }
//       else if (merge_idx > 0) {
//         // Merge with previous bin
//         bins[merge_idx - 1].categories.insert(
//             bins[merge_idx - 1].categories.end(),
//             bins[merge_idx].categories.begin(),
//             bins[merge_idx].categories.end()
//         );
//         bins[merge_idx - 1].count_pos += bins[merge_idx].count_pos;
//         bins[merge_idx - 1].count_neg += bins[merge_idx].count_neg;
//         calculate_woe_iv(bins[merge_idx - 1]);
//         bins.erase(bins.begin() + merge_idx);
//       }
//       merging = true;
//     }
//     
//     // Check for convergence
//     double total_iv = 0.0;
//     for (const auto& bin : bins) {
//       total_iv += bin.iv;
//     }
//     if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
//       converged = true;
//       break;
//     }
//     prev_total_iv = total_iv;
//   }
// }
// 
// std::string OptimalBinningCategoricalMILP::join_categories(const std::vector<std::string>& categories) {
//   std::string result;
//   for (size_t i = 0; i < categories.size(); ++i) {
//     if (i > 0) result += bin_separator;
//     result += categories[i];
//   }
//   return result;
// }
// 
// Rcpp::List OptimalBinningCategoricalMILP::fit() {
//   try {
//     validate_input();
//     initialize_bins();
//     
//     // If number of unique categories is less than or equal to max_bins, no need to optimize
//     if (bins.size() <= static_cast<size_t>(max_bins)) {
//       converged = true;
//       iterations_run = 0;
//     } else {
//       merge_bins();
//     }
//     
//     // Prepare output
//     size_t num_bins = bins.size();
//     Rcpp::CharacterVector bin_names(num_bins);
//     Rcpp::NumericVector bin_woe(num_bins);
//     Rcpp::NumericVector bin_iv(num_bins);
//     Rcpp::IntegerVector bin_count(num_bins);
//     Rcpp::IntegerVector bin_count_pos(num_bins);
//     Rcpp::IntegerVector bin_count_neg(num_bins);
//     
//     for (size_t i = 0; i < num_bins; ++i) {
//       const Bin& bin = bins[i];
//       bin_names[i] = join_categories(bin.categories);
//       bin_woe[i] = bin.woe;
//       bin_iv[i] = bin.iv;
//       bin_count[i] = bin.count_pos + bin.count_neg;
//       bin_count_pos[i] = bin.count_pos;
//       bin_count_neg[i] = bin.count_neg;
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("bin") = bin_names,
//       Rcpp::Named("woe") = bin_woe,
//       Rcpp::Named("iv") = bin_iv,
//       Rcpp::Named("count") = bin_count,
//       Rcpp::Named("count_pos") = bin_count_pos,
//       Rcpp::Named("count_neg") = bin_count_neg,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   } catch (const std::exception& e) {
//     Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//   }
// }
// 
// //' @title Optimal Binning for Categorical Variables using MILP
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Mixed Integer Linear Programming (MILP) inspired approach. It creates optimal bins for a categorical feature based on its relationship with a binary target variable, maximizing the predictive power while respecting user-defined constraints.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
// //' @param bin_separator Separator used to join categories within a bin (default: "%;%").
// //' @param convergence_threshold Threshold for convergence of total Information Value (default: 1e-6).
// //' @param max_iterations Maximum number of iterations for the optimization process (default: 1000).
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item bins: Character vector of bin categories.
// //'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
// //'   \item iv: Numeric vector of Information Value (IV) for each bin.
// //'   \item count: Integer vector of total observations in each bin.
// //'   \item count_pos: Integer vector of positive target observations in each bin.
// //'   \item count_neg: Integer vector of negative target observations in each bin.
// //'   \item converged: Logical indicating whether the algorithm converged.
// //'   \item iterations: Integer indicating the number of iterations run.
// //' }
// //'
// //' @details
// //' The Optimal Binning algorithm for categorical variables using a MILP-inspired approach works as follows:
// //' 1. Validate input and initialize bins for each unique category.
// //' 2. If the number of unique categories is less than or equal to max_bins, no optimization is performed.
// //' 3. Otherwise, merge bins iteratively based on the following criteria:
// //'    a. Merge bins with counts below the bin_cutoff.
// //'    b. Ensure the number of bins is between min_bins and max_bins.
// //'    c. Attempt to achieve monotonicity in Weight of Evidence (WoE) values.
// //' 4. The algorithm stops when convergence is reached or max_iterations is hit.
// //'
// //' Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
// //'
// //' Information Value (IV) is calculated as:
// //' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
// //'
// //' @references
// //' \itemize{
// //'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
// //'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' # Create sample data
// //' set.seed(123)
// //' n <- 1000
// //' target <- sample(0:1, n, replace = TRUE)
// //' feature <- sample(LETTERS[1:10], n, replace = TRUE)
// //'
// //' # Run optimal binning
// //' result <- optimal_binning_categorical_milp(target, feature, min_bins = 2, max_bins = 4)
// //'
// //' # Print results
// //' print(result)
// //' }
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_milp(
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
//  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//  std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//  
//  // Instantiate the binning class
//  OptimalBinningCategoricalMILP obcm(
//      target_vec,
//      feature_vec,
//      min_bins,
//      max_bins,
//      bin_cutoff,
//      max_n_prebins,
//      bin_separator,
//      convergence_threshold,
//      max_iterations
//  );
//  
//  // Perform the binning and return the results
//  return obcm.fit();
// }
