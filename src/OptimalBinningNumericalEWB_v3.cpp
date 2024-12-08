// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <unordered_set>
#include <numeric>

using namespace Rcpp;

// Class for Optimal Binning using Equal-Width Binning
class OptimalBinningNumericalEWB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  bool converged;
  int iterations_run;
  
  struct Bin {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    
    Bin(double lb = -std::numeric_limits<double>::infinity(),
        double ub = std::numeric_limits<double>::infinity(),
        int c = 0, int cp = 0, int cn = 0)
      : lower(lb), upper(ub), count(c), count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
  };
  
  std::vector<Bin> bins;
  
  // Total positives and negatives
  int total_pos;
  int total_neg;
  
  // Number of unique feature values
  int unique_count;
  
  static constexpr double EPSILON = 1e-10;
  
  std::string double_to_string(double value) const {
    if (std::isinf(value)) {
      return value > 0 ? "+Inf" : "-Inf";
    }
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }
  
  double calculate_woe(int pos, int neg) const {
    double pos_rate = (pos > 0) ? (static_cast<double>(pos) / total_pos) : (EPSILON / (total_pos + EPSILON));
    double neg_rate = (neg > 0) ? (static_cast<double>(neg) / total_neg) : (EPSILON / (total_neg + EPSILON));
    // Avoid division by zero or log of zero by adding EPSILON
    pos_rate = std::max(pos_rate, EPSILON);
    neg_rate = std::max(neg_rate, EPSILON);
    return std::log(pos_rate / neg_rate);
  }
  
  double calculate_iv(double woe, int pos, int neg) const {
    double p_rate = (total_pos > 0) ? (static_cast<double>(pos) / total_pos) : EPSILON;
    double n_rate = (total_neg > 0) ? (static_cast<double>(neg) / total_neg) : EPSILON;
    return (p_rate - n_rate) * woe;
  }
  
  void validate_inputs() {
    if (feature.empty()) {
      Rcpp::stop("Feature vector is empty.");
    }
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target vectors must have the same length.");
    }
    
    for (const auto& t : target) {
      if (t != 0 && t != 1) {
        Rcpp::stop("Target vector must contain only 0 and 1.");
      }
    }
    
    std::unordered_set<double> unique_values(feature.begin(), feature.end());
    unique_count = static_cast<int>(unique_values.size());
    
    if (unique_count <= 1) {
      Rcpp::stop("Feature vector must contain at least two unique values.");
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
    if (max_n_prebins <= 0) {
      Rcpp::stop("max_n_prebins must be positive.");
    }
    if (max_iterations <= 0) {
      Rcpp::stop("max_iterations must be positive.");
    }
  }
  
  void create_unique_bins() {
    bins.clear();
    bins.reserve(static_cast<size_t>(unique_count));
    
    std::vector<double> unique_values(feature);
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    for (const auto& val : unique_values) {
      bins.emplace_back(val, val, 0, 0, 0);
    }
    
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int target_value = target[i];
      // Exact bin match since lower=upper for unique bins
      for (auto& bin : bins) {
        if (std::fabs(value - bin.lower) < EPSILON) {
          bin.count++;
          if (target_value == 1) {
            bin.count_pos++;
          } else {
            bin.count_neg++;
          }
          break;
        }
      }
    }
  }
  
  void create_prebins() {
    double min_value = *std::min_element(feature.begin(), feature.end());
    double max_value = *std::max_element(feature.begin(), feature.end());
    
    if (std::fabs(min_value - max_value) < EPSILON) {
      // All values are approximately the same, single bin
      int total_c = static_cast<int>(feature.size());
      int pos_c = std::count(target.begin(), target.end(), 1);
      int neg_c = total_c - pos_c;
      bins.emplace_back(min_value, max_value, total_c, pos_c, neg_c);
      return;
    }
    
    int n_prebins = std::min(max_n_prebins, static_cast<int>(feature.size()));
    // Avoid division by zero
    double range = max_value - min_value;
    if (range < EPSILON) {
      range = EPSILON;
    }
    double bin_width = range / n_prebins;
    
    bins.clear();
    bins.reserve(static_cast<size_t>(n_prebins));
    
    for (int i = 0; i < n_prebins; ++i) {
      double lower = min_value + i * bin_width;
      double upper = (i == n_prebins - 1) ? max_value : (min_value + (i + 1) * bin_width);
      // Ensure upper >= lower
      if (upper < lower) {
        upper = lower;
      }
      bins.emplace_back(lower, upper, 0, 0, 0);
    }
  }
  
  void assign_data_to_bins() {
    // Assign each observation to a bin
    for (size_t i = 0; i < feature.size(); ++i) {
      double value = feature[i];
      int target_value = target[i];
      int bin_index = find_bin(value);
      if (bin_index < 0) {
        // Should not happen if bins cover the range correctly
        // Fall back to closest bin if needed
        if (value < bins.front().lower) {
          bin_index = 0;
        } else if (value > bins.back().upper) {
          bin_index = static_cast<int>(bins.size()) - 1;
        } else {
          Rcpp::stop("Value does not fit into any bin.");
        }
      }
      bins[static_cast<size_t>(bin_index)].count++;
      if (target_value == 1) {
        bins[static_cast<size_t>(bin_index)].count_pos++;
      } else {
        bins[static_cast<size_t>(bin_index)].count_neg++;
      }
    }
  }
  
  int find_bin(double value) const {
    // Since bins are in ascending order, we can find via linear search or binary search.
    // Here linear is used. For large data, binary search could be considered.
    for (size_t b = 0; b < bins.size(); ++b) {
      if (value >= bins[b].lower && value <= bins[b].upper) {
        return static_cast<int>(b);
      }
    }
    return -1;
  }
  
  void merge_rare_bins() {
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const Bin& bin) { return sum + bin.count; });
    double cutoff_count = bin_cutoff * total_count;
    int iterations = 0;
    
    while (iterations < max_iterations) {
      bool merged = false;
      // Avoid merging below min_bins
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }
      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
          if (i == 0 && bins.size() > 1) {
            // Merge with next bin
            bins[0].upper = bins[1].upper;
            bins[0].count += bins[1].count;
            bins[0].count_pos += bins[1].count_pos;
            bins[0].count_neg += bins[1].count_neg;
            bins.erase(bins.begin() + 1);
          } else if (i > 0) {
            // Merge with previous bin
            bins[i - 1].upper = bins[i].upper;
            bins[i - 1].count += bins[i].count;
            bins[i - 1].count_pos += bins[i].count_pos;
            bins[i - 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
          }
          merged = true;
          break;
        }
      }
      iterations++;
      if (!merged) {
        break;
      }
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  void ensure_max_bins() {
    int iterations = 0;
    // Merge until bins.size() <= max_bins
    while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
      // Find pair of adjacent bins with smallest combined count
      if (bins.size() <= 1) break;
      size_t min_index = 0;
      int min_count = bins[0].count + bins[1].count;
      for (size_t i = 1; i < bins.size() - 1; ++i) {
        int combined_count = bins[i].count + bins[i + 1].count;
        if (combined_count < min_count) {
          min_count = combined_count;
          min_index = i;
        }
      }
      // Merge bins[min_index] and bins[min_index + 1]
      if (min_index < bins.size() - 1) {
        bins[min_index].upper = bins[min_index + 1].upper;
        bins[min_index].count += bins[min_index + 1].count;
        bins[min_index].count_pos += bins[min_index + 1].count_pos;
        bins[min_index].count_neg += bins[min_index + 1].count_neg;
        bins.erase(bins.begin() + (min_index + 1));
      }
      iterations++;
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  void enforce_monotonicity() {
    if (bins.size() <= 2) {
      return;
    }
    
    int iterations = 0;
    bool is_monotonic = false;
    bool increasing = true;
    if (bins.size() >= 2) {
      increasing = (bins[1].woe >= bins[0].woe);
    }
    
    while (!is_monotonic && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
      is_monotonic = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((increasing && bins[i].woe < bins[i - 1].woe) ||
            (!increasing && bins[i].woe > bins[i - 1].woe)) {
          // Merge bins[i - 1] and bins[i]
          if (i < bins.size()) {
            bins[i - 1].upper = bins[i].upper;
            bins[i - 1].count += bins[i].count;
            bins[i - 1].count_pos += bins[i].count_pos;
            bins[i - 1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
            calculate_woe_iv();
            is_monotonic = false;
            break;
          }
        }
      }
      iterations++;
      if (static_cast<int>(bins.size()) == min_bins) {
        break;
      }
    }
    iterations_run += iterations;
    if (iterations >= max_iterations) {
      converged = false;
    }
  }
  
  void calculate_woe_iv() {
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg);
    }
  }
  
public:
  OptimalBinningNumericalEWB(const std::vector<double>& feature_, const std::vector<int>& target_,
                             int min_bins_ = 3, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20,
                             double convergence_threshold_ = 1e-6, int max_iterations_ = 1000)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      converged(true), iterations_run(0), total_pos(0), total_neg(0), unique_count(0) {}
  
  void fit() {
    validate_inputs();
    
    // Compute totals
    total_pos = std::accumulate(target.begin(), target.end(), 0);
    total_neg = static_cast<int>(target.size()) - total_pos;
    
    if (total_pos == 0 || total_neg == 0) {
      Rcpp::stop("Target must contain at least one positive and one negative case.");
    }
    
    if (unique_count <= 2) {
      // If 2 or fewer unique values, simple bin creation
      create_unique_bins();
      calculate_woe_iv();
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Create and assign to prebins
    create_prebins();
    assign_data_to_bins();
    
    // Merge rare bins
    merge_rare_bins();
    
    // Calculate initial WoE/IV
    calculate_woe_iv();
    
    // Enforce monotonicity
    enforce_monotonicity();
    
    // Ensure max_bins
    ensure_max_bins();
    
    // Recalculate WoE/IV
    calculate_woe_iv();
  }
  
  List get_results() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> counts;
    std::vector<int> counts_pos;
    std::vector<int> counts_neg;
    std::vector<double> cutpoints;
    
    bin_labels.reserve(bins.size());
    woe_values.reserve(bins.size());
    iv_values.reserve(bins.size());
    counts.reserve(bins.size());
    counts_pos.reserve(bins.size());
    counts_neg.reserve(bins.size());
    cutpoints.reserve(bins.size() > 0 ? bins.size() - 1 : 0);
    
    for (size_t i = 0; i < bins.size(); ++i) {
      const auto& bin = bins[i];
      std::string label = "(" + double_to_string(bin.lower) + ";" + double_to_string(bin.upper) + "]";
      bin_labels.push_back(label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      // Cutpoints exclude last bin upper
      if (i < bins.size() - 1) {
        cutpoints.push_back(bin.upper);
      }
    }
    
    return List::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_values,
      Named("iv") = iv_values,
      Named("count") = counts,
      Named("count_pos") = counts_pos,
      Named("count_neg") = counts_neg,
      Named("cutpoints") = cutpoints,
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};


//' @title Optimal Binning for Numerical Variables using Equal-Width Binning
//'
//' @description
//' Performs optimal binning for numerical variables using equal-width intervals (Equal-Width Binning) 
//' with subsequent merging and adjustment steps. This procedure aims to create an interpretable binning 
//' strategy with good predictive power, taking into account monotonicity and minimum splits within the bins.
//'
//' @param target Integer binary vector (0 or 1) representing the target variable.
//' @param feature Numeric vector with the values of the feature to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of observations each bin must contain (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before optimization (default: 20).
//' @param convergence_threshold Convergence threshold (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed (default: 1000).
//'
//' @return A list containing:
//' \item{bins}{Character vector with the interval of each bin.}
//' \item{woe}{Numeric vector with the WoE values for each bin.}
//' \item{iv}{Numeric vector with the IV value for each bin.}
//' \item{count}{Numeric vector with the total number of observations in each bin.}
//' \item{count_pos}{Numeric vector with the total number of positive observations in each bin.}
//' \item{count_neg}{Numeric vector with the total number of negative observations in each bin.}
//' \item{cutpoints}{Numeric vector with the cut points.}
//' \item{converged}{Logical value indicating whether the algorithm converged.}
//' \item{iterations}{Number of iterations performed by the algorithm.}
//'
//' @details
//' The algorithm consists of the following steps:
//' 1. Creation of equal-width pre-bins.
//' 2. Assignment of data to these pre-bins.
//' 3. Merging of rare bins (with few observations).
//' 4. Calculation of initial WoE and IV.
//' 5. Ensuring WoE monotonicity by merging non-monotonic bins.
//' 6. Adjustment to ensure the maximum number of bins does not exceed max_bins.
//' 7. Recalculating WoE and IV at the end.
//'
//' This method aims to provide bins that balance interpretability, monotonicity, and predictive power, useful in risk modeling and credit scoring.
//'
//' @examples
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_ewb(target, feature)
//' print(result)
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ewb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                         int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
                                         int max_n_prebins = 20,
                                         double convergence_threshold = 1e-6, int max_iterations = 1000) {
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
  
  OptimalBinningNumericalEWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
                                    convergence_threshold, max_iterations);
  binner.fit();
  return binner.get_results();
}

/*
 Melhorias Implementadas:
 - Adicionada verificação robusta contra loops infinitos, limitando iterações em cada etapa.
 - Adicionada proteção para casos de taxas zero, usando EPSILON para evitar log(0).
 - Verificação de índices antes de mesclar bins para evitar acessos fora do limite do vetor.
 - Garantia de que sempre que bins forem mesclados, WOE e IV sejam recalculados.
 - Manutenção dos nomes e estruturas de entrada/saída.
 - Comentários em inglês e lógica interna mais robusta e estável.
 - Ajustes na lógica de busca de bins e merges, evitando travamentos ou situações inesperadas.
*/



// #include <Rcpp.h>
// #include <vector>
// #include <algorithm>
// #include <cmath>
// #include <limits>
// #include <sstream>
// #include <iomanip>
// #include <unordered_set>
// #include <numeric>
// 
// using namespace Rcpp;
// 
// // Class for Optimal Binning using Equal-Width Binning
// class OptimalBinningNumericalEWB {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   double convergence_threshold;
//   int max_iterations;
//   
//   bool converged;
//   int iterations_run;
//   
//   struct Bin {
//     double lower;
//     double upper;
//     int count;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//     
//     Bin(double lb = -std::numeric_limits<double>::infinity(),
//         double ub = std::numeric_limits<double>::infinity(),
//         int c = 0, int cp = 0, int cn = 0)
//       : lower(lb), upper(ub), count(c), count_pos(cp), count_neg(cn), woe(0.0), iv(0.0) {}
//   };
//   
//   std::vector<Bin> bins;
//   
//   // Total positives and negatives in the dataset
//   int total_pos;
//   int total_neg;
//   
//   // Number of unique feature values
//   int unique_count;
//   
//   // Helper function to convert double to string with proper formatting
//   std::string double_to_string(double value) const {
//     if (std::isinf(value)) {
//       return value > 0 ? "+Inf" : "-Inf";
//     }
//     std::ostringstream ss;
//     ss << std::fixed << std::setprecision(6) << value;
//     return ss.str();
//   }
//   
//   // Calculate Weight of Evidence (WoE)
//   double calculate_woe(int pos, int neg) const {
//     const double EPSILON = 1e-10;
//     double pos_rate = (pos > 0) ? static_cast<double>(pos) / total_pos : EPSILON / total_pos;
//     double neg_rate = (neg > 0) ? static_cast<double>(neg) / total_neg : EPSILON / total_neg;
//     return std::log(pos_rate / neg_rate);
//   }
//   
//   // Calculate Information Value (IV)
//   double calculate_iv(double woe, int pos, int neg) const {
//     double pos_rate = static_cast<double>(pos) / total_pos;
//     double neg_rate = static_cast<double>(neg) / total_neg;
//     return (pos_rate - neg_rate) * woe;
//   }
//   
//   // Validate input parameters and data
//   void validate_inputs() {
//     if (feature.empty()) {
//       Rcpp::stop("Feature vector is empty.");
//     }
//     if (feature.size() != target.size()) {
//       Rcpp::stop("Feature and target vectors must have the same length.");
//     }
//     
//     // Check target values
//     for (const auto& t : target) {
//       if (t != 0 && t != 1) {
//         Rcpp::stop("Target vector must contain only 0 and 1.");
//       }
//     }
//     
//     // Determine unique feature values
//     std::unordered_set<double> unique_values(feature.begin(), feature.end());
//     unique_count = unique_values.size();
//     
//     if (unique_count <= 1) {
//       Rcpp::stop("Feature vector must contain at least two unique values.");
//     }
//     
//     if (min_bins < 2) {
//       Rcpp::stop("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       Rcpp::stop("max_bins must be greater than or equal to min_bins.");
//     }
//     if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//       Rcpp::stop("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins <= 0) {
//       Rcpp::stop("max_n_prebins must be positive.");
//     }
//     if (max_iterations <= 0) {
//       Rcpp::stop("max_iterations must be positive.");
//     }
//   }
//   
//   // Create bins based on unique values (for unique_count <= 2)
//   void create_unique_bins() {
//     bins.clear();
//     bins.reserve(unique_count);
//     
//     // Collect unique values and sort them
//     std::vector<double> unique_values(feature);
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     for (const auto& val : unique_values) {
//       bins.emplace_back(val, val, 0, 0, 0);
//     }
//     
//     // Assign data to bins
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
//       // Find the bin with exact match
//       for (auto& bin : bins) {
//         if (value == bin.lower) { // Since lower == upper for unique bins
//           bin.count++;
//           if (target_value == 1) {
//             bin.count_pos++;
//           } else {
//             bin.count_neg++;
//           }
//           break;
//         }
//       }
//     }
//   }
//   
//   // Create initial equal-width pre-bins
//   void create_prebins() {
//     double min_value = *std::min_element(feature.begin(), feature.end());
//     double max_value = *std::max_element(feature.begin(), feature.end());
//     
//     // Handle case where all feature values are the same
//     if (min_value == max_value) {
//       bins.emplace_back(min_value, max_value, feature.size(),
//                         std::count(target.begin(), target.end(), 1),
//                         std::count(target.begin(), target.end(), 0));
//       return;
//     }
//     
//     int n_prebins = std::min(max_n_prebins, static_cast<int>(feature.size()));
//     double bin_width = (max_value - min_value) / n_prebins;
//     
//     bins.clear();
//     bins.reserve(n_prebins);
//     
//     for (int i = 0; i < n_prebins; ++i) {
//       double lower = min_value + i * bin_width;
//       double upper = (i == n_prebins - 1) ? max_value : min_value + (i + 1) * bin_width;
//       bins.emplace_back(lower, upper, 0, 0, 0);
//     }
//   }
//   
//   // Assign data to bins
//   void assign_data_to_bins() {
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double value = feature[i];
//       int target_value = target[i];
//       int bin_index = -1;
//       
//       // Find the appropriate bin
//       for (size_t b = 0; b < bins.size(); ++b) {
//         if (value >= bins[b].lower && value <= bins[b].upper) {
//           bin_index = b;
//           break;
//         }
//       }
//       if (bin_index == -1) {
//         if (value < bins.front().lower) {
//           bin_index = 0;
//         } else if (value > bins.back().upper) {
//           bin_index = bins.size() - 1;
//         } else {
//           Rcpp::stop("Value does not fit into any bin.");
//         }
//       }
//       bins[bin_index].count++;
//       if (target_value == 1) {
//         bins[bin_index].count_pos++;
//       } else {
//         bins[bin_index].count_neg++;
//       }
//     }
//   }
//   
//   // Merge bins that are below the bin_cutoff threshold
//   void merge_rare_bins() {
//     int total_count = std::accumulate(bins.begin(), bins.end(), 0,
//                                       [](int sum, const Bin& bin) { return sum + bin.count; });
//     double cutoff_count = bin_cutoff * total_count;
//     int iterations = 0;
//     converged = true;
//     
//     while (iterations < max_iterations) {
//       bool merged = false;
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
//           if (i == 0) {
//             // Merge with next bin
//             bins[i].upper = bins[i + 1].upper;
//             bins[i].count += bins[i + 1].count;
//             bins[i].count_pos += bins[i + 1].count_pos;
//             bins[i].count_neg += bins[i + 1].count_neg;
//             bins.erase(bins.begin() + i + 1);
//           } else {
//             // Merge with previous bin
//             bins[i - 1].upper = bins[i].upper;
//             bins[i - 1].count += bins[i].count;
//             bins[i - 1].count_pos += bins[i].count_pos;
//             bins[i - 1].count_neg += bins[i].count_neg;
//             bins.erase(bins.begin() + i);
//           }
//           merged = true;
//           break;
//         }
//       }
//       iterations++;
//       if (!merged) {
//         break;
//       }
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   // Ensure the number of bins does not exceed max_bins by merging the smallest bins
//   void ensure_max_bins() {
//     int iterations = 0;
//     while (static_cast<int>(bins.size()) > max_bins && iterations < max_iterations) {
//       // Find the pair of adjacent bins with the smallest count
//       size_t min_index = 0;
//       int min_count = bins[0].count + bins[1].count;
//       for (size_t i = 1; i < bins.size() - 1; ++i) {
//         int combined_count = bins[i].count + bins[i + 1].count;
//         if (combined_count < min_count) {
//           min_count = combined_count;
//           min_index = i;
//         }
//       }
//       // Merge bins[min_index] and bins[min_index + 1]
//       bins[min_index].upper = bins[min_index + 1].upper;
//       bins[min_index].count += bins[min_index + 1].count;
//       bins[min_index].count_pos += bins[min_index + 1].count_pos;
//       bins[min_index].count_neg += bins[min_index + 1].count_neg;
//       bins.erase(bins.begin() + min_index + 1);
//       iterations++;
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   // Enforce monotonicity of WoE values
//   void enforce_monotonicity() {
//     if (bins.size() <= 2) {
//       // If feature has two or fewer bins, ignore monotonicity enforcement
//       return;
//     }
//     int iterations = 0;
//     bool is_monotonic = false;
//     bool increasing = true;
//     if (bins.size() >= 2) {
//       increasing = (bins[1].woe >= bins[0].woe);
//     }
//     while (!is_monotonic && static_cast<int>(bins.size()) > min_bins && iterations < max_iterations) {
//       is_monotonic = true;
//       for (size_t i = 1; i < bins.size(); ++i) {
//         if ((increasing && bins[i].woe < bins[i - 1].woe) ||
//             (!increasing && bins[i].woe > bins[i - 1].woe)) {
//           // Merge bins[i - 1] and bins[i]
//           bins[i - 1].upper = bins[i].upper;
//           bins[i - 1].count += bins[i].count;
//           bins[i - 1].count_pos += bins[i].count_pos;
//           bins[i - 1].count_neg += bins[i].count_neg;
//           bins.erase(bins.begin() + i);
//           calculate_woe_iv();
//           is_monotonic = false;
//           break;
//         }
//       }
//       iterations++;
//       if (static_cast<int>(bins.size()) == min_bins) {
//         // min_bins reached, stop merging even if monotonicity is not achieved
//         break;
//       }
//     }
//     iterations_run += iterations;
//     if (iterations >= max_iterations) {
//       converged = false;
//     }
//   }
//   
//   // Calculate WoE and IV for each bin
//   void calculate_woe_iv() {
//     for (auto& bin : bins) {
//       bin.woe = calculate_woe(bin.count_pos, bin.count_neg);
//       bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg);
//     }
//   }
//   
// public:
//   OptimalBinningNumericalEWB(const std::vector<double>& feature_, const std::vector<int>& target_,
//                              int min_bins_ = 3, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20,
//                              double convergence_threshold_ = 1e-6, int max_iterations_ = 1000)
//     : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
//       convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//       converged(true), iterations_run(0), total_pos(0), total_neg(0), unique_count(0) {}
//   
//   void fit() {
//     validate_inputs();
//     
//     // Calculate total positives and negatives
//     total_pos = std::accumulate(target.begin(), target.end(), 0);
//     total_neg = target.size() - total_pos;
//     
//     if (total_pos == 0 || total_neg == 0) {
//       Rcpp::stop("Target vector must contain at least one positive and one negative case.");
//     }
//     
//     if (unique_count <= 2) {
//       // If unique values are 1 or 2, create bins based on unique values without optimization
//       create_unique_bins();
//       calculate_woe_iv();
//       converged = true;
//       iterations_run = 0;
//       return;
//     }
//     
//     // Create initial pre-bins
//     create_prebins();
//     assign_data_to_bins();
//     
//     // Merge rare bins
//     merge_rare_bins();
//     
//     // Calculate WoE and IV
//     calculate_woe_iv();
//     
//     // Enforce monotonicity
//     enforce_monotonicity();
//     
//     // Ensure the number of bins does not exceed max_bins
//     ensure_max_bins();
//     
//     // Recalculate WoE and IV after merging
//     calculate_woe_iv();
//   }
//   
//   // Get binning results
//   List get_results() const {
//     std::vector<std::string> bin_labels;
//     std::vector<double> woe_values;
//     std::vector<double> iv_values;
//     std::vector<int> counts;
//     std::vector<int> counts_pos;
//     std::vector<int> counts_neg;
//     std::vector<double> cutpoints;
//     
//     for (size_t i = 0; i < bins.size(); ++i) {
//       const auto& bin = bins[i];
//       std::string label = "(" + double_to_string(bin.lower) + ";" + double_to_string(bin.upper) + "]";
//       bin_labels.push_back(label);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       counts.push_back(bin.count);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//       if (i < bins.size() - 1) {
//         cutpoints.push_back(bin.upper);
//       }
//     }
//     
//     return List::create(
//       Named("bin") = bin_labels,
//       Named("woe") = woe_values,
//       Named("iv") = iv_values,
//       Named("count") = counts,
//       Named("count_pos") = counts_pos,
//       Named("count_neg") = counts_neg,
//       Named("cutpoints") = cutpoints,
//       Named("converged") = converged,
//       Named("iterations") = iterations_run
//     );
//   }
// };
// 
// //' @title Optimal Binning for Numerical Variables using Equal-Width Binning
// //'
// //' @description
// //' Performs optimal binning for numerical variables using an Equal-Width Binning approach with subsequent merging and adjustment. It aims to find a good binning strategy that balances interpretability and predictive power.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bins}{Character vector of bin ranges.}
// //' \item{woe}{Numeric vector of WoE values for each bin.}
// //' \item{iv}{Numeric vector of Information Value (IV) for each bin.}
// //' \item{count}{Numeric vector of total observations in each bin.}
// //' \item{count_pos}{Numeric vector of positive target observations in each bin.}
// //' \item{count_neg}{Numeric vector of negative target observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points to generate the bins.}
// //' \item{converged}{Logical indicating if the algorithm converged.}
// //' \item{iterations}{Integer number of iterations run by the algorithm.}
// //'
// //' @details
// //' The optimal binning algorithm using Equal-Width Binning consists of several steps:
// //' 1. Initial binning: The feature range is divided into \code{max_n_prebins} bins of equal width.
// //' 2. Assign data to bins based on feature values.
// //' 3. Merge rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
// //' 4. Enforce monotonicity of WoE values by merging adjacent non-monotonic bins.
// //' 5. Adjust the number of bins to not exceed \code{max_bins} by merging bins with the smallest counts.
// //' 6. Calculate WoE and IV for each bin.
// //'
// //' The algorithm aims to create bins that maximize the predictive power of the numerical variable while adhering to the specified constraints. It enforces monotonicity of WoE values, which is particularly useful for credit scoring and risk modeling applications.
// //'
// //' @examples
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- rnorm(1000)
// //' result <- optimal_binning_numerical_ewb(target, feature)
// //' print(result)
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_ewb(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
//                                          int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05,
//                                          int max_n_prebins = 20,
//                                          double convergence_threshold = 1e-6, int max_iterations = 1000) {
//   // Convert R vectors to C++ vectors
//   std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
//   std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//   
//   // Initialize the binning class
//   OptimalBinningNumericalEWB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins,
//                                     convergence_threshold, max_iterations);
//   
//   // Fit the binning
//   binner.fit();
//   
//   // Get binning results
//   List binning_results = binner.get_results();
//   
//   return binning_results;
// }
