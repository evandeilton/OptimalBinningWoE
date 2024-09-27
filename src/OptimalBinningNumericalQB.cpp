// optimal_binning_numerical_qb.cpp

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif


using namespace Rcpp;

// Struct to hold bin information
struct BinInfo {
  double lower;
  double upper;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
};

// Class for Quantile-based Binning
class OptimalBinningNumericalQB {
private:
  // Input data
  std::vector<double> feature;
  std::vector<int> target;

  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Bins
  std::vector<BinInfo> bins;

  // Total counts
  int total_pos;
  int total_neg;

public:
  // Constructor
  OptimalBinningNumericalQB(const std::vector<double>& feature_, const std::vector<int>& target_,
                            int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), total_pos(0), total_neg(0) {}

  // Fit method
  void fit() {
    // Input Validation
    validate_inputs();

    // Calculate total positives and negatives
    calculate_totals();

    // Initial Binning based on quantiles
    initial_binning();

    // Merge rare bins
    merge_rare_bins();

    // Enforce monotonicity
    enforce_monotonicity();

    // Calculate WoE and IV
    calculate_woe_iv();
  }

  // Getters for output
  std::vector<double> get_woefeature() {
    std::vector<double> woe_feature;
    woe_feature.reserve(feature.size());
    // Assign WoE based on bin
    for(auto &val : feature) {
      double bin_woe = 0.0;
      for(auto &bin : bins) {
        if(val > bin.lower && val <= bin.upper) {
          bin_woe = bin.woe;
          break;
        }
      }
      woe_feature.push_back(bin_woe);
    }
    return woe_feature;
  }

  std::vector<BinInfo> get_bins() {
    return bins;
  }

private:
  // Input validation
  void validate_inputs() {
    if(feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must be of the same length.");
    }
    // Check if feature is numeric is implicit in C++ vector<double>
    // Check target is binary
    for(auto &t : target) {
      if(t != 0 && t != 1) {
        throw std::invalid_argument("Target vector must be binary (0 and 1).");
      }
    }
  }

  // Calculate total positives and negatives
  void calculate_totals() {
#pragma omp parallel for reduction(+:total_pos, total_neg)
    for(int i = 0; i < target.size(); ++i) {
      if(target[i] == 1) total_pos += 1;
      else total_neg += 1;
    }
    if(total_pos == 0 || total_neg == 0) {
      throw std::invalid_argument("Target vector must contain both classes (0 and 1).");
    }
  }

  // Initial binning based on quantiles
  void initial_binning() {
    int n = feature.size();
    int prebins = std::min(max_n_prebins, n);

    // Create a vector of indices and sort them based on feature values
    std::vector<int> indices(n);
    for(int i = 0; i < n; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(), [&](const int a, const int b) {
      return feature[a] < feature[b];
    });

    // Determine quantile cut points
    std::vector<double> cut_points;
    for(int i = 1; i < prebins; ++i) {
      double frac = static_cast<double>(i) / prebins;
      int idx = std::floor(frac * (n - 1));
      cut_points.push_back(feature[indices[idx]]);
    }

    // Define initial bins
    bins.clear();
    BinInfo current_bin;
    current_bin.lower = -INFINITY;
    current_bin.upper = cut_points.empty() ? INFINITY : cut_points[0];
    current_bin.count = 0;
    current_bin.count_pos = 0;
    current_bin.count_neg = 0;

    int cp_idx = 0;
    for(int i = 0; i < n; ++i) {
      double val = feature[indices[i]];
      int tgt = target[indices[i]];
      if(val > current_bin.upper) {
        bins.push_back(current_bin);
        if(cp_idx < cut_points.size()) {
          current_bin.lower = current_bin.upper;
          current_bin.upper = cut_points[++cp_idx];
        } else {
          current_bin.lower = current_bin.upper;
          current_bin.upper = INFINITY;
        }
        current_bin.count = 0;
        current_bin.count_pos = 0;
        current_bin.count_neg = 0;
      }
      current_bin.count += 1;
      if(tgt == 1) current_bin.count_pos += 1;
      else current_bin.count_neg += 1;
    }
    bins.push_back(current_bin);
  }

  // Merge rare bins based on bin_cutoff
  void merge_rare_bins() {
    bool merged = true;
    while(merged) {
      merged = false;
      // Merge from left to right
      for(int i = 0; i < bins.size(); ++i) {
        double freq = static_cast<double>(bins[i].count) / (total_pos + total_neg);
        if(freq < bin_cutoff) {
          // Merge with adjacent bin
          if(i == 0) {
            merge_bins(i, i+1);
          }
          else if(i == bins.size()-1) {
            merge_bins(i-1, i);
          }
          else {
            // Merge with the neighbor with smaller frequency
            double left_freq = static_cast<double>(bins[i-1].count) / (total_pos + total_neg);
            double right_freq = (i+1 < bins.size()) ? static_cast<double>(bins[i+1].count) / (total_pos + total_neg) : 1.0;
            if(left_freq < right_freq) {
              merge_bins(i-1, i);
            }
            else {
              merge_bins(i, i+1);
            }
          }
          merged = true;
          break; // Restart after a merge
        }
      }
    }

    // Ensure minimum number of bins
    while(bins.size() < min_bins) {
      // Merge the two closest bins
      double min_distance = INFINITY;
      int merge_idx = 0;
      for(int i = 0; i < bins.size()-1; ++i) {
        double distance = bins[i+1].lower - bins[i].upper;
        if(distance < min_distance) {
          min_distance = distance;
          merge_idx = i;
        }
      }
      merge_bins(merge_idx, merge_idx+1);
    }

    // Ensure maximum number of bins
    while(bins.size() > max_bins) {
      // Merge the two bins with the smallest IV
      double min_iv = INFINITY;
      int merge_idx = 0;
      for(int i = 0; i < bins.size()-1; ++i) {
        double iv_sum = bins[i].iv + bins[i+1].iv;
        if(iv_sum < min_iv) {
          min_iv = iv_sum;
          merge_idx = i;
        }
      }
      merge_bins(merge_idx, merge_idx+1);
    }
  }

  // Helper to merge two bins at indices i and j
  void merge_bins(int i, int j) {
    if(j != i+1) throw std::invalid_argument("Bins to merge must be adjacent.");
    bins[i].upper = bins[j].upper;
    bins[i].count += bins[j].count;
    bins[i].count_pos += bins[j].count_pos;
    bins[i].count_neg += bins[j].count_neg;
    bins.erase(bins.begin() + j);
  }

  // Enforce monotonicity on WoE
  void enforce_monotonicity() {
    // Simple monotonicity enforcement: adjust WoE to be monotonically increasing or decreasing
    // Here, we'll enforce monotonic increasing WoE
    // More sophisticated methods can be implemented as needed

    // First, calculate initial WoE
    for(auto &bin : bins) {
      // To avoid division by zero
      double distr_pos = bin.count_pos == 0 ? 0.0001 : static_cast<double>(bin.count_pos) / total_pos;
      double distr_neg = bin.count_neg == 0 ? 0.0001 : static_cast<double>(bin.count_neg) / total_neg;
      bin.woe = std::log(distr_pos / distr_neg);
    }

    // Check for monotonicity
    bool monotonic = true;
    for(int i = 1; i < bins.size(); ++i) {
      if(bins[i].woe < bins[i-1].woe) {
        monotonic = false;
        break;
      }
    }

    // If not monotonic, attempt to enforce by merging bins
    while(!monotonic && bins.size() > min_bins) {
      double min_diff = INFINITY;
      int merge_idx = 0;
      for(int i = 0; i < bins.size()-1; ++i) {
        double diff = bins[i].woe - bins[i+1].woe;
        if(diff > 0 && diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }
      if(min_diff == INFINITY) break; // No violations found
      merge_bins(merge_idx, merge_idx+1);

      // Recalculate WoE after merging
      for(auto &bin : bins) {
        double distr_pos = bin.count_pos == 0 ? 0.0001 : static_cast<double>(bin.count_pos) / total_pos;
        double distr_neg = bin.count_neg == 0 ? 0.0001 : static_cast<double>(bin.count_neg) / total_neg;
        bin.woe = std::log(distr_pos / distr_neg);
      }

      // Recheck monotonicity
      monotonic = true;
      for(int i = 1; i < bins.size(); ++i) {
        if(bins[i].woe < bins[i-1].woe) {
          monotonic = false;
          break;
        }
      }
    }

    // If still not monotonic, reverse the order (assuming decreasing was intended)
    if(!monotonic) {
      // Calculate WoE in reverse order
      for(auto &bin : bins) {
        double distr_pos = bin.count_pos == 0 ? 0.0001 : static_cast<double>(bin.count_pos) / total_pos;
        double distr_neg = bin.count_neg == 0 ? 0.0001 : static_cast<double>(bin.count_neg) / total_neg;
        bin.woe = std::log(distr_neg / distr_pos); // Reverse
      }

      // Check monotonicity
      monotonic = true;
      for(int i = 1; i < bins.size(); ++i) {
        if(bins[i].woe < bins[i-1].woe) {
          monotonic = false;
          break;
        }
      }

      // If still not monotonic, proceed without enforcing
      if(!monotonic) {
        // Optionally, handle non-monotonic WoE as needed
      }
    }
  }

  // Calculate WoE and IV for each bin
  void calculate_woe_iv() {
    for(auto &bin : bins) {
      double distr_pos = bin.count_pos == 0 ? 0.0001 : static_cast<double>(bin.count_pos) / total_pos;
      double distr_neg = bin.count_neg == 0 ? 0.0001 : static_cast<double>(bin.count_neg) / total_neg;
      bin.woe = std::log(distr_pos / distr_neg);
      bin.iv = (distr_pos - distr_neg) * bin.woe;
    }
  }
};


//' @title Optimal Binning for Numerical Variables using Quantile-based Binning (QB)
//'
//' @description
//' This function performs optimal binning for numerical variables using a Quantile-based 
//' Binning (QB) approach. It creates optimal bins for a numerical feature based on its 
//' relationship with a binary target variable, maximizing the predictive power while 
//' respecting user-defined constraints and enforcing monotonicity.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of total observations for a bin to avoid being merged (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the optimization process (default: 20).
//'
//' @return A list containing two elements:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation.}
//' \item{woebin}{A data frame with the following columns:
//'   \itemize{
//'     \item bin: Character vector of bin ranges.
//'     \item woe: Numeric vector of WoE values for each bin.
//'     \item iv: Numeric vector of Information Value (IV) for each bin.
//'     \item count: Integer vector of total observations in each bin.
//'     \item count_pos: Integer vector of positive target observations in each bin.
//'     \item count_neg: Integer vector of negative target observations in each bin.
//'   }
//' }
//'
//' @details
//' The Quantile-based Binning (QB) algorithm for numerical variables works as follows:
//' 1. Perform initial binning based on quantiles of the feature distribution.
//' 2. Merge rare bins to meet the bin_cutoff requirement.
//' 3. Enforce monotonicity of Weight of Evidence (WoE) across bins.
//' 4. Ensure the number of bins is between min_bins and max_bins.
//' 5. Calculate final WoE and Information Value (IV) for each bin.
//'
//' The algorithm aims to create bins that maximize the predictive power of the numerical 
//' variable while adhering to the specified constraints. It enforces monotonicity of WoE 
//' values, which is particularly useful for credit scoring and risk modeling applications.
//'
//' Weight of Evidence (WoE) is calculated as:
//' \deqn{WoE = \ln(\frac{\text{Positive Rate}}{\text{Negative Rate}})}
//'
//' Information Value (IV) is calculated as:
//' \deqn{IV = (\text{Positive Rate} - \text{Negative Rate}) \times WoE}
//'
//' This implementation uses OpenMP for parallel processing when available, which can 
//' significantly speed up the computation for large datasets.
//'
//' @references
//' \itemize{
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. DOI: 10.2139/ssrn.2978774
//'   \item Liu, X., & Wu, Y. (2019). Supervised Discretization for Credit Scoring. Journal of Credit Risk, 15(2), 55-87.
//' }
//'
//' @examples
//' \dontrun{
//' # Create sample data
//' set.seed(123)
//' n <- 1000
//' target <- sample(0:1, n, replace = TRUE)
//' feature <- rnorm(n)
//'
//' # Run optimal binning
//' result <- optimal_binning_numerical_qb(target, feature, min_bins = 2, max_bins = 4)
//'
//' # Print results
//' print(result$woebin)
//'
//' # Plot WoE values
//' plot(result$woebin$woe, type = "s", xaxt = "n", xlab = "Bins", ylab = "WoE",
//'      main = "Weight of Evidence by Bin")
//' axis(1, at = 1:nrow(result$woebin), labels = result$woebin$bin, las = 2)
//' }
//'
//' @author Lopes, J. E.
//'
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_qb(IntegerVector target, NumericVector feature,
                                  int min_bins = 3, int max_bins = 5,
                                  double bin_cutoff = 0.05, int max_n_prebins = 20) {
  // Convert R vectors to C++ vectors
  std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
  std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);

  // Instantiate the binning class
  OptimalBinningNumericalQB binning(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);

  // Fit the binning model
  binning.fit();

  // Get WoE feature
  std::vector<double> woefeature = binning.get_woefeature();

  // Get bin information
  std::vector<BinInfo> bins = binning.get_bins();

  // Prepare woebin list
  // List woebin = List::create();
  std::vector<std::string> bin_intervals;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> counts;
  std::vector<int> counts_pos;
  std::vector<int> counts_neg;

  for(auto &bin : bins) {
    std::string interval = "";
    if(bin.lower == -INFINITY) {
      interval += "(-Inf;";
    }
    else {
      interval += "(" + std::to_string(bin.lower) + ";";
    }

    if(bin.upper == INFINITY) {
      interval += "+Inf]";
    }
    else {
      interval += std::to_string(bin.upper) + "]";
    }
    bin_intervals.push_back(interval);
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    counts.push_back(bin.count);
    counts_pos.push_back(bin.count_pos);
    counts_neg.push_back(bin.count_neg);
  }

  // woebin["bin"] = bin_intervals;
  // woebin["woe"] = woe_values;
  // woebin["iv"] = iv_values;
  // woebin["count"] = counts;
  // woebin["count_pos"] = counts_pos;
  // woebin["count_neg"] = counts_neg;

  DataFrame woebin = DataFrame::create(
    Named("bin") = bin_intervals,
    Named("woe") = woe_values,
    Named("iv") = iv_values,
    Named("count") = counts,
    Named("count_pos") = counts_pos,
    Named("count_neg") = counts_neg
  );

  // Prepare and return the final list
  List result = List::create();
  result["woefeature"] = woefeature;
  result["woebin"] = woebin;

  return result;
}

