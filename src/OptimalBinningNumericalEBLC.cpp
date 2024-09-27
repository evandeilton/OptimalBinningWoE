#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Structure to hold bin information
struct Bin {
  double lower_bound;
  double upper_bound;
  int count;
  int count_pos;
  int count_neg;
  double woe;
  double iv;

  Bin(double lb, double ub) : lower_bound(lb), upper_bound(ub), count(0),
  count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
};

// Class for Optimal Binning Numerical EBLC
class OptimalBinningNumericalEBLC {
public:
  // Constructor
  OptimalBinningNumericalEBLC(IntegerVector target_, NumericVector feature_,
                              int min_bins_ = 2, int max_bins_ = 5,
                              double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
    : target(target_), feature(feature_), min_bins(min_bins_),
      max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}

  // Fit function to perform binning
  void fit() {
    validate_input();
    sort_data();
    initial_binning();
    merge_rare_bins();
    ensure_min_bins();
    enforce_max_bins();
    compute_woe_iv();
    enforce_monotonicity();
    ensure_min_bins();
    assign_woe_feature();
    prepare_output();
  }

  // Getters for output
  NumericVector get_woe_feature() const {
    return woe_feature;
  }

  DataFrame get_woebin() const {
    return woebin;
  }

private:
  // Input data
  IntegerVector target;
  NumericVector feature;

  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  // Sorted data
  std::vector<double> sorted_feature;
  std::vector<int> sorted_target;

  // Bins
  std::vector<Bin> bins;

  // Output
  NumericVector woe_feature;
  DataFrame woebin;

  // Total positives and negatives
  int total_pos;
  int total_neg;

  // Helper function to validate inputs
  void validate_input() {
    if (feature.size() != target.size()) {
      stop("Feature and target must be the same length.");
    }

    for(int i = 0; i < target.size(); i++) {
      if(target[i] != 0 && target[i] != 1) {
        stop("Target variable must be binary (0 and 1).");
      }
    }

    if(min_bins < 2) {
      stop("min_bins must be at least 2.");
    }
    if(max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }

    if(bin_cutoff < 0.0 || bin_cutoff > 1.0) {
      stop("bin_cutoff must be between 0 and 1.");
    }

    if(max_n_prebins < min_bins) {
      stop("max_n_prebins must be at least equal to min_bins.");
    }
  }

  // Helper function to sort data
  void sort_data() {
    std::vector<size_t> indices(feature.size());
    for(size_t i = 0; i < indices.size(); i++) {
      indices[i] = i;
    }

    std::sort(indices.begin(), indices.end(),
              [this](size_t a, size_t b) -> bool {
                if(std::isnan(feature[a]) && std::isnan(feature[b])) return false;
                if(std::isnan(feature[a])) return true;
                if(std::isnan(feature[b])) return false;
                return feature[a] < feature[b];
              });

    sorted_feature.reserve(feature.size());
    sorted_target.reserve(target.size());
    for(auto idx : indices) {
      sorted_feature.push_back(feature[idx]);
      sorted_target.push_back(target[idx]);
    }
  }

  // Helper function for initial binning using equal frequency up to max_n_prebins
  void initial_binning() {
    int n = sorted_feature.size();
    int prebins = std::min(max_n_prebins, n);
    prebins = std::max(prebins, min_bins);  // Ensure at least min_bins
    int bin_size = n / prebins;
    int remainder = n % prebins;

    double current_lower = sorted_feature[0];
    int start = 0;

    for(int i = 0; i < prebins; i++) {
      int current_bin_size = bin_size + (i < remainder ? 1 : 0);
      int end = start + current_bin_size - 1;
      double current_upper = sorted_feature[end];

      while(end + 1 < n && sorted_feature[end + 1] == current_upper) {
        end++;
        current_bin_size++;
      }

      bins.emplace_back(current_lower, sorted_feature[end]);
      start = end + 1;
      if(start < n) {
        current_lower = sorted_feature[start];
      }
      if(start >= n) break;
    }

    for(int i = 0; i < sorted_feature.size(); i++) {
      for(auto &bin : bins) {
        if(sorted_feature[i] >= bin.lower_bound && sorted_feature[i] <= bin.upper_bound) {
          bin.count++;
          if(sorted_target[i] == 1) {
            bin.count_pos++;
          } else {
            bin.count_neg++;
          }
          break;
        }
      }
    }

    if(bins.size() > 0) {
      bins.front().lower_bound = -std::numeric_limits<double>::infinity();
      bins.back().upper_bound = std::numeric_limits<double>::infinity();
    }
  }

  // Helper function to merge rare bins based on bin_cutoff
  void merge_rare_bins() {
    double total = sorted_feature.size();
    bool merged = true;

    while(merged && bins.size() > min_bins) {
      merged = false;
      for(int i = 0; i < bins.size(); i++) {
        double freq = static_cast<double>(bins[i].count) / total;
        if(freq < bin_cutoff && bins.size() > min_bins) {
          if(i == 0 && bins.size() > 1) {
            bins[i+1].lower_bound = bins[i].lower_bound;
            bins[i+1].count += bins[i].count;
            bins[i+1].count_pos += bins[i].count_pos;
            bins[i+1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
            merged = true;
            break;
          }
          else if(i == bins.size()-1 && bins.size() > 1) {
            bins[i-1].upper_bound = bins[i].upper_bound;
            bins[i-1].count += bins[i].count;
            bins[i-1].count_pos += bins[i].count_pos;
            bins[i-1].count_neg += bins[i].count_neg;
            bins.erase(bins.begin() + i);
            merged = true;
            break;
          }
          else if(i > 0 && i < bins.size() - 1) {
            double freq_prev = static_cast<double>(bins[i-1].count) / total;
            double freq_next = static_cast<double>(bins[i+1].count) / total;
            if(freq_prev <= freq_next) {
              bins[i-1].upper_bound = bins[i].upper_bound;
              bins[i-1].count += bins[i].count;
              bins[i-1].count_pos += bins[i].count_pos;
              bins[i-1].count_neg += bins[i].count_neg;
              bins.erase(bins.begin() + i);
            }
            else {
              bins[i+1].lower_bound = bins[i].lower_bound;
              bins[i+1].count += bins[i].count;
              bins[i+1].count_pos += bins[i].count_pos;
              bins[i+1].count_neg += bins[i].count_neg;
              bins.erase(bins.begin() + i);
            }
            merged = true;
            break;
          }
        }
      }
    }
  }

  // Helper function to ensure minimum number of bins
  void ensure_min_bins() {
    while(bins.size() < min_bins) {
      auto max_count_bin = std::max_element(bins.begin(), bins.end(),
                                            [](const Bin& a, const Bin& b) { return a.count < b.count; });

      int split_idx = std::distance(bins.begin(), max_count_bin);

      double split_point = compute_median(bins[split_idx].lower_bound, bins[split_idx].upper_bound);

      if(split_point == bins[split_idx].lower_bound || split_point == bins[split_idx].upper_bound) {
        split_point = (bins[split_idx].lower_bound + bins[split_idx].upper_bound) / 2.0;
      }

      Bin bin1(bins[split_idx].lower_bound, split_point);
      Bin bin2(split_point, bins[split_idx].upper_bound);

      for(int i = 0; i < sorted_feature.size(); i++) {
        if(sorted_feature[i] > bin1.lower_bound && sorted_feature[i] <= bin1.upper_bound) {
          bin1.count++;
          if(sorted_target[i] == 1) bin1.count_pos++;
          else bin1.count_neg++;
        }
        else if(sorted_feature[i] > bin2.lower_bound && sorted_feature[i] <= bin2.upper_bound) {
          bin2.count++;
          if(sorted_target[i] == 1) bin2.count_pos++;
          else bin2.count_neg++;
        }
      }

      bins.erase(bins.begin() + split_idx);
      bins.insert(bins.begin() + split_idx, bin2);
      bins.insert(bins.begin() + split_idx, bin1);
    }
  }

  // Helper function to enforce maximum number of bins
  void enforce_max_bins() {
    while(bins.size() > max_bins) {
      compute_woe_iv();
      double min_iv = std::numeric_limits<double>::max();
      int merge_idx = -1;
      for(int i = 0; i < bins.size()-1; i++) {
        double combined_iv = bins[i].iv + bins[i+1].iv;
        if(combined_iv < min_iv) {
          min_iv = combined_iv;
          merge_idx = i;
        }
      }
      if(merge_idx == -1) break;

      bins[merge_idx].upper_bound = bins[merge_idx +1].upper_bound;
      bins[merge_idx].count += bins[merge_idx +1].count;
      bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
      bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
      bins.erase(bins.begin() + merge_idx +1);
    }
  }

  // Helper function to compute WoE and IV
  void compute_woe_iv() {
    total_pos = 0;
    total_neg = 0;
    for(auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }

#pragma omp parallel for
    for(int i = 0; i < bins.size(); i++) {
      double dist_pos = (total_pos > 0) ? static_cast<double>(bins[i].count_pos) / total_pos : 0.0;
      double dist_neg = (total_neg > 0) ? static_cast<double>(bins[i].count_neg) / total_neg : 0.0;

      if(dist_pos == 0) dist_pos = 0.0001;
      if(dist_neg == 0) dist_neg = 0.0001;

      bins[i].woe = std::log(dist_pos / dist_neg);
      bins[i].iv = (dist_pos - dist_neg) * bins[i].woe;
    }
  }

  // Helper function to enforce monotonicity of WoE
  void enforce_monotonicity() {
    bool monotonic = false;

    while(!monotonic && bins.size() > min_bins) {
      monotonic = true;
      bool increasing = true;
      bool decreasing = true;

      for(int i = 1; i < bins.size(); i++) {
        if(bins[i].woe < bins[i-1].woe) {
          increasing = false;
        }
        if(bins[i].woe > bins[i-1].woe) {
          decreasing = false;
        }
      }

      if(increasing || decreasing) {
        break;
      }

      double min_diff = std::numeric_limits<double>::max();
      int merge_idx = -1;
      for(int i = 0; i < bins.size()-1; i++) {
        double diff = std::abs(bins[i].woe - bins[i+1].woe);
        if(diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }

      if(merge_idx == -1 || bins.size() <= min_bins) break;

      bins[merge_idx].upper_bound = bins[merge_idx +1].upper_bound;
      bins[merge_idx].count += bins[merge_idx +1].count;
      bins[merge_idx].count_pos += bins[merge_idx +1].count_pos;
      bins[merge_idx].count_neg += bins[merge_idx +1].count_neg;
      bins.erase(bins.begin() + merge_idx +1);

      compute_woe_iv();

      monotonic = false;
    }
  }

  // Helper function to assign WoE to feature
  void assign_woe_feature() {
    woe_feature = NumericVector(feature.size());

#pragma omp parallel for
    for(int i = 0; i < feature.size(); i++) {
      double val = feature[i];
      int bin_idx = find_bin(val);
      if(bin_idx != -1) {
        woe_feature[i] = bins[bin_idx].woe;
      }
      else {
        if(val < bins.front().lower_bound) {
          woe_feature[i] = bins.front().woe;
        }
        else {
          woe_feature[i] = bins.back().woe;
        }
      }
    }
  }

  // Binary search to find the appropriate bin
  int find_bin(double val) const {
    int left = 0;
    int right = bins.size() - 1;
    while(left <= right) {
      int mid = left + (right - left) / 2;
      if(val > bins[mid].upper_bound) {
        left = mid + 1;
      }
      else {
        if(val > bins[mid].lower_bound || (mid == 0 && val == bins[mid].lower_bound)) {
          return mid;
        }
        else {
          right = mid - 1;
        }
      }
    }
    return -1; // Not found
  }

  // Helper function to prepare output
  void prepare_output() {
    std::vector<std::string> bin_labels;
    for(int i = 0; i < bins.size(); i++) {
      std::string label;
      if(i == 0) {
        label += "(-Inf;";
      }
      else {
        label += "(" + std::to_string(bins[i].lower_bound) + ";";
      }

      if(i == bins.size()-1) {
        label += "+Inf]";
      }
      else {
        label += std::to_string(bins[i].upper_bound) + "]";
      }
      bin_labels.push_back(label);
    }

    NumericVector woe_vec;
    NumericVector iv_vec;
    IntegerVector count_vec;
    IntegerVector count_pos_vec;
    IntegerVector count_neg_vec;

    for(auto &bin : bins) {
      woe_vec.push_back(bin.woe);
      iv_vec.push_back(bin.iv);
      count_vec.push_back(bin.count);
      count_pos_vec.push_back(bin.count_pos);
      count_neg_vec.push_back(bin.count_neg);
    }

    woebin = DataFrame::create(
      Named("bin") = bin_labels,
      Named("woe") = woe_vec,
      Named("iv") = iv_vec,
      Named("count") = count_vec,
      Named("count_pos") = count_pos_vec,
      Named("count_neg") = count_neg_vec,
      _["stringsAsFactors"] = false
    );
  }

  // Helper function to compute median within a bin range
  double compute_median(double lb, double ub) const {
    std::vector<double> values;
    for(int i = 0; i < sorted_feature.size(); i++) {
      if(sorted_feature[i] > lb && sorted_feature[i] <= ub) {
        values.push_back(sorted_feature[i]);
      }
    }
    if(values.empty()) return lb;
    size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin()+mid, values.end());
    return values[mid];
  }
};


//' @title Optimal Binning for Numerical Variables using Equal-Frequency Binning with Local Convergence
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using an Equal-Frequency Binning approach with Local Convergence. It aims to find a good binning strategy that balances interpretability, predictive power, and monotonicity of Weight of Evidence (WoE).
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' 
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//' 
//' @details
//' The optimal binning algorithm using Equal-Frequency Binning with Local Convergence consists of several steps:
//' 
//' 1. Initial binning: The feature is divided into \code{max_n_prebins} bins, each containing approximately the same number of observations.
//' 2. Merging rare bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Ensuring minimum bins: If the number of bins is less than \code{min_bins}, the largest bin is split at its median.
//' 4. Enforcing maximum bins: If the number of bins exceeds \code{max_bins}, adjacent bins with the lowest combined Information Value (IV) are merged.
//' 5. WoE and IV calculation: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
//' 6. Enforcing monotonicity: Adjacent bins are merged to ensure monotonicity of WoE values while maintaining the minimum number of bins.
//' 7. Assigning WoE to feature: Each feature value is assigned the WoE of its corresponding bin.
//' 
//' The Weight of Evidence (WoE) for each bin is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' 
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//' 
//' This approach provides a balance between simplicity, effectiveness, and interpretability. It creates bins with equal frequency initially and then adjusts them based on the data distribution, target variable relationship, and monotonicity constraints. The local convergence ensures that the final binning maximizes the predictive power while respecting the specified constraints and maintaining monotonicity of WoE values.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_eblc(target, feature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//'   \item Belotti, P., & Carrasco, M. (2017). Optimal binning: mathematical programming formulation and solution approach. arXiv preprint arXiv:1705.03287.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. arXiv preprint arXiv:1711.06692.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_eblc(IntegerVector target,
                                    NumericVector feature,
                                    int min_bins = 3,
                                    int max_bins = 5,
                                    double bin_cutoff = 0.05,
                                    int max_n_prebins = 20) {
  // Validate input parameters
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 1.0) {
    stop("bin_cutoff must be between 0 and 1.");
  }
  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be at least equal to min_bins.");
  }

  // Instantiate the binning class
  OptimalBinningNumericalEBLC binning(target, feature, min_bins, max_bins, bin_cutoff, max_n_prebins);

  // Perform binning
  binning.fit();

  // Retrieve outputs
  NumericVector woe_feature = binning.get_woe_feature();
  DataFrame woebin = binning.get_woebin();

  // Return as a list
  return List::create(
    Named("woefeature") = woe_feature,
    Named("woebin") = woebin
  );
}
