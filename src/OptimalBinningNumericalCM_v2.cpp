#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>

// Class implementing the Optimal Binning using ChiMerge method
class OptimalBinningNumericalCM {
private:
  // Input data
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  double convergence_threshold;
  int max_iterations;
  
  // Structure representing a bin
  struct Bin {
    double lower_bound;
    double upper_bound;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  // Binning results
  std::vector<Bin> bins;
  bool converged;
  int iterations_run;
  
  static constexpr double EPSILON = 1e-10;
  
  // Calculate chi-square statistic between two bins
  double calculate_chi_square(const Bin& bin1, const Bin& bin2) const {
    int total_pos = bin1.count_pos + bin2.count_pos;
    int total_neg = bin1.count_neg + bin2.count_neg;
    int total = total_pos + total_neg;
    
    double expected_pos1 = std::max(static_cast<double>(bin1.count) * total_pos / total, EPSILON);
    double expected_neg1 = std::max(static_cast<double>(bin1.count) * total_neg / total, EPSILON);
    double expected_pos2 = std::max(static_cast<double>(bin2.count) * total_pos / total, EPSILON);
    double expected_neg2 = std::max(static_cast<double>(bin2.count) * total_neg / total, EPSILON);
    
    double chi_square =
      std::pow(bin1.count_pos - expected_pos1, 2) / expected_pos1 +
      std::pow(bin1.count_neg - expected_neg1, 2) / expected_neg1 +
      std::pow(bin2.count_pos - expected_pos2, 2) / expected_pos2 +
      std::pow(bin2.count_neg - expected_neg2, 2) / expected_neg2;
    
    return chi_square;
  }
  
  // Merge two adjacent bins at the given index
  void merge_bins(size_t index) {
    Bin& left = bins[index];
    Bin& right = bins[index + 1];
    
    left.upper_bound = right.upper_bound;
    left.count += right.count;
    left.count_pos += right.count_pos;
    left.count_neg += right.count_neg;
    
    bins.erase(bins.begin() + index + 1);
  }
  
  // Calculate Weight of Evidence (WoE) and Information Value (IV) for each bin
  void calculate_woe_iv() {
    double total_pos = 0, total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    for (auto& bin : bins) {
      double pos_rate = std::max(static_cast<double>(bin.count_pos) / total_pos, EPSILON);
      double neg_rate = std::max(static_cast<double>(bin.count_neg) / total_neg, EPSILON);
      
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
    }
  }
  
  // Check if the bins are monotonic in terms of WoE
  bool is_monotonic() const {
    if (bins.size() <= 2) return true;
    
    bool increasing = bins[1].woe > bins[0].woe;
    for (size_t i = 2; i < bins.size(); ++i) {
      if ((increasing && bins[i].woe < bins[i - 1].woe) ||
          (!increasing && bins[i].woe > bins[i - 1].woe)) {
        return false;
      }
    }
    return true;
  }
  
public:
  // Constructor with input validation
  OptimalBinningNumericalCM(
    const std::vector<double>& feature,
    const std::vector<int>& target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : feature(feature), target(target), min_bins(min_bins), max_bins(max_bins),
  bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins),
  convergence_threshold(convergence_threshold), max_iterations(max_iterations),
  converged(false), iterations_run(0) {
    
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }
    
    if (min_bins < 2 || max_bins < min_bins) {
      throw std::invalid_argument("Invalid bin constraints: min_bins must be at least 2 and max_bins must be greater than or equal to min_bins");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
    }
    
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be greater than or equal to max_bins");
    }
  }
  
  // Fit the optimal binning model
  void fit() {
    // Count number of unique feature values
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    int num_unique_values = static_cast<int>(unique_values.size());
    
    if (num_unique_values <= min_bins) {
      // No need to optimize; create bins based on unique values
      bins.clear();
      bins.reserve(num_unique_values);
      for (size_t i = 0; i < unique_values.size(); ++i) {
        Bin bin;
        bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
        bin.upper_bound = unique_values[i];
        bin.count = 0;
        bin.count_pos = 0;
        bin.count_neg = 0;
        bins.push_back(bin);
      }
      
      // Assign counts to bins
      for (size_t i = 0; i < feature.size(); ++i) {
        double val = feature[i];
        int tgt = target[i];
        for (auto& bin : bins) {
          if (val <= bin.upper_bound) {
            bin.count++;
            if (tgt == 1) {
              bin.count_pos++;
            } else {
              bin.count_neg++;
            }
            break;
          }
        }
      }
      calculate_woe_iv();
      converged = true;
      iterations_run = 0;
      return;
    }
    
    // Initial binning
    std::vector<std::pair<double, int>> sorted_data(feature.size());
    for (size_t i = 0; i < feature.size(); ++i) {
      sorted_data[i] = {feature[i], target[i]};
    }
    std::sort(sorted_data.begin(), sorted_data.end());
    
    // Create initial bins
    int records_per_bin = std::max(1, static_cast<int>(sorted_data.size()) / max_n_prebins);
    bins.clear();
    bins.reserve(max_n_prebins);
    
    for (size_t i = 0; i < sorted_data.size(); i += records_per_bin) {
      size_t end = std::min(i + records_per_bin, sorted_data.size());
      Bin bin;
      bin.lower_bound = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_data[i].first;
      bin.upper_bound = (end == sorted_data.size()) ? std::numeric_limits<double>::infinity() : sorted_data[end - 1].first;
      bin.count = end - i;
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (size_t j = i; j < end; ++j) {
        if (sorted_data[j].second == 1) {
          bin.count_pos++;
        } else {
          bin.count_neg++;
        }
      }
      
      bins.push_back(bin);
    }
    
    // ChiMerge algorithm
    double prev_total_iv = 0;
    for (iterations_run = 0; iterations_run < max_iterations; ++iterations_run) {
      if (bins.size() <= min_bins) {
        converged = true;
        break;
      }
      
      double min_chi_square = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
        if (chi_square < min_chi_square) {
          min_chi_square = chi_square;
          merge_index = i;
        }
      }
      
      merge_bins(merge_index);
      calculate_woe_iv();
      
      double total_iv = 0;
      for (const auto& bin : bins) {
        total_iv += bin.iv;
      }
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = total_iv;
      
      if (bins.size() <= max_bins && is_monotonic()) {
        converged = true;
        break;
      }
    }
    
    // Merge rare bins based on bin_cutoff
    double total_count = static_cast<double>(feature.size());
    bool merged_bins = true;
    while (merged_bins) {
      merged_bins = false;
      for (size_t i = 0; i < bins.size(); ) {
        if (static_cast<double>(bins[i].count) / total_count < bin_cutoff) {
          if (i == 0) {
            merge_bins(0);
          } else {
            merge_bins(i - 1);
          }
          merged_bins = true;
          calculate_woe_iv();
          i = 0; // Restart after merging
        } else {
          ++i;
        }
      }
    }
    
    // Ensure monotonicity if possible
    while (!is_monotonic() && bins.size() > min_bins) {
      double min_chi_square = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double chi_square = calculate_chi_square(bins[i], bins[i + 1]);
        if (chi_square < min_chi_square) {
          min_chi_square = chi_square;
          merge_index = i;
        }
      }
      
      merge_bins(merge_index);
      calculate_woe_iv();
    }
  }
  
  // Retrieve the binning results
  Rcpp::List get_results() const {
    Rcpp::StringVector bin_names;
    Rcpp::NumericVector bin_woe;
    Rcpp::NumericVector bin_iv;
    Rcpp::IntegerVector bin_count;
    Rcpp::IntegerVector bin_count_pos;
    Rcpp::IntegerVector bin_count_neg;
    Rcpp::NumericVector bin_cutpoints;
    
    for (const auto& bin : bins) {
      std::string bin_name = (bin.lower_bound == -std::numeric_limits<double>::infinity() ? "(-Inf" : "(" + std::to_string(bin.lower_bound)) +
        ";" + (bin.upper_bound == std::numeric_limits<double>::infinity() ? "+Inf]" : std::to_string(bin.upper_bound) + "]");
      bin_names.push_back(bin_name);
      bin_woe.push_back(bin.woe);
      bin_iv.push_back(bin.iv);
      bin_count.push_back(bin.count);
      bin_count_pos.push_back(bin.count_pos);
      bin_count_neg.push_back(bin.count_neg);
      
      if (bin.upper_bound != std::numeric_limits<double>::infinity()) {
        bin_cutpoints.push_back(bin.upper_bound);
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bins") = bin_names,
      Rcpp::Named("woe") = bin_woe,
      Rcpp::Named("iv") = bin_iv,
      Rcpp::Named("count") = bin_count,
      Rcpp::Named("count_pos") = bin_count_pos,
      Rcpp::Named("count_neg") = bin_count_neg,
      Rcpp::Named("cutpoints") = bin_cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

//' @title Optimal Binning for Numerical Variables using ChiMerge
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial discretization (default: 20).
//' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bins}{A character vector of bin names.}
//' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
//' \item{iv}{A numeric vector of Information Value for each bin.}
//' \item{count}{An integer vector of total counts for each bin.}
//' \item{count_pos}{An integer vector of positive target counts for each bin.}
//' \item{count_neg}{An integer vector of negative target counts for each bin.}
//' \item{cutpoints}{A numeric vector of cutpoints used to create the bins.}
//' \item{converged}{A logical value indicating whether the algorithm converged.}
//' \item{iterations}{An integer value indicating the number of iterations run.}
//'
//' @details
//' The optimal binning algorithm for numerical variables uses the ChiMerge approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization into max_n_prebins
//' 2. Iterative merging of adjacent bins based on chi-square statistic
//' 3. Merging of rare bins based on the bin_cutoff parameter
//' 4. Calculation of WoE and IV for each final bin
//'
//' The chi-square statistic for two adjacent bins is calculated as:
//'
//' \deqn{\chi^2 = \sum_{i=1}^{2} \sum_{j=1}^{2} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}}
//'
//' where \eqn{O_{ij}} is the observed frequency and \eqn{E_{ij}} is the expected frequency for bin i and class j.
//'
//' Weight of Evidence (WoE) is calculated for each bin as:
//'
//' \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//'
//' where \eqn{P(X_i|Y=1)} is the proportion of positive cases in bin i, and \eqn{P(X_i|Y=0)} is the proportion of negative cases in bin i.
//'
//' Information Value (IV) for each bin is calculated as:
//'
//' \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'
//' The total IV for the feature is the sum of IVs across all bins:
//'
//' \deqn{IV_{total} = \sum_{i=1}^{n} IV_i}
//'
//' The ChiMerge approach ensures that the resulting binning maximizes the separation between classes while maintaining the desired number of bins and respecting the minimum bin frequency constraint.
//'
//' @references
//' \itemize{
//'   \item Kerber, R. (1992). ChiMerge: Discretization of Numeric Attributes. In Proceedings of the tenth national conference on Artificial intelligence (pp. 123-128). AAAI Press.
//'   \item Zeng, G. (2014). A necessary condition for a good binning algorithm in credit scoring. Applied Mathematical Sciences, 8(65), 3229-3242.
//' }
//'
//' @examples
//' \dontrun{
//' # Generate sample data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_cm(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result)
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_cm(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 // Input validation
 if (target.size() != feature.size()) {
   Rcpp::stop("Target and feature vectors must have the same length");
 }
 
 if (min_bins < 2 || max_bins < min_bins) {
   Rcpp::stop("Invalid bin constraints: min_bins must be at least 2 and max_bins must be greater than or equal to min_bins");
 }
 
 if (bin_cutoff <= 0 || bin_cutoff >= 1) {
   Rcpp::stop("bin_cutoff must be between 0 and 1");
 }
 
 if (max_n_prebins < max_bins) {
   Rcpp::stop("max_n_prebins must be greater than or equal to max_bins");
 }
 
 // Convert Rcpp vectors to std::vector
 std::vector<double> feature_vec = Rcpp::as<std::vector<double>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create and run the binner
 try {
   OptimalBinningNumericalCM binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins, convergence_threshold, max_iterations);
   binner.fit();
   
   // Get and return the results
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
