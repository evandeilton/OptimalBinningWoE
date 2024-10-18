#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

class OptimalBinningNumericalBB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  bool is_monotonic;
  double convergence_threshold;
  int max_iterations;
  
  static constexpr double EPSILON = 1e-10;
  
  struct Bin {
    double lower;
    double upper;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  bool converged;
  int iterations_run;
  
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be greater than or equal to min_bins.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be greater than 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be greater than 0.");
    }
    
    // Check if all values in the feature are the same
    if (std::adjacent_find(feature.begin(), feature.end(), std::not_equal_to<>()) == feature.end()) {
      throw std::invalid_argument("All values in the feature are the same. Unable to perform binning.");
    }
  }
  
  void prebinning() {
    // Divide data into quantile-based pre-bins
    std::vector<double> unique_values = feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    // If the number of unique values is less than or equal to min_bins, use them directly
    if (unique_values.size() <= static_cast<size_t>(min_bins)) {
      bins.clear();
      for (size_t i = 0; i < unique_values.size(); ++i) {
        Bin bin;
        bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
        bin.upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
        bin.count_pos = 0;
        bin.count_neg = 0;
        bins.push_back(bin);
      }
    } else {
      int n_prebins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
      std::vector<double> quantiles;
      for (int i = 1; i < n_prebins; ++i) {
        quantiles.push_back(quantile(feature, i / double(n_prebins)));
      }
      quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
      
      // Initialize bins
      bins.clear();
      bins.resize(quantiles.size() + 1);
      for (size_t i = 0; i < bins.size(); ++i) {
        if (i == 0) {
          bins[i].lower = -std::numeric_limits<double>::infinity();
          bins[i].upper = quantiles[i];
        } else if (i == bins.size() - 1) {
          bins[i].lower = quantiles[i - 1];
          bins[i].upper = std::numeric_limits<double>::infinity();
        } else {
          bins[i].lower = quantiles[i - 1];
          bins[i].upper = quantiles[i];
        }
        bins[i].count_pos = 0;
        bins[i].count_neg = 0;
      }
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int tgt = target[i];
      for (auto& bin : bins) {
        if (val > bin.lower - EPSILON && val <= bin.upper + EPSILON) {
          if (tgt == 1) {
            bin.count_pos++;
          } else {
            bin.count_neg++;
          }
          break;
        }
      }
    }
  }
  
  double quantile(const std::vector<double>& data, double q) {
    std::vector<double> temp = data;
    std::sort(temp.begin(), temp.end());
    size_t idx = std::floor(q * (temp.size() - 1));
    return temp[idx];
  }
  
  void merge_rare_bins() {
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const Bin& bin) { return sum + bin.count_pos + bin.count_neg; });
    double cutoff_count = bin_cutoff * total_count;
    
    for (auto it = bins.begin(); it != bins.end(); ) {
      if (it->count_pos + it->count_neg < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
        if (it != bins.begin()) {
          auto prev = std::prev(it);
          prev->upper = it->upper;
          prev->count_pos += it->count_pos;
          prev->count_neg += it->count_neg;
          it = bins.erase(it);
        } else if (std::next(it) != bins.end()) {
          auto next = std::next(it);
          next->lower = it->lower;
          next->count_pos += it->count_pos;
          next->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          ++it;
        }
      } else {
        ++it;
      }
    }
  }
  
  void compute_woe_iv() {
    int total_pos = std::accumulate(bins.begin(), bins.end(), 0,
                                    [](int sum, const Bin& bin) { return sum + bin.count_pos; });
    int total_neg = std::accumulate(bins.begin(), bins.end(), 0,
                                    [](int sum, const Bin& bin) { return sum + bin.count_neg; });
    
    for (auto& bin : bins) {
      double dist_pos = (bin.count_pos + 0.5) / (total_pos + 1);
      double dist_neg = (bin.count_neg + 0.5) / (total_neg + 1);
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  void enforce_monotonicity() {
    bool increasing = std::is_sorted(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) { return a.woe < b.woe; });
    bool decreasing = std::is_sorted(bins.begin(), bins.end(),
                                     [](const Bin& a, const Bin& b) { return a.woe > b.woe; });
    
    if (!increasing && !decreasing) {
      for (auto it = std::next(bins.begin()); it != bins.end() && bins.size() > static_cast<size_t>(min_bins); ) {
        if ((it->woe < std::prev(it)->woe - EPSILON && decreasing) ||
            (it->woe > std::prev(it)->woe + EPSILON && increasing)) {
          std::prev(it)->upper = it->upper;
          std::prev(it)->count_pos += it->count_pos;
          std::prev(it)->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          ++it;
        }
      }
      compute_woe_iv();
    }
  }
  
public:
  OptimalBinningNumericalBB(const std::vector<double>& feature_,
                            const std::vector<int>& target_,
                            int min_bins_ = 2,
                            int max_bins_ = 5,
                            double bin_cutoff_ = 0.05,
                            int max_n_prebins_ = 20,
                            bool is_monotonic_ = true,
                            double convergence_threshold_ = 1e-6,
                            int max_iterations_ = 1000)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), is_monotonic(is_monotonic_),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
      converged(false), iterations_run(0) {
    validate_inputs();
  }
  
  Rcpp::List fit() {
    prebinning();
    merge_rare_bins();
    compute_woe_iv();
    if (is_monotonic) {
      enforce_monotonicity();
    }
    
    double prev_total_iv = std::numeric_limits<double>::infinity();
    iterations_run = 0;
    converged = false;
    
    while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
      auto min_iv_it = std::min_element(bins.begin(), bins.end(),
                                        [](const Bin& a, const Bin& b) { return a.iv < b.iv; });
      
      if (min_iv_it != bins.begin()) {
        auto prev = std::prev(min_iv_it);
        prev->upper = min_iv_it->upper;
        prev->count_pos += min_iv_it->count_pos;
        prev->count_neg += min_iv_it->count_neg;
        bins.erase(min_iv_it);
      } else {
        auto next = std::next(min_iv_it);
        next->lower = min_iv_it->lower;
        next->count_pos += min_iv_it->count_pos;
        next->count_neg += min_iv_it->count_neg;
        bins.erase(min_iv_it);
      }
      compute_woe_iv();
      if (is_monotonic) {
        enforce_monotonicity();
      }
      
      double total_iv = std::accumulate(bins.begin(), bins.end(), 0.0,
                                        [](double sum, const Bin& bin) { return sum + bin.iv; });
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = total_iv;
      iterations_run++;
    }
    
    // Prepare outputs
    std::vector<std::string> bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector counts;
    Rcpp::IntegerVector counts_pos;
    Rcpp::IntegerVector counts_neg;
    Rcpp::NumericVector cutpoints;
    
    // Prepare binning table
    for (const auto& bin : bins) {
      std::string bin_label = "(" + (std::isinf(bin.lower) ? "-Inf" : std::to_string(bin.lower)) +
        ";" + (std::isinf(bin.upper) ? "+Inf" : std::to_string(bin.upper)) + "]";
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count_pos + bin.count_neg);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      if (!std::isinf(bin.upper)) {
        cutpoints.push_back(bin.upper);
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bins") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

//' @title
//' Optimal Binning for Numerical Variables using Branch and Bound
//'
//' @description
//' This function implements an optimal binning algorithm for numerical variables using a Branch and Bound approach with Weight of Evidence (WoE) and Information Value (IV) criteria.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency of observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins for initial quantile-based discretization (default: 20).
//' @param is_monotonic Boolean indicating whether to enforce monotonicity of WoE across bins (default: TRUE).
//' @param convergence_threshold Threshold for convergence of total IV (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//'
//' @return A list containing the following elements:
//' \item{bins}{A character vector of bin labels.}
//' \item{woe}{A numeric vector of Weight of Evidence values for each bin.}
//' \item{iv}{A numeric vector of Information Value for each bin.}
//' \item{count}{An integer vector of total count of observations in each bin.}
//' \item{count_pos}{An integer vector of count of positive observations in each bin.}
//' \item{count_neg}{An integer vector of count of negative observations in each bin.}
//' \item{cutpoints}{A numeric vector of cutpoints for generating the bins.}
//' \item{converged}{A boolean indicating whether the algorithm converged.}
//' \item{iterations}{An integer indicating the number of iterations run.}
//'
//' @details
//' The optimal binning algorithm for numerical variables uses a Branch and Bound approach with Weight of Evidence (WoE) and Information Value (IV) to create bins that maximize the predictive power of the feature while maintaining interpretability.
//'
//' The algorithm follows these steps:
//' 1. Initial discretization using quantile-based binning
//' 2. Merging of rare bins
//' 3. Calculation of WoE and IV for each bin
//' 4. Enforcing monotonicity of WoE across bins (if is_monotonic is TRUE)
//' 5. Adjusting the number of bins to be within the specified range using a Branch and Bound approach
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
//' The Branch and Bound approach iteratively merges bins with the lowest IV contribution while respecting the constraints on the number of bins and minimum bin frequency. This process ensures that the resulting binning maximizes the total IV while maintaining the desired number of bins.
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
//' result <- optimal_binning_numerical_bb(target, feature, min_bins = 3, max_bins = 5)
//'
//' # View binning results
//' print(result)
//' }
//'
//' @references
//' \itemize{
//'   \item Farooq, B., & Miller, E. J. (2015). Optimal Binning for Continuous Variables in Credit Scoring. Journal of Risk Model Validation, 9(1), 1-21.
//'   \item Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey. GESTS International Transactions on Computer Science and Engineering, 32(1), 47-58.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_bb(
   Rcpp::IntegerVector target,
   Rcpp::NumericVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   bool is_monotonic = true,
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 try {
   OptimalBinningNumericalBB obb(
       Rcpp::as<std::vector<double>>(feature),
       Rcpp::as<std::vector<int>>(target),
       min_bins,
       max_bins,
       bin_cutoff,
       max_n_prebins,
       is_monotonic,
       convergence_threshold,
       max_iterations
   );
   return obb.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
