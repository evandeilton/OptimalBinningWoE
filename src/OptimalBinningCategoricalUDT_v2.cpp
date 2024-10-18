#include <Rcpp.h>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

// For convenience
using namespace Rcpp;

// Define a structure to hold bin information
struct BinInfo {
  std::vector<std::string> categories;
  double woe;
  double iv;
  int count;
  int count_pos;
  int count_neg;
  
  // Constructor to initialize counts
  BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0) {}
};

// OptimalBinningCategoricalUDT Class
class OptimalBinningCategoricalUDT {
private:
  // Parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  // Data
  std::vector<std::string> feature;
  std::vector<int> target;
  std::vector<BinInfo> final_bins;
  bool converged;
  int iterations_run;
  
  // Helper Functions
  void validate_input_parameters(const size_t unique_categories) {
    min_bins = std::max(2, std::min(static_cast<int>(unique_categories), min_bins));
    max_bins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_bins));
    bin_cutoff = std::max(0.0, std::min(1.0, bin_cutoff));
    max_n_prebins = std::max(min_bins, std::min(static_cast<int>(unique_categories), max_n_prebins));
  }
  
  void validate_input_data() const {
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target vectors must not be empty.");
    }
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    std::unordered_set<int> unique_targets(target.begin(), target.end());
    if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
      throw std::invalid_argument("Target variable must be binary (0 and 1).");
    }
  }
  
  double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) const {
    const double epsilon = 1e-10;
    double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
    double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
    return std::log(event_rate / non_event_rate);
  }
  
  double calculate_iv(int count_pos, int count_neg, int total_pos, int total_neg) const {
    const double epsilon = 1e-10;
    double event_rate = std::max(epsilon, static_cast<double>(count_pos) / total_pos);
    double non_event_rate = std::max(epsilon, static_cast<double>(count_neg) / total_neg);
    return (event_rate - non_event_rate) * std::log(event_rate / non_event_rate);
  }
  
  void compute_initial_bins(std::vector<BinInfo>& initial_bins, int& total_pos, int& total_neg) {
    std::unordered_map<std::string, BinInfo> temp_bins;
    total_pos = 0;
    total_neg = 0;
    for (size_t i = 0; i < feature.size(); ++i) {
      const std::string& cat = feature[i];
      int tgt = target[i];
      auto& bin = temp_bins[cat];
      if (bin.categories.empty()) {
        bin.categories.push_back(cat);
      }
      bin.count++;
      bin.count_pos += tgt;
      bin.count_neg += (1 - tgt);
      
      total_pos += tgt;
      total_neg += (1 - tgt);
    }
    
    initial_bins.clear();
    for (auto& pair : temp_bins) {
      initial_bins.push_back(std::move(pair.second));
    }
  }
  
  void calculate_woe_iv(std::vector<BinInfo>& bins, int total_pos, int total_neg) {
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculate_iv(bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
  }
  
  bool is_monotonic(const std::vector<BinInfo>& bins) const {
    if (bins.size() <= 2) return true;
    bool increasing = bins[1].woe > bins[0].woe;
    for (size_t i = 2; i < bins.size(); ++i) {
      if ((increasing && bins[i].woe < bins[i-1].woe) || (!increasing && bins[i].woe > bins[i-1].woe)) {
        return false;
      }
    }
    return true;
  }
  
  void merge_bins(std::vector<BinInfo>& bins, int idx1, int idx2, int total_pos, int total_neg) {
    BinInfo& bin1 = bins[idx1];
    BinInfo& bin2 = bins[idx2];
    bin1.categories.insert(bin1.categories.end(), bin2.categories.begin(), bin2.categories.end());
    bin1.count += bin2.count;
    bin1.count_pos += bin2.count_pos;
    bin1.count_neg += bin2.count_neg;
    bin1.woe = calculate_woe(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
    bin1.iv = calculate_iv(bin1.count_pos, bin1.count_neg, total_pos, total_neg);
    
    bins.erase(bins.begin() + idx2);
  }
  
public:
  // Constructor
  OptimalBinningCategoricalUDT(int min_bins_ = 2, int max_bins_ = 5, double bin_cutoff_ = 0.05, 
                               int max_n_prebins_ = 20, std::string bin_separator_ = "%;%",
                               double convergence_threshold_ = 1e-6, int max_iterations_ = 1000) :
  min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_),
  bin_separator(bin_separator_), convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
  converged(false), iterations_run(0) {}
  
  // Fit Function
  void fit(const std::vector<std::string>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    
    std::vector<BinInfo> initial_bins;
    int total_pos = 0, total_neg = 0;
    compute_initial_bins(initial_bins, total_pos, total_neg);
    size_t unique_categories = initial_bins.size();
    
    validate_input_parameters(unique_categories);
    validate_input_data();
    
    calculate_woe_iv(initial_bins, total_pos, total_neg);
    
    // If ncat <= max_bins, no optimization needed
    if (unique_categories <= static_cast<size_t>(max_bins)) {
      final_bins = std::move(initial_bins);
      converged = true;
      iterations_run = 0;
      return;
    }
    
    std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.count < b.count;
    });
    
    while (initial_bins.size() > static_cast<size_t>(max_n_prebins)) {
      merge_bins(initial_bins, 0, 1, total_pos, total_neg);
      std::sort(initial_bins.begin(), initial_bins.end(), [](const BinInfo& a, const BinInfo& b) {
        return a.count < b.count;
      });
    }
    
    final_bins = std::move(initial_bins);
    
    std::sort(final_bins.begin(), final_bins.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
    
    double prev_total_iv = 0.0;
    for (const auto& bin : final_bins) {
      prev_total_iv += bin.iv;
    }
    
    converged = false;
    iterations_run = 0;
    
    while (final_bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_idx = -1;
      for (size_t i = 0; i < final_bins.size() - 1; ++i) {
        double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
        if (iv_diff < min_iv_diff) {
          min_iv_diff = iv_diff;
          merge_idx = static_cast<int>(i);
        }
      }
      
      if (merge_idx == -1) break;
      
      merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
      
      double total_iv = 0.0;
      for (const auto& bin : final_bins) {
        total_iv += bin.iv;
      }
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        converged = true;
        break;
      }
      
      prev_total_iv = total_iv;
      iterations_run++;
    }
    
    // Ensure monotonicity if possible
    while (!is_monotonic(final_bins) && final_bins.size() > static_cast<size_t>(min_bins) && iterations_run < max_iterations) {
      double min_iv_diff = std::numeric_limits<double>::max();
      int merge_idx = -1;
      for (size_t i = 0; i < final_bins.size() - 1; ++i) {
        double iv_diff = std::abs(final_bins[i].iv - final_bins[i+1].iv);
        if (iv_diff < min_iv_diff) {
          min_iv_diff = iv_diff;
          merge_idx = static_cast<int>(i);
        }
      }
      
      if (merge_idx == -1) break;
      
      merge_bins(final_bins, merge_idx, merge_idx + 1, total_pos, total_neg);
      iterations_run++;
    }
    
    if (iterations_run >= max_iterations) {
      Rcpp::warning("Maximum iterations reached. Optimal binning may not have converged.");
    }
  }
  
  // Getter for output
  Rcpp::List get_woe_bin() const {
    std::vector<std::string> bin_labels;
    std::vector<double> woe_vals;
    std::vector<double> iv_vals;
    std::vector<int> counts;
    std::vector<int> count_pos;
    std::vector<int> count_neg;
    
    for (const auto& bin : final_bins) {
      std::string label = std::accumulate(std::next(bin.categories.begin()), bin.categories.end(),
                                          bin.categories[0],
                                                        [this](std::string a, std::string b) {
                                                          return std::move(a) + this->bin_separator + b;
                                                        });
      bin_labels.push_back(label);
      woe_vals.push_back(bin.woe);
      iv_vals.push_back(bin.iv);
      counts.push_back(bin.count);
      count_pos.push_back(bin.count_pos);
      count_neg.push_back(bin.count_neg);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_vals,
      Rcpp::Named("iv") = iv_vals,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

//' @title
//' Optimal Binning for Categorical Variables using Unsupervised Decision Tree (UDT)
//'
//' @description
//' This function performs optimal binning for categorical variables using an Unsupervised Decision Tree (UDT) approach,
//' which combines Weight of Evidence (WOE) and Information Value (IV) methods.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum frequency for a category to be considered as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before merging (default: 20).
//' @param bin_separator Separator used for merging category names (default: "%;%").
//' @param convergence_threshold Threshold for convergence of the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//'
//' @return A list containing bin information:
//' \item{bins}{A character vector of bin labels}
//' \item{woe}{A numeric vector of WOE values for each bin}
//' \item{iv}{A numeric vector of IV values for each bin}
//' \item{count}{An integer vector of total counts for each bin}
//' \item{count_pos}{An integer vector of positive target counts for each bin}
//' \item{count_neg}{An integer vector of negative target counts for each bin}
//' \item{converged}{A logical indicating whether the algorithm converged}
//' \item{iterations}{An integer indicating the number of iterations run}
//'
//' @details
//' The algorithm performs the following steps:
//' 1. Input validation and preprocessing
//' 2. Initial binning based on unique categories
//' 3. Merge bins to respect max_n_prebins
//' 4. Iterative merging of bins based on minimum IV difference
//' 5. Ensure monotonicity of WOE values across bins (if possible)
//' 6. Respect min_bins and max_bins constraints
//'
//' The Weight of Evidence (WOE) is calculated as:
//'
//' WOE = ln((Distribution of Good) / (Distribution of Bad))
//'
//' The Information Value (IV) for each bin is calculated as:
//'
//' IV = (Distribution of Good - Distribution of Bad) * WOE
//'
//' The algorithm aims to find an optimal binning solution while respecting the specified constraints.
//' It uses a convergence threshold and maximum number of iterations to ensure stability and prevent infinite loops.
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
//' result <- optimal_binning_categorical_udt(target, feature)
//'
//' # View results
//' print(result)
//' }
//'
//' @export
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
 // Convert R vectors to C++ vectors
 std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
 std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
 
 // Create OptimalBinningCategoricalUDT object
 OptimalBinningCategoricalUDT binning(
     min_bins, max_bins, bin_cutoff, max_n_prebins,
     bin_separator, convergence_threshold, max_iterations
 );
 
 try {
   // Perform optimal binning
   binning.fit(feature_vec, target_vec);
   
   // Get results
   return binning.get_woe_bin();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}