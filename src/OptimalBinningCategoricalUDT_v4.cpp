// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <functional>
#include <sstream>

using namespace Rcpp;

// Global constants for better readability and consistency
static constexpr double EPSILON = 1e-10;
static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
static constexpr const char* MISSING_VALUE = "__MISSING__";  // Special category for missing values

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Laplace smoothing for more robust probability estimates
inline std::pair<double, double> smoothed_proportions(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  // Apply Laplace (add-alpha) smoothing
  double smoothed_pos_rate = (positive_count + alpha) / (total_positive + alpha * 2);
  double smoothed_neg_rate = (negative_count + alpha) / (total_negative + alpha * 2);
  
  return {smoothed_pos_rate, smoothed_neg_rate};
}

// Calculate Weight of Evidence with Laplace smoothing
inline double calculate_woe(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  return safe_log(smoothed_pos_rate / smoothed_neg_rate);
}

// Calculate Information Value with Laplace smoothing
inline double calculate_iv(
    int positive_count, 
    int negative_count, 
    int total_positive, 
    int total_negative, 
    double alpha = LAPLACE_ALPHA) {
  
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
    positive_count, negative_count, total_positive, total_negative, alpha);
  
  double woe = safe_log(smoothed_pos_rate / smoothed_neg_rate);
  return (smoothed_pos_rate - smoothed_neg_rate) * woe;
}

// Calculate Jensen-Shannon divergence between two bins
inline double calculate_divergence(
    int bin1_pos, int bin1_neg, 
    int bin2_pos, int bin2_neg, 
    int total_pos, int total_neg) {
  
  // Jensen-Shannon divergence (symmetric KL divergence)
  auto [p1, n1] = smoothed_proportions(bin1_pos, bin1_neg, total_pos, total_neg);
  auto [p2, n2] = smoothed_proportions(bin2_pos, bin2_neg, total_pos, total_neg);
  
  // Average proportions
  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;
  
  // KL(P1 || P_avg) + KL(P2 || P_avg)
  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;
  
  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}

// Join vector of categories with uniqueness checking
inline std::string join_categories(const std::vector<std::string>& categories, 
                                   const std::string& separator) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];
  
  // Ensure uniqueness
  std::unordered_set<std::string> unique_cats;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_cats.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  // Join with separator
  std::ostringstream result;
  result << unique_vec[0];
  for (size_t i = 1; i < unique_vec.size(); ++i) {
    result << separator << unique_vec[i];
  }
  
  return result.str();
}
}

class OptimalBinningCategoricalUDT {
private:
  // Enhanced bin structure with uniqueness guarantee
  struct BinInfo {
    std::unordered_set<std::string> category_set;  // For fast uniqueness check
    std::vector<std::string> categories;           // For ordered storage
    double woe;
    double iv;
    int count;
    int count_pos;
    int count_neg;
    double event_rate;  // New field for event rate
    
    BinInfo() : woe(0.0), iv(0.0), count(0), count_pos(0), count_neg(0), event_rate(0.0) {
      categories.reserve(8);  // Pre-allocate for better performance
    }
    
    // Add a category ensuring uniqueness
    void add_category(const std::string& cat) {
      if (category_set.insert(cat).second) {  // Only add if not already present
        categories.push_back(cat);
      }
    }
    
    // Add a category with its counts
    void add_instance(const std::string& cat, int is_positive) {
      add_category(cat);
      count++;
      count_pos += is_positive;
      count_neg += (1 - is_positive);
      update_event_rate();
    }
    
    // Merge with another bin
    void merge_with(const BinInfo& other) {
      // Add all categories from other bin, ensuring uniqueness
      for (const auto& cat : other.categories) {
        add_category(cat);
      }
      
      count += other.count;
      count_pos += other.count_pos;
      count_neg += other.count_neg;
      update_event_rate();
    }
    
    // Update event rate
    void update_event_rate() {
      event_rate = count > 0 ? static_cast<double>(count_pos) / count : 0.0;
    }
    
    // Calculate metrics with Laplace smoothing
    void calculate_metrics(int total_pos, int total_neg) {
      woe = utils::calculate_woe(count_pos, count_neg, total_pos, total_neg);
      iv = utils::calculate_iv(count_pos, count_neg, total_pos, total_neg);
    }
    
    // Calculate statistical divergence from another bin
    double divergence_from(const BinInfo& other, int total_pos, int total_neg) const {
      return utils::calculate_divergence(
        count_pos, count_neg, other.count_pos, other.count_neg, total_pos, total_neg);
    }
  };
  
  // Class parameters
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;
  
  // Internal state
  std::vector<BinInfo> bins_;
  bool converged_;
  int iterations_;
  int total_pos_;
  int total_neg_;
  
  // Input validation with improved error messages
  void validate_inputs(const std::vector<std::string>& feature, const std::vector<int>& target) {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Input vectors cannot be empty.");
    }
    
    // Check target values and count positives/negatives
    bool has_zero = false;
    bool has_one = false;
    
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target vector must contain only 0 and 1.");
      
      // Early termination
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
    
    // Validate parameter ranges
    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1.");
    }
    if (max_bins_ < min_bins_) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins.");
    }
    if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins_ < min_bins_) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins.");
    }
  }
  
  // Initial binning with one bin per unique category
  void initial_binning(const std::vector<std::string>& feature, const std::vector<int>& target) {
    std::unordered_map<std::string, BinInfo> bin_map;
    total_pos_ = 0;
    total_neg_ = 0;
    
    // Process each observation
    for (size_t i = 0; i < feature.size(); ++i) {
      auto& bin = bin_map[feature[i]];
      bin.add_instance(feature[i], target[i]);
      total_pos_ += target[i];
      total_neg_ += (1 - target[i]);
    }
    
    // Transfer to bins vector
    bins_.clear();
    bins_.reserve(bin_map.size());
    for (auto& pair : bin_map) {
      bins_.push_back(std::move(pair.second));
    }
  }
  
  // Merge low-frequency bins with improved strategy
  void merge_low_frequency_bins() {
    // Calculate cutoff threshold
    int total_count = std::accumulate(bins_.begin(), bins_.end(), 0,
                                      [](int sum, const BinInfo& bin) { return sum + bin.count; });
    double cutoff_count = total_count * bin_cutoff_;
    
    // Sort bins by count (ascending)
    std::sort(bins_.begin(), bins_.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.count < b.count;
    });
    
    // Process bins, keeping those above threshold
    std::vector<BinInfo> new_bins;
    BinInfo low_freq_bin;
    
    for (auto& bin : bins_) {
      if (bin.count >= cutoff_count || (int)new_bins.size() < min_bins_) {
        new_bins.push_back(bin);
      } else {
        // Merge into low frequency bin
        low_freq_bin.merge_with(bin);
      }
    }
    
    // Add low frequency bin if not empty
    if (low_freq_bin.count > 0) {
      new_bins.push_back(low_freq_bin);
    }
    
    bins_ = std::move(new_bins);
  }
  
  // Calculate WoE and IV for all bins with Laplace smoothing
  void calculate_woe_iv() {
    for (auto& bin : bins_) {
      bin.calculate_metrics(total_pos_, total_neg_);
    }
  }
  
  // Calculate total IV across all bins
  double calculate_total_iv() const {
    return std::accumulate(bins_.begin(), bins_.end(), 0.0,
                           [](double sum, const BinInfo& bin) { return sum + std::fabs(bin.iv); });
  }
  
  // Ensure monotonicity by sorting bins by WoE
  void ensure_monotonicity() {
    std::sort(bins_.begin(), bins_.end(), [](const BinInfo& a, const BinInfo& b) {
      return a.woe < b.woe;
    });
  }
  
  // Find the most similar bins for merging based on statistical divergence
  std::pair<size_t, size_t> find_most_similar_bins() const {
    double min_divergence = std::numeric_limits<double>::max();
    size_t idx1 = 0, idx2 = 1;
    
    for (size_t i = 0; i < bins_.size(); ++i) {
      for (size_t j = i + 1; j < bins_.size(); ++j) {
        double div = bins_[i].divergence_from(bins_[j], total_pos_, total_neg_);
        
        // Prefer adjacent bins if divergence is similar
        if (j == i + 1) {
          div *= 0.95;  // Slight bias towards adjacent bins
        }
        
        if (div < min_divergence) {
          min_divergence = div;
          idx1 = i;
          idx2 = j;
        }
      }
    }
    
    return {idx1, idx2};
  }
  
  // Merge bins with improved strategy using statistical similarity
  void merge_bins() {
    while ((int)bins_.size() > max_bins_) {
      // Find most statistically similar bins
      auto [idx1, idx2] = find_most_similar_bins();
      
      // Ensure lower index first
      if (idx2 < idx1) std::swap(idx1, idx2);
      
      // Merge bins
      bins_[idx1].merge_with(bins_[idx2]);
      bins_[idx1].calculate_metrics(total_pos_, total_neg_);
      bins_.erase(bins_.begin() + idx2);
      
      // Recalculate WoE/IV after merge
      calculate_woe_iv();
    }
  }
  
public:
  // Constructor with improved defaults and documentation
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
  converged_(false), iterations_(0), total_pos_(0), total_neg_(0) {}
  
  // Main fitting method with improved logic
  void fit(const std::vector<std::string>& feature, const std::vector<int>& target) {
    validate_inputs(feature, target);
    
    // Count unique categories
    std::unordered_set<std::string> unique_cats(feature.begin(), feature.end());
    int ncat = static_cast<int>(unique_cats.size());
    
    // Adjust min_bins and max_bins based on unique categories
    max_bins_ = std::min(max_bins_, ncat);
    min_bins_ = std::min(min_bins_, max_bins_);
    
    // Initial binning (one bin per category)
    initial_binning(feature, target);
    
    // Special case: 1 or 2 unique levels
    if (ncat <= 2) {
      calculate_woe_iv();
      converged_ = true;
      iterations_ = 0;
      return;
    }
    
    // Merge low frequency bins
    merge_low_frequency_bins();
    calculate_woe_iv();
    ensure_monotonicity();
    
    // Main optimization loop
    double prev_total_iv = calculate_total_iv();
    converged_ = false;
    iterations_ = 0;
    
    while (!converged_ && iterations_ < max_iterations_) {
      // Merge bins if needed
      if ((int)bins_.size() > max_bins_) {
        merge_bins();
      } else {
        // If within min_bins and max_bins range, we're done
        if ((int)bins_.size() >= min_bins_) {
          converged_ = true;
          break;
        }
        
        // If we can't increase the number of bins, we're done
        // No splitting is performed to avoid artificial categories
        converged_ = true;
        break;
      }
      
      // Ensure monotonicity
      ensure_monotonicity();
      
      // Check convergence
      double total_iv = calculate_total_iv();
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold_) {
        converged_ = true;
      }
      
      prev_total_iv = total_iv;
      iterations_++;
    }
    
    // Final calculations
    calculate_woe_iv();
    ensure_monotonicity();
  }
  
  // Get results as Rcpp List with improved structure
  Rcpp::List get_woe_bin() const {
    // Prepare result vectors
    Rcpp::CharacterVector bin_names;
    Rcpp::NumericVector woe_values, iv_values, event_rates;
    Rcpp::IntegerVector counts, counts_pos, counts_neg;
    
    // Fill result vectors
    for (const auto& bin : bins_) {
      bin_names.push_back(utils::join_categories(bin.categories, bin_separator_));
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      event_rates.push_back(bin.event_rate);
    }
    
    // Calculate total IV
    double total_iv = calculate_total_iv();
    
    // Create sequential IDs
    Rcpp::NumericVector ids(bin_names.size());
    for (int i = 0; i < bin_names.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Return results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("event_rate") = event_rates,
      Rcpp::Named("converged") = converged_,
      Rcpp::Named("iterations") = iterations_,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};

//' @title Optimal Binning for Categorical Variables using a User-Defined Technique (UDT)
//'
//' @description
//' This function performs binning for categorical variables using a user-defined technique (UDT).
//' The algorithm creates bins with optimal predictive power (measured by Information Value)
//' while maintaining monotonicity of Weight of Evidence and avoiding the creation of artificial categories.
//' Enhanced with statistical robustness features like Laplace smoothing and Jensen-Shannon divergence.
//'
//' @param target Integer binary vector (0 or 1) representing the response variable.
//' @param feature Character vector representing the categories of the explanatory variable.
//' @param min_bins Minimum number of desired bins (default: 3).
//' @param max_bins Maximum number of desired bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations to consider a category as a separate bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins before the main binning step (default: 20).
//' @param bin_separator String used to separate names of categories grouped in the same bin (default: "%;%").
//' @param convergence_threshold Threshold for stopping criteria based on IV convergence (default: 1e-6).
//' @param max_iterations Maximum number of iterations in the optimization process (default: 1000).
//'
//' @return A list containing:
//' \itemize{
//'   \item id: Numeric identifiers for each bin.
//'   \item bin: String vector with bin names representing grouped categories.
//'   \item woe: Numeric vector with Weight of Evidence values for each bin.
//'   \item iv: Numeric vector with Information Value for each bin.
//'   \item count: Integer vector with the total count of observations in each bin.
//'   \item count_pos: Integer vector with the count of positive cases (target=1) in each bin.
//'   \item count_neg: Integer vector with the count of negative cases (target=0) in each bin.
//'   \item event_rate: Numeric vector with the proportion of positive cases in each bin.
//'   \item converged: Logical value indicating if the algorithm converged.
//'   \item iterations: Integer value indicating the number of optimization iterations executed.
//'   \item total_iv: The total Information Value of the binning solution.
//' }
//'
//' @details
//' ## Statistical Methodology
//' 
//' The UDT algorithm optimizes binning based on statistical concepts of Weight of Evidence 
//' and Information Value with Laplace smoothing for robustness:
//'
//' Weight of Evidence measures the predictive power of a bin:
//' \deqn{WoE_i = \ln\left(\frac{(n_{i+} + \alpha)/(n_+ + 2\alpha)}{(n_{i-} + \alpha)/(n_- + 2\alpha)}\right)}
//'
//' Where:
//' - \eqn{n_{i+}} is the number of positive cases (target=1) in bin i
//' - \eqn{n_{i-}} is the number of negative cases (target=0) in bin i
//' - \eqn{n_+} is the total number of positive cases
//' - \eqn{n_-} is the total number of negative cases
//' - \eqn{\alpha} is the Laplace smoothing parameter (default: 0.5)
//'
//' Information Value measures the overall predictive power:
//' \deqn{IV_i = \left(\frac{n_{i+}}{n_+} - \frac{n_{i-}}{n_-}\right) \times WoE_i}
//' \deqn{IV_{total} = \sum_{i=1}^{k} |IV_i|}
//'
//' ## Algorithm Steps
//'
//' 1. Input validation and creation of initial bins (one bin per unique category)
//'    - Special handling for variables with 1-2 unique levels
//' 2. Merge low-frequency categories below the bin_cutoff threshold
//' 3. Calculate WoE and IV for each bin using Laplace smoothing
//' 4. Iteratively merge similar bins based on Jensen-Shannon divergence until constraints are satisfied
//' 5. Ensure WoE monotonicity across bins for better interpretability
//' 6. The process continues until convergence or max_iterations is reached
//'
//' The algorithm uses Jensen-Shannon divergence to measure statistical similarity between bins:
//' \deqn{JS(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)}
//'
//' Where:
//' - \eqn{KL} is the Kullback-Leibler divergence
//' - \eqn{M = \frac{1}{2}(P+Q)} is the midpoint distribution
//' - \eqn{P} and \eqn{Q} are the event rate distributions of two bins
//'
//' ## Important Notes
//'
//' - Missing values in the feature are handled as a special category
//' - The algorithm naturally handles sparse data through Laplace smoothing
//' - No splitting is performed to avoid creating artificial category names
//' - Uniqueness of categories within bins is guaranteed
//'
//' @references
//' \itemize{
//'   \item Beltrán, C., et al. (2022). Weight of Evidence (WoE) and Information Value (IV): A novel implementation for predictive modeling in credit scoring. Expert Systems with Applications, 183, 115351.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//' result <- optimal_binning_categorical_udt(target, feature)
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
 try {
   // Handle missing values in feature
   std::vector<std::string> feature_vec;
   feature_vec.reserve(feature.size());
   
   for (R_xlen_t i = 0; i < feature.size(); ++i) {
     if (feature[i] == NA_STRING) {
       feature_vec.push_back(MISSING_VALUE);
     } else {
       feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
     }
   }
   
   // Check for missing values in target
   std::vector<int> target_vec;
   target_vec.reserve(target.size());
   
   for (R_xlen_t i = 0; i < target.size(); ++i) {
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values");
     }
     target_vec.push_back(target[i]);
   }
   
   OptimalBinningCategoricalUDT binning(
       min_bins, max_bins, bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold, max_iterations
   );
   
   binning.fit(feature_vec, target_vec);
   return binning.get_woe_bin();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}
