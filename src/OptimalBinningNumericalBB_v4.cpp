#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <numeric>

using namespace Rcpp;

/**
 * Core class implementing Optimal Binning for numerical variables using Branch and Bound algorithm.
 * 
 * This class transforms continuous numerical variables into optimal discrete bins 
 * based on their relationship with a binary target variable, maximizing predictive power
 * while maintaining statistical stability and interpretability constraints.
 */
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
  
  // Small constant to handle floating-point comparison
  static constexpr double EPSILON = 1e-10;
  
  // Structure to represent a bin with its properties
  struct Bin {
    double lower;            // Lower bound (inclusive)
    double upper;            // Upper bound (inclusive)
    int count_pos;           // Count of positive examples
    int count_neg;           // Count of negative examples
    double woe;              // Weight of Evidence value
    double iv;               // Information Value contribution
  };
  
  std::vector<Bin> bins;     // Container for all bins
  bool converged;            // Flag indicating algorithm convergence
  int iterations_run;        // Count of iterations performed
  
  /**
   * Validate input arguments for correctness and consistency
   * Throws std::invalid_argument if validation fails
   */
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length.");
    }
    
    // Check for valid binary target values (0 or 1)
    for (auto& val : target) {
      if (val != 0 && val != 1) {
        throw std::invalid_argument("Target values must be binary (0 or 1).");
      }
    }
    
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff < 0 || bin_cutoff > 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins.");
    }
    if (convergence_threshold <= 0) {
      throw std::invalid_argument("convergence_threshold must be > 0.");
    }
    if (max_iterations <= 0) {
      throw std::invalid_argument("max_iterations must be > 0.");
    }
  }
  
  /**
   * Handle the special case where feature has <= 2 unique values
   * This creates bins directly without optimization
   * 
   * @param unique_values Vector of unique values in the feature
   */
  void handle_two_or_fewer_unique_values(const std::vector<double>& unique_values) {
    bins.clear();
    // Construct bins based on unique values
    for (size_t i = 0; i < unique_values.size(); ++i) {
      Bin bin;
      bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
      bin.upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
      bin.count_pos = 0;
      bin.count_neg = 0;
      bin.woe = 0.0;
      bin.iv = 0.0;
      bins.push_back(bin);
    }
    
    // Assign observations to bins
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      int tgt = target[i];
      
      // With only 1 or 2 bins, linear scan is efficient enough
      for (auto &bin : bins) {
        if (val > bin.lower - EPSILON && val <= bin.upper + EPSILON) {
          if (tgt == 1) bin.count_pos++;
          else bin.count_neg++;
          break;
        }
      }
    }
    
    compute_woe_iv();
    converged = true;
    iterations_run = 0;
  }
  
  /**
   * Compute a quantile from a sorted vector
   * 
   * @param data Vector of values
   * @param q Quantile value between 0 and 1
   * @return The q-th quantile value
   */
  double quantile(const std::vector<double>& data, double q) {
    if (data.empty()) return 0.0;
    
    std::vector<double> temp = data;
    std::sort(temp.begin(), temp.end());
    
    if (q <= 0.0) return temp.front();
    if (q >= 1.0) return temp.back();
    
    // Calculate index with interpolation
    double idx_exact = q * (temp.size() - 1);
    size_t idx_lower = static_cast<size_t>(std::floor(idx_exact));
    size_t idx_upper = static_cast<size_t>(std::ceil(idx_exact));
    
    // Handle edge cases
    if (idx_lower == idx_upper) return temp[idx_lower];
    
    // Linear interpolation
    double weight_upper = idx_exact - idx_lower;
    double weight_lower = 1.0 - weight_upper;
    
    return weight_lower * temp[idx_lower] + weight_upper * temp[idx_upper];
  }
  
  /**
   * Initial binning step using quantiles to create starting bins
   * This provides a good starting point for optimization
   */
  void prebinning() {
    // Handle missing values by excluding them
    std::vector<double> clean_feature;
    std::vector<int> clean_target;
    
    for (size_t i = 0; i < feature.size(); ++i) {
      if (!std::isnan(feature[i])) {
        clean_feature.push_back(feature[i]);
        clean_target.push_back(target[i]);
      }
    }
    
    if (clean_feature.empty()) {
      throw std::invalid_argument("No valid non-NA values in feature.");
    }
    
    // Get unique values
    std::vector<double> unique_values = clean_feature;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    // Special case for few unique values
    if (unique_values.size() <= 2) {
      handle_two_or_fewer_unique_values(unique_values);
      return;
    }
    
    // If limited unique values, create a bin for each value
    if (unique_values.size() <= static_cast<size_t>(min_bins)) {
      bins.clear();
      for (size_t i = 0; i < unique_values.size(); ++i) {
        Bin bin;
        bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
        bin.upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
        bin.count_pos = 0;
        bin.count_neg = 0;
        bin.woe = 0.0;
        bin.iv = 0.0;
        bins.push_back(bin);
      }
    } else {
      // Use quantile-based initial binning for better distribution
      int n_prebins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
      
      std::vector<double> quantiles;
      for (int i = 1; i < n_prebins; ++i) {
        double q = static_cast<double>(i) / n_prebins;
        double qval = quantile(clean_feature, q);
        quantiles.push_back(qval);
      }
      
      // Remove duplicate quantiles (can happen with skewed distributions)
      std::sort(quantiles.begin(), quantiles.end());
      quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
      
      // Create bins based on quantiles
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
        bins[i].woe = 0.0;
        bins[i].iv = 0.0;
      }
    }
    
    // Extract upper boundaries for binary search
    std::vector<double> uppers;
    uppers.reserve(bins.size());
    for (auto &b : bins) {
      uppers.push_back(b.upper);
    }
    
    // Assign observations to bins using binary search for performance
    for (size_t i = 0; i < clean_feature.size(); ++i) {
      double val = clean_feature[i];
      int tgt = clean_target[i];
      
      // Binary search for the correct bin:
      // Find the first bin with upper boundary >= value
      auto it = std::lower_bound(uppers.begin(), uppers.end(), val + EPSILON);
      size_t idx = it - uppers.begin();
      
      // Count by target value
      if (tgt == 1) {
        bins[idx].count_pos++;
      } else {
        bins[idx].count_neg++;
      }
    }
  }
  
  /**
   * Merge bins with frequency below the specified cutoff
   * This ensures each bin has sufficient statistical reliability
   */
  void merge_rare_bins() {
    // Calculate total count across all bins
    int total_count = 0;
    for (auto &bin : bins) {
      total_count += (bin.count_pos + bin.count_neg);
    }
    
    double cutoff_count = bin_cutoff * total_count;
    
    // Iteratively merge rare bins with neighbors
    for (auto it = bins.begin(); it != bins.end();) {
      int bin_count = it->count_pos + it->count_neg;
      
      // Check if bin is too small and we can still merge (respecting min_bins)
      if (bin_count < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
        if (it != bins.begin()) {
          // Merge with previous bin (preferred)
          auto prev = std::prev(it);
          prev->upper = it->upper;
          prev->count_pos += it->count_pos;
          prev->count_neg += it->count_neg;
          it = bins.erase(it);
        } else if (std::next(it) != bins.end()) {
          // Merge with next bin if this is the first bin
          auto nxt = std::next(it);
          nxt->lower = it->lower;
          nxt->count_pos += it->count_pos;
          nxt->count_neg += it->count_neg;
          it = bins.erase(it);
        } else {
          // Edge case: only one bin or no valid merge candidate
          ++it;
        }
      } else {
        ++it;
      }
    }
  }
  
  /**
   * Compute Weight of Evidence (WoE) and Information Value (IV) for each bin
   * 
   * WoE = ln((proportion of positives) / (proportion of negatives))
   * IV = (proportion of positives - proportion of negatives) * WoE
   */
  void compute_woe_iv() {
    // Calculate totals
    int total_pos = 0;
    int total_neg = 0;
    
    for (auto &bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    // Apply Laplace smoothing to avoid division by zero
    // Using a Bayesian-inspired approach with small pseudo-counts
    double pos_denom = total_pos + bins.size() * 0.5;
    double neg_denom = total_neg + bins.size() * 0.5;
    
    for (auto &bin : bins) {
      // Calculate proportions with smoothing
      double dist_pos = (bin.count_pos + 0.5) / pos_denom;
      double dist_neg = (bin.count_neg + 0.5) / neg_denom;
      
      // Calculate WoE and IV
      bin.woe = std::log(dist_pos / dist_neg);
      bin.iv = (dist_pos - dist_neg) * bin.woe;
    }
  }
  
  /**
   * Enforce monotonicity in Weight of Evidence across bins
   * This improves interpretability and stability of the binning
   */
  void enforce_monotonicity() {
    if (bins.size() <= 1) return;
    
    // Check if already monotonic (either increasing or decreasing)
    bool is_increasing = true;
    bool is_decreasing = true;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].woe < bins[i-1].woe - EPSILON) is_increasing = false;
      if (bins[i].woe > bins[i-1].woe + EPSILON) is_decreasing = false;
    }
    
    // If already monotonic, do nothing
    if (is_increasing || is_decreasing) return;
    
    // Determine preferred direction (maximize total IV)
    double iv_increase = 0.0;
    double iv_decrease = 0.0;
    
    // Test both directions by simulating merges
    std::vector<Bin> temp_bins = bins;
    
    // Simulate increasing monotonicity
    for (size_t i = 1; i < temp_bins.size(); ++i) {
      if (temp_bins[i].woe < temp_bins[i-1].woe - EPSILON) {
        // Merge current bin into previous
        temp_bins[i-1].upper = temp_bins[i].upper;
        temp_bins[i-1].count_pos += temp_bins[i].count_pos;
        temp_bins[i-1].count_neg += temp_bins[i].count_neg;
        temp_bins.erase(temp_bins.begin() + i);
        i--; // Adjust index after erase
      }
    }
    
    // Recalculate IV for increasing scenario
    for (auto &bin : temp_bins) {
      iv_increase += bin.iv;
    }
    
    // Reset and simulate decreasing monotonicity
    temp_bins = bins;
    for (size_t i = 1; i < temp_bins.size(); ++i) {
      if (temp_bins[i].woe > temp_bins[i-1].woe + EPSILON) {
        // Merge current bin into previous
        temp_bins[i-1].upper = temp_bins[i].upper;
        temp_bins[i-1].count_pos += temp_bins[i].count_pos;
        temp_bins[i-1].count_neg += temp_bins[i].count_neg;
        temp_bins.erase(temp_bins.begin() + i);
        i--; // Adjust index after erase
      }
    }
    
    // Recalculate IV for decreasing scenario
    for (auto &bin : temp_bins) {
      iv_decrease += bin.iv;
    }
    
    // Choose direction with higher total IV
    bool prefer_increasing = (iv_increase >= iv_decrease);
    
    // Apply monotonicity in chosen direction
    for (auto it = std::next(bins.begin()); it != bins.end() && bins.size() > static_cast<size_t>(min_bins); ) {
      if ((prefer_increasing && it->woe < std::prev(it)->woe - EPSILON) || 
          (!prefer_increasing && it->woe > std::prev(it)->woe + EPSILON)) {
        // Merge current bin into previous
        std::prev(it)->upper = it->upper;
        std::prev(it)->count_pos += it->count_pos;
        std::prev(it)->count_neg += it->count_neg;
        it = bins.erase(it);
      } else {
        ++it;
      }
    }
    
    // Recompute WoE/IV after merges
    compute_woe_iv();
  }
  
public:
  /**
   * Constructor for the OptimalBinningNumericalBB algorithm
   * 
   * @param feature_ Numeric vector of feature values to be binned
   * @param target_ Binary vector (0/1) representing the target variable
   * @param min_bins_ Minimum number of bins to generate
   * @param max_bins_ Maximum number of bins to generate
   * @param bin_cutoff_ Minimum frequency fraction for each bin
   * @param max_n_prebins_ Maximum number of pre-bins before optimization
   * @param is_monotonic_ Whether to enforce monotonicity in WoE
   * @param convergence_threshold_ Convergence threshold for IV change
   * @param max_iterations_ Maximum number of iterations allowed
   */
  OptimalBinningNumericalBB(const std::vector<double> &feature_,
                            const std::vector<int> &target_,
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
  
  /**
   * Execute the optimal binning algorithm and return results
   * 
   * @return Rcpp::List containing bin information, WoE, IV, and other metrics
   */
  Rcpp::List fit() {
    // Step 1: Initial prebinning
    prebinning();
    
    if (!converged) {  // Only proceed if not already converged (e.g., few unique values)
      // Step 2: Merge rare bins
      merge_rare_bins();
      compute_woe_iv();
      
      // Step 3: Enforce monotonicity if requested
      if (is_monotonic) {
        enforce_monotonicity();
      }
      
      double prev_total_iv = std::numeric_limits<double>::infinity();
      iterations_run = 0;
      
      // Step 4: Branch and Bound optimization
      // Iteratively merge bins with smallest IV until max_bins constraint is met
      while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
        // Find bin with minimum IV contribution
        auto min_iv_it = std::min_element(bins.begin(), bins.end(),
                                          [](const Bin &a, const Bin &b) { return a.iv < b.iv; });
        
        // Merge the minimum IV bin with an adjacent bin
        if (min_iv_it != bins.begin()) {
          // Prefer merging with previous bin
          auto prev = std::prev(min_iv_it);
          prev->upper = min_iv_it->upper;
          prev->count_pos += min_iv_it->count_pos;
          prev->count_neg += min_iv_it->count_neg;
          bins.erase(min_iv_it);
        } else {
          // If it's the first bin, merge with the next one
          auto nxt = std::next(min_iv_it);
          nxt->lower = min_iv_it->lower;
          nxt->count_pos += min_iv_it->count_pos;
          nxt->count_neg += min_iv_it->count_neg;
          bins.erase(min_iv_it);
        }
        
        // Recompute metrics after merge
        compute_woe_iv();
        
        // Re-enforce monotonicity if needed
        if (is_monotonic) {
          enforce_monotonicity();
        }
        
        // Calculate total IV for convergence check
        double total_iv = std::accumulate(bins.begin(), bins.end(), 0.0,
                                          [](double sum, const Bin &bin) { return sum + bin.iv; });
        
        // Check convergence based on IV change
        if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
          converged = true;
          break;
        }
        
        prev_total_iv = total_iv;
        iterations_run++;
      }
    }
    
    // Step 5: Prepare output
    std::vector<std::string> bin_labels;
    Rcpp::NumericVector woe_values;
    Rcpp::NumericVector iv_values;
    Rcpp::IntegerVector counts;
    Rcpp::IntegerVector counts_pos;
    Rcpp::IntegerVector counts_neg;
    Rcpp::NumericVector cutpoints;
    
    for (const auto &bin : bins) {
      // Create readable bin labels with interval notation
      std::string lower_str = std::isinf(bin.lower) ? "-Inf" : std::to_string(bin.lower);
      std::string upper_str = std::isinf(bin.upper) ? "+Inf" : std::to_string(bin.upper);
      std::string bin_label = "(" + lower_str + ";" + upper_str + "]";
      
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.count_pos + bin.count_neg);
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      
      // Store cutpoints (excluding infinity)
      if (!std::isinf(bin.upper)) {
        cutpoints.push_back(bin.upper);
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for(int i = 0; i < bin_labels.size(); i++) {
      ids[i] = i + 1;
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (size_t i = 0; i < iv_values.size(); i++) {
      total_iv += iv_values[i];
    }
    
    // Return comprehensive results
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_labels,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = counts,
      Rcpp::Named("count_pos") = counts_pos,
      Rcpp::Named("count_neg") = counts_neg,
      Rcpp::Named("cutpoints") = cutpoints,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run,
      Rcpp::Named("total_iv") = total_iv
    );
  }
};


//' @title Optimal Binning for Numerical Variables using Branch and Bound Algorithm
//'
//' @description
//' Performs optimal binning for numerical variables using a Branch and Bound approach. 
//' This method transforms continuous features into discrete bins by maximizing the statistical 
//' relationship with a binary target variable while maintaining interpretability constraints.
//' The algorithm optimizes Weight of Evidence (WoE) and Information Value (IV) metrics
//' commonly used in risk modeling, credit scoring, and statistical analysis.
//'
//' @param target An integer binary vector (0 or 1) representing the target variable.
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins to generate (default: 3).
//' @param max_bins Maximum number of bins to generate (default: 5).
//' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins generated before optimization (default: 20).
//' @param is_monotonic Logical value indicating whether to enforce monotonicity in WoE (default: TRUE).
//' @param convergence_threshold Convergence threshold for total Information Value (IV) change (default: 1e-6).
//' @param max_iterations Maximum number of iterations allowed for the optimization process (default: 1000).
//'
//' @return A list containing:
//' \item{id}{Numeric identifiers for each bin (1-based).}
//' \item{bin}{Character vector with the intervals of each bin (e.g., `(-Inf; 0]`, `(0; +Inf)`).}
//' \item{woe}{Numeric vector with the Weight of Evidence values for each bin.}
//' \item{iv}{Numeric vector with the Information Value contribution for each bin.}
//' \item{count}{Integer vector with the total number of observations in each bin.}
//' \item{count_pos}{Integer vector with the number of positive observations in each bin.}
//' \item{count_neg}{Integer vector with the number of negative observations in each bin.}
//' \item{cutpoints}{Numeric vector of cut points between bins (excluding infinity).}
//' \item{converged}{Logical value indicating whether the algorithm converged.}
//' \item{iterations}{Number of iterations executed by the optimization algorithm.}
//' \item{total_iv}{The total Information Value of the binning solution.}
//'
//' @details
//' ## Algorithm Overview
//' The implementation follows a five-phase approach:
//' 
//' 1. **Input Validation**: Ensures data integrity and parameter validity.
//' 
//' 2. **Pre-Binning**: 
//'    - Creates initial bins using quantile-based division
//'    - Handles special cases for limited unique values
//'    - Uses binary search for efficient observation assignment
//' 
//' 3. **Statistical Stabilization**:
//'    - Merges bins with frequencies below the specified threshold
//'    - Ensures each bin has sufficient observations for reliable statistics
//' 
//' 4. **Monotonicity Enforcement** (optional):
//'    - Ensures WoE values follow a consistent trend (increasing or decreasing)
//'    - Improves interpretability and aligns with business expectations
//'    - Selects optimal monotonicity direction based on IV preservation
//' 
//' 5. **Branch and Bound Optimization**:
//'    - Iteratively merges bins with minimal IV contribution
//'    - Continues until reaching the target number of bins or convergence
//'    - Preserves predictive power while reducing complexity
//'
//' ## Mathematical Foundation
//' 
//' The algorithm optimizes two key metrics:
//' 
//' 1. **Weight of Evidence (WoE)** for bin \eqn{i}:
//'    \deqn{WoE_i = \ln\left(\frac{p_i/P}{n_i/N}\right)}
//'    
//'    Where:
//'    - \eqn{p_i}: Number of positive cases in bin \eqn{i}
//'    - \eqn{P}: Total number of positive cases
//'    - \eqn{n_i}: Number of negative cases in bin \eqn{i}
//'    - \eqn{N}: Total number of negative cases
//'    
//' 2. **Information Value (IV)** for bin \eqn{i}:
//'    \deqn{IV_i = \left(\frac{p_i}{P} - \frac{n_i}{N}\right) \times WoE_i}
//'    
//'    The total Information Value is the sum across all bins:
//'    \deqn{IV_{total} = \sum_{i=1}^{k} IV_i}
//'    
//' 3. **Smoothing**:
//'    The implementation uses Laplace smoothing to handle zero counts:
//'    \deqn{\frac{p_i + \alpha}{P + k\alpha}, \frac{n_i + \alpha}{N + k\alpha}}
//'    
//'    Where:
//'    - \eqn{\alpha}: Small constant (0.5 in this implementation)
//'    - \eqn{k}: Number of bins
//'
//' ## Branch and Bound Strategy
//' 
//' The core optimization uses a greedy iterative approach:
//' 
//' 1. Start with more bins than needed (from pre-binning)
//' 2. Identify the bin with the smallest IV contribution
//' 3. Merge this bin with an adjacent bin
//' 4. Recompute WoE and IV values
//' 5. If monotonicity is required, enforce it
//' 6. Repeat until target number of bins is reached or convergence
//' 
//' This approach minimizes information loss while reducing model complexity.
//'
//' @examples
//' \dontrun{
//' # Generate synthetic data
//' set.seed(123)
//' n <- 10000
//' feature <- rnorm(n)
//' # Create target with logistic relationship
//' target <- rbinom(n, 1, plogis(0.5 * feature))
//'
//' # Apply optimal binning
//' result <- optimal_binning_numerical_bb(target, feature, min_bins = 3, max_bins = 5)
//' print(result)
//' 
//' # Access specific components
//' bins <- result$bin
//' woe_values <- result$woe
//' total_iv <- result$total_iv
//' 
//' # Example with custom parameters
//' result2 <- optimal_binning_numerical_bb(
//'   target = target,
//'   feature = feature,
//'   min_bins = 2,
//'   max_bins = 8,
//'   bin_cutoff = 0.02,
//'   is_monotonic = TRUE
//' )
//' }
//'
//' @references
//' Belson, W. A. (1959). Matching and prediction on the principle of biological classification. 
//' *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 8(2), 65-75.
//' 
//' Siddiqi, N. (2006). *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*. 
//' John Wiley & Sons.
//' 
//' Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002). *Credit Scoring and Its Applications*. 
//' Society for Industrial and Applied Mathematics.
//' 
//' Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey. 
//' *GESTS International Transactions on Computer Science and Engineering*, 32(1), 47-58.
//' 
//' Dougherty, J., Kohavi, R., & Sahami, M. (1995). Supervised and Unsupervised Discretization of 
//' Continuous Features. *Proceedings of the Twelfth International Conference on Machine Learning*, 194-202.
//'
//' Bertsimas, D., & Dunn, J. (2017). Optimal classification trees. *Machine Learning*, 106(7), 1039-1082.
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
 // Convert R vectors to STL containers for C++ processing
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
 
 // Execute algorithm and return results
 return obb.fit();
} catch (const std::exception& e) {
 Rcpp::Rcerr << "Error in optimal_binning_numerical_bb: " << e.what() << std::endl;
 Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
}
}



// #include <Rcpp.h>
// #include <algorithm>
// #include <vector>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// #include <string>
// #include <numeric>
// 
// // Core class implementing the Optimal Binning
// class OptimalBinningNumericalBB {
// private:
//   std::vector<double> feature;
//   std::vector<int> target;
//   int min_bins;
//   int max_bins;
//   double bin_cutoff;
//   int max_n_prebins;
//   bool is_monotonic;
//   double convergence_threshold;
//   int max_iterations;
//   
//   static constexpr double EPSILON = 1e-10;
//   
//   struct Bin {
//     double lower;
//     double upper;
//     int count_pos;
//     int count_neg;
//     double woe;
//     double iv;
//   };
//   
//   std::vector<Bin> bins;
//   bool converged;
//   int iterations_run;
//   
//   // Validate input arguments
//   void validate_inputs() {
//     if (feature.size() != target.size()) {
//       throw std::invalid_argument("Feature and target vectors must have the same length.");
//     }
//     if (min_bins < 2) {
//       throw std::invalid_argument("min_bins must be at least 2.");
//     }
//     if (max_bins < min_bins) {
//       throw std::invalid_argument("max_bins must be >= min_bins.");
//     }
//     if (bin_cutoff < 0 || bin_cutoff > 1) {
//       throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
//     }
//     if (max_n_prebins < min_bins) {
//       throw std::invalid_argument("max_n_prebins must be >= min_bins.");
//     }
//     if (convergence_threshold <= 0) {
//       throw std::invalid_argument("convergence_threshold must be > 0.");
//     }
//     if (max_iterations <= 0) {
//       throw std::invalid_argument("max_iterations must be > 0.");
//     }
//   }
//   
//   // Handle the case of <= 2 unique values directly
//   void handle_two_or_fewer_unique_values(const std::vector<double>& unique_values) {
//     bins.clear();
//     // Construct bins based on unique values
//     for (size_t i = 0; i < unique_values.size(); ++i) {
//       Bin bin;
//       bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
//       bin.upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
//       bin.count_pos = 0;
//       bin.count_neg = 0;
//       bin.woe = 0.0;
//       bin.iv = 0.0;
//       bins.push_back(bin);
//     }
//     
//     // Direct assignment (only 1 or 2 bins, simple loop acceptable)
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       int tgt = target[i];
//       // With only 1 or 2 bins, linear scan is trivial
//       for (auto &bin : bins) {
//         if (val > bin.lower - EPSILON && val <= bin.upper + EPSILON) {
//           if (tgt == 1) bin.count_pos++;
//           else bin.count_neg++;
//           break;
//         }
//       }
//     }
//     
//     compute_woe_iv();
//     converged = true;
//     iterations_run = 0;
//   }
//   
//   // Compute a quantile from a sorted vector
//   double quantile(const std::vector<double>& data, double q) {
//     // q is assumed in [0,1]; index calculation safe
//     std::vector<double> temp = data;
//     std::sort(temp.begin(), temp.end());
//     size_t idx = static_cast<size_t>(std::floor(q * (temp.size() - 1)));
//     return temp[idx];
//   }
//   
//   // Prebin step using quantiles
//   void prebinning() {
//     std::vector<double> unique_values = feature;
//     std::sort(unique_values.begin(), unique_values.end());
//     unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
//     
//     if (unique_values.size() <= 2) {
//       handle_two_or_fewer_unique_values(unique_values);
//       return;
//     }
//     
//     // If few unique values, just create bins for each unique value up to min_bins
//     if (unique_values.size() <= static_cast<size_t>(min_bins)) {
//       bins.clear();
//       for (size_t i = 0; i < unique_values.size(); ++i) {
//         Bin bin;
//         bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : unique_values[i - 1];
//         bin.upper = (i == unique_values.size() - 1) ? std::numeric_limits<double>::infinity() : unique_values[i];
//         bin.count_pos = 0;
//         bin.count_neg = 0;
//         bin.woe = 0.0;
//         bin.iv = 0.0;
//         bins.push_back(bin);
//       }
//     } else {
//       // Use quantile-based initial binning
//       int n_prebins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);
//       std::vector<double> quantiles;
//       for (int i = 1; i < n_prebins; ++i) {
//         double qval = quantile(feature, i / double(n_prebins));
//         quantiles.push_back(qval);
//       }
//       
//       // Remove duplicate quantiles (just in case)
//       std::sort(quantiles.begin(), quantiles.end());
//       quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
//       
//       bins.clear();
//       bins.resize(quantiles.size() + 1);
//       for (size_t i = 0; i < bins.size(); ++i) {
//         if (i == 0) {
//           bins[i].lower = -std::numeric_limits<double>::infinity();
//           bins[i].upper = quantiles[i];
//         } else if (i == bins.size() - 1) {
//           bins[i].lower = quantiles[i - 1];
//           bins[i].upper = std::numeric_limits<double>::infinity();
//         } else {
//           bins[i].lower = quantiles[i - 1];
//           bins[i].upper = quantiles[i];
//         }
//         bins[i].count_pos = 0;
//         bins[i].count_neg = 0;
//         bins[i].woe = 0.0;
//         bins[i].iv = 0.0;
//       }
//     }
//     
//     // Assign observations to bins using binary search for performance
//     // Bins are sorted by upper boundary, so we can find bin via upper_bound
//     // Condition: value <= bin.upper
//     // Note: last bin has +Inf upper, always catches largest values
//     std::vector<double> uppers;
//     uppers.reserve(bins.size());
//     for (auto &b : bins) {
//       uppers.push_back(b.upper);
//     }
//     
//     for (size_t i = 0; i < feature.size(); ++i) {
//       double val = feature[i];
//       int tgt = target[i];
//       // Binary search for correct bin:
//       // We find the first bin with upper >= val
//       auto it = std::lower_bound(uppers.begin(), uppers.end(), val + EPSILON);
//       // it should never be end because last bin has +Inf upper
//       size_t idx = static_cast<size_t>(it - uppers.begin());
//       if (tgt == 1) {
//         bins[idx].count_pos++;
//       } else {
//         bins[idx].count_neg++;
//       }
//     }
//   }
//   
//   // Merge bins that are too rare
//   void merge_rare_bins() {
//     int total_count = 0;
//     for (auto &bin : bins) {
//       total_count += (bin.count_pos + bin.count_neg);
//     }
//     
//     double cutoff_count = bin_cutoff * total_count;
//     
//     // Attempt merging rare bins with neighbors
//     // Use a forward iteration pattern; after merging, iterator is adjusted
//     for (auto it = bins.begin(); it != bins.end();) {
//       int bin_count = it->count_pos + it->count_neg;
//       if (bin_count < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
//         if (it != bins.begin()) {
//           // Merge with previous
//           auto prev = std::prev(it);
//           prev->upper = it->upper;
//           prev->count_pos += it->count_pos;
//           prev->count_neg += it->count_neg;
//           it = bins.erase(it);
//         } else if (std::next(it) != bins.end()) {
//           // Merge with next if first bin is too small
//           auto nxt = std::next(it);
//           nxt->lower = it->lower;
//           nxt->count_pos += it->count_pos;
//           nxt->count_neg += it->count_neg;
//           it = bins.erase(it);
//         } else {
//           // If this is the only bin left or no valid merge candidate
//           ++it;
//         }
//       } else {
//         ++it;
//       }
//     }
//   }
//   
//   // Compute WoE and IV for each bin
//   void compute_woe_iv() {
//     int total_pos = 0;
//     int total_neg = 0;
//     for (auto &bin : bins) {
//       total_pos += bin.count_pos;
//       total_neg += bin.count_neg;
//     }
//     
//     // Adding small offsets to avoid division by zero and log of zero
//     double pos_denom = total_pos + 1.0;
//     double neg_denom = total_neg + 1.0;
//     
//     for (auto &bin : bins) {
//       double dist_pos = (bin.count_pos + 0.5) / pos_denom;
//       double dist_neg = (bin.count_neg + 0.5) / neg_denom;
//       // dist_neg, dist_pos > 0 due to smoothing
//       bin.woe = std::log(dist_pos / dist_neg);
//       bin.iv = (dist_pos - dist_neg) * bin.woe;
//     }
//   }
//   
//   // Enforce monotonicity if required
//   void enforce_monotonicity() {
//     bool increasing = std::is_sorted(bins.begin(), bins.end(),
//                                      [](const Bin &a, const Bin &b){ return a.woe <= b.woe + EPSILON; });
//     bool decreasing = std::is_sorted(bins.begin(), bins.end(),
//                                      [](const Bin &a, const Bin &b){ return a.woe >= b.woe - EPSILON; });
//     
//     // If already monotonic, do nothing
//     if (increasing || decreasing) return;
//     
//     // If not monotonic, attempt merges to fix it
//     for (auto it = std::next(bins.begin()); it != bins.end() && bins.size() > static_cast<size_t>(min_bins); ) {
//       // Merging logic for monotonic enforcement
//       // Check relative to previous bin's woe
//       double curr_woe = it->woe;
//       double prev_woe = std::prev(it)->woe;
//       
//       // If we detect a violation of monotonic pattern, merge the current bin into the previous one
//       // We attempt to enforce a direction similar to the initial observed trend in the first bins
//       // If WOE "jumps" in a way that breaks monotonicity significantly, merge it
//       if ((curr_woe < prev_woe - EPSILON && !increasing) || 
//           (curr_woe > prev_woe + EPSILON && !decreasing)) {
//         std::prev(it)->upper = it->upper;
//         std::prev(it)->count_pos += it->count_pos;
//         std::prev(it)->count_neg += it->count_neg;
//         it = bins.erase(it);
//       } else {
//         ++it;
//       }
//     }
//     
//     // Recompute WoE/IV after merges
//     compute_woe_iv();
//   }
//   
// public:
//   OptimalBinningNumericalBB(const std::vector<double> &feature_,
//                             const std::vector<int> &target_,
//                             int min_bins_ = 2,
//                             int max_bins_ = 5,
//                             double bin_cutoff_ = 0.05,
//                             int max_n_prebins_ = 20,
//                             bool is_monotonic_ = true,
//                             double convergence_threshold_ = 1e-6,
//                             int max_iterations_ = 1000)
//     : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//       bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), is_monotonic(is_monotonic_),
//       convergence_threshold(convergence_threshold_), max_iterations(max_iterations_),
//       converged(false), iterations_run(0) {
//     validate_inputs();
//   }
//   
//   Rcpp::List fit() {
//     // Initial prebinning
//     prebinning();
//     
//     if (!converged) {
//       // Merge rare bins
//       merge_rare_bins();
//       compute_woe_iv();
//       
//       // Enforce monotonicity if requested
//       if (is_monotonic) {
//         enforce_monotonicity();
//       }
//       
//       double prev_total_iv = std::numeric_limits<double>::infinity();
//       iterations_run = 0;
//       
//       // Branch and Bound approach: merge bins until conditions met
//       while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
//         // Find bin with minimum IV to merge
//         auto min_iv_it = std::min_element(bins.begin(), bins.end(),
//                                           [](const Bin &a, const Bin &b) { return a.iv < b.iv; });
//         
//         // Merge the min IV bin with a neighbor
//         if (min_iv_it != bins.begin()) {
//           auto prev = std::prev(min_iv_it);
//           prev->upper = min_iv_it->upper;
//           prev->count_pos += min_iv_it->count_pos;
//           prev->count_neg += min_iv_it->count_neg;
//           bins.erase(min_iv_it);
//         } else {
//           // min_iv_it is the first bin, merge forward
//           auto nxt = std::next(min_iv_it);
//           nxt->lower = min_iv_it->lower;
//           nxt->count_pos += min_iv_it->count_pos;
//           nxt->count_neg += min_iv_it->count_neg;
//           bins.erase(min_iv_it);
//         }
//         
//         // Recompute WoE and IV
//         compute_woe_iv();
//         // Re-enforce monotonicity if needed
//         if (is_monotonic) {
//           enforce_monotonicity();
//         }
//         
//         double total_iv = std::accumulate(bins.begin(), bins.end(), 0.0,
//                                           [](double sum, const Bin &bin) { return sum + bin.iv; });
//         
//         // Check convergence
//         if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
//           converged = true;
//           break;
//         }
//         prev_total_iv = total_iv;
//         iterations_run++;
//       }
//     }
//     
//     // Prepare output
//     std::vector<std::string> bin_labels;
//     Rcpp::NumericVector woe_values;
//     Rcpp::NumericVector iv_values;
//     Rcpp::IntegerVector counts;
//     Rcpp::IntegerVector counts_pos;
//     Rcpp::IntegerVector counts_neg;
//     Rcpp::NumericVector cutpoints;
//     
//     for (const auto &bin : bins) {
//       std::string lower_str = std::isinf(bin.lower) ? "-Inf" : std::to_string(bin.lower);
//       std::string upper_str = std::isinf(bin.upper) ? "+Inf" : std::to_string(bin.upper);
//       std::string bin_label = "(" + lower_str + ";" + upper_str + "]";
//       bin_labels.push_back(bin_label);
//       woe_values.push_back(bin.woe);
//       iv_values.push_back(bin.iv);
//       counts.push_back(bin.count_pos + bin.count_neg);
//       counts_pos.push_back(bin.count_pos);
//       counts_neg.push_back(bin.count_neg);
//       if (!std::isinf(bin.upper)) {
//         cutpoints.push_back(bin.upper);
//       }
//     }
//     
//     Rcpp::NumericVector ids(bin_labels.size());
//     for(int i = 0; i < bin_labels.size(); i++) {
//       ids[i] = i + 1;
//     }
//     
//     return Rcpp::List::create(
//       Rcpp::Named("id") = ids,
//       Rcpp::Named("bin") = bin_labels,
//       Rcpp::Named("woe") = woe_values,
//       Rcpp::Named("iv") = iv_values,
//       Rcpp::Named("count") = counts,
//       Rcpp::Named("count_pos") = counts_pos,
//       Rcpp::Named("count_neg") = counts_neg,
//       Rcpp::Named("cutpoints") = cutpoints,
//       Rcpp::Named("converged") = converged,
//       Rcpp::Named("iterations") = iterations_run
//     );
//   }
// };
// 
// 
// //' @title Optimal Binning for Numerical Variables using Branch and Bound
// //'
// //' @description
// //' Performs optimal binning for numerical variables using a Branch and Bound approach. 
// //' This method generates stable, high-quality bins while balancing interpretability and predictive power. 
// //' It ensures monotonicity in the Weight of Evidence (WoE), if requested, and guarantees that bins meet 
// //' user-defined constraints, such as minimum frequency and number of bins.
// //'
// //' @param target An integer binary vector (0 or 1) representing the target variable.
// //' @param feature A numeric vector of feature values to be binned.
// //' @param min_bins Minimum number of bins to generate (default: 3).
// //' @param max_bins Maximum number of bins to generate (default: 5).
// //' @param bin_cutoff Minimum frequency fraction for each bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins generated before optimization (default: 20).
// //' @param is_monotonic Logical value indicating whether to enforce monotonicity in WoE (default: TRUE).
// //' @param convergence_threshold Convergence threshold for total Information Value (IV) change (default: 1e-6).
// //' @param max_iterations Maximum number of iterations allowed for the optimization process (default: 1000).
// //'
// //' @return A list containing:
// //' \item{bin}{Character vector with the intervals of each bin (e.g., `(-Inf; 0]`, `(0; +Inf)`).}
// //' \item{woe}{Numeric vector with the WoE values for each bin.}
// //' \item{iv}{Numeric vector with the IV values for each bin.}
// //' \item{count}{Integer vector with the total number of observations in each bin.}
// //' \item{count_pos}{Integer vector with the number of positive observations in each bin.}
// //' \item{count_neg}{Integer vector with the number of negative observations in each bin.}
// //' \item{cutpoints}{Numeric vector of cut points between bins (excluding infinity).}
// //' \item{converged}{Logical value indicating whether the algorithm converged.}
// //' \item{iterations}{Number of iterations executed by the optimization algorithm.}
// //'
// //' @details
// //' The algorithm executes the following steps:
// //' 1. **Input Validation**: Ensures that inputs meet the requirements, such as compatible vector lengths 
// //'    and valid parameter ranges.
// //' 2. **Pre-Binning**: 
// //'    - If the feature has 2 or fewer unique values, assigns them directly to bins.
// //'    - Otherwise, generates quantile-based pre-bins, ensuring sufficient granularity.
// //' 3. **Rare Bin Merging**: Combines bins with frequencies below `bin_cutoff` with neighboring bins to 
// //'    ensure robustness and statistical reliability.
// //' 4. **WoE and IV Calculation**:
// //'    - Weight of Evidence (WoE): \eqn{\log(\text{Dist}_{\text{pos}} / \text{Dist}_{\text{neg}})}
// //'    - Information Value (IV): \eqn{\sum (\text{Dist}_{\text{pos}} - \text{Dist}_{\text{neg}}) \times \text{WoE}}
// //' 5. **Monotonicity Enforcement (Optional)**: Merges bins iteratively to ensure that WoE values follow a 
// //'    consistent increasing or decreasing trend, if `is_monotonic = TRUE`.
// //' 6. **Branch and Bound Optimization**: Iteratively merges bins with the smallest IV until the number of 
// //'    bins meets the `max_bins` constraint or IV change falls below `convergence_threshold`.
// //' 7. **Convergence Check**: Stops the process when the algorithm converges or reaches `max_iterations`.
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' n <- 10000
// //' feature <- rnorm(n)
// //' target <- rbinom(n, 1, plogis(0.5 * feature))
// //'
// //' result <- optimal_binning_numerical_bb(target, feature, min_bins = 3, max_bins = 5)
// //' print(result)
// //' }
// //'
// //' @references
// //' Farooq, B., & Miller, E. J. (2015). Optimal Binning for Continuous Variables.
// //' Kotsiantis, S., & Kanellopoulos, D. (2006). Discretization Techniques: A Recent Survey.
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_numerical_bb(
//    Rcpp::IntegerVector target,
//    Rcpp::NumericVector feature,
//    int min_bins = 3,
//    int max_bins = 5,
//    double bin_cutoff = 0.05,
//    int max_n_prebins = 20,
//    bool is_monotonic = true,
//    double convergence_threshold = 1e-6,
//    int max_iterations = 1000
// ) {
//  try {
//    OptimalBinningNumericalBB obb(
//        Rcpp::as<std::vector<double>>(feature),
//        Rcpp::as<std::vector<int>>(target),
//        min_bins,
//        max_bins,
//        bin_cutoff,
//        max_n_prebins,
//        is_monotonic,
//        convergence_threshold,
//        max_iterations
//    );
//    return obb.fit();
//  } catch (const std::exception& e) {
//    Rcpp::Rcerr << "Error in optimal_binning_numerical_bb: " << e.what() << std::endl;
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
