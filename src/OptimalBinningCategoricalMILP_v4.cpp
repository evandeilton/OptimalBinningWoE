// [[Rcpp::depends(Rcpp)]]
// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

using namespace Rcpp;

// Constants for better readability and consistency
static constexpr double EPSILON = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
static constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

class OptimalBinningCategoricalMILP {
private:
  struct Bin {
    std::vector<std::string> categories;
    std::unordered_set<std::string> category_set; // For efficient uniqueness checking
    int count_pos;
    int count_neg;
    double woe;
    double iv;
    double event_rate; // Cache for event rate
    
    Bin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), event_rate(0.0) {
      categories.reserve(8); // Pre-allocate to avoid frequent reallocations
    }
    
    // Add a category ensuring uniqueness
    inline void add_category(const std::string& cat, int is_positive) {
      if (category_set.insert(cat).second) { // Only add if not already present
        categories.push_back(cat);
      }
      
      if (is_positive) {
        count_pos++;
      } else {
        count_neg++;
      }
      update_event_rate();
    }
    
    // Merge with another bin
    inline void merge_with(const Bin& other) {
      // Add only categories that don't already exist
      for (const auto& cat : other.categories) {
        if (category_set.insert(cat).second) {
          categories.push_back(cat);
        }
      }
      
      count_pos += other.count_pos;
      count_neg += other.count_neg;
      update_event_rate();
    }
    
    // Update event rate
    inline void update_event_rate() {
      int total = count_pos + count_neg;
      event_rate = total > 0 ? static_cast<double>(count_pos) / total : 0.0;
    }
    
    // Total count
    inline int total_count() const {
      return count_pos + count_neg;
    }
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
  double calculate_total_iv() const;
  size_t find_best_merge_candidate(size_t bin_idx) const;
  void merge_rare_categories();
  void enforce_monotonicity();
  
  inline double safe_log(double value) const {
    // Safe log: avoids log(0) by adding a small epsilon
    return std::log(std::max(value, EPSILON));
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
  
  // Check for empty strings in feature
  if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
    return s.empty(); 
  })) {
    throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
  }
  
  // Check for binary target
  bool has_zero = false;
  bool has_one = false;
  
  for (int val : target) {
    if (val == 0) has_zero = true;
    else if (val == 1) has_one = true;
    else throw std::invalid_argument("Target must contain only 0 and 1.");
    
    // Early termination once we've found both values
    if (has_zero && has_one) break;
  }
  
  if (!has_zero || !has_one) {
    throw std::invalid_argument("Target must contain both 0 and 1 values.");
  }
}

void OptimalBinningCategoricalMILP::handle_zero_counts() {
  if (total_pos == 0 || total_neg == 0) {
    throw std::runtime_error("Target variable must have at least one positive and one negative case.");
  }
  
  // Check for extremely imbalanced datasets
  if (total_pos < 5 || total_neg < 5) {
    Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
  }
}

void OptimalBinningCategoricalMILP::initialize_bins() {
  std::unordered_map<std::string, Bin> bin_map;
  bin_map.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
  
  total_pos = 0;
  total_neg = 0;
  
  for (size_t i = 0; i < target.size(); ++i) {
    const std::string& cat = feature[i];
    int tar = target[i];
    
    if (bin_map.find(cat) == bin_map.end()) {
      bin_map[cat] = Bin();
    }
    
    bin_map[cat].add_category(cat, tar);
    
    if (tar == 1) {
      total_pos++;
    } else {
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
  // Calculate Bayesian prior based on overall prevalence
  double prior_weight = BAYESIAN_PRIOR_STRENGTH;
  double overall_event_rate = static_cast<double>(total_pos) / 
    (total_pos + total_neg);
  
  double prior_pos = prior_weight * overall_event_rate;
  double prior_neg = prior_weight * (1.0 - overall_event_rate);
  
  // Apply Bayesian smoothing to proportions
  double dist_pos = static_cast<double>(bin.count_pos + prior_pos) / 
    static_cast<double>(total_pos + prior_weight);
  double dist_neg = static_cast<double>(bin.count_neg + prior_neg) / 
    static_cast<double>(total_neg + prior_weight);
  
  // Calculate WoE and IV with numerical stability
  if (dist_pos < EPSILON && dist_neg < EPSILON) {
    // Both are effectively zero
    bin.woe = 0.0;
    bin.iv = 0.0;
  } else {
    bin.woe = safe_log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
  
  // Handle non-finite values
  if (!std::isfinite(bin.woe)) bin.woe = 0.0;
  if (!std::isfinite(bin.iv)) bin.iv = 0.0;
}

bool OptimalBinningCategoricalMILP::is_monotonic() const {
  if (bins.size() <= 2) {
    return true;  // Monotonic if <= 2 bins
  }
  
  // Calculate average WoE gap for adaptive threshold
  double total_woe_gap = 0.0;
  for (size_t i = 1; i < bins.size(); ++i) {
    total_woe_gap += std::fabs(bins[i].woe - bins[i-1].woe);
  }
  
  double avg_gap = total_woe_gap / (bins.size() - 1);
  double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);
  
  bool increasing = true;
  bool decreasing = true;
  
  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
      increasing = false;
    }
    if (bins[i].woe > bins[i-1].woe + monotonicity_threshold) {
      decreasing = false;
    }
    if (!increasing && !decreasing) {
      return false;
    }
  }
  
  return true;
}

double OptimalBinningCategoricalMILP::calculate_total_iv() const {
  double total_iv = 0.0;
  for (const auto& bin : bins) {
    total_iv += std::fabs(bin.iv);
  }
  return total_iv;
}

size_t OptimalBinningCategoricalMILP::find_best_merge_candidate(size_t bin_idx) const {
  if (bin_idx >= bins.size()) return bin_idx;
  
  // Find best merge candidate based on event rate similarity
  double best_similarity = -1.0;
  size_t best_candidate = bin_idx;
  
  for (size_t j = 0; j < bins.size(); ++j) {
    if (j == bin_idx) continue;
    
    // Calculate event rate similarity
    double rate_diff = std::fabs(bins[bin_idx].event_rate - bins[j].event_rate);
    double similarity = 1.0 / (rate_diff + EPSILON);
    
    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_candidate = j;
    }
  }
  
  return best_candidate;
}

void OptimalBinningCategoricalMILP::merge_rare_categories() {
  double total_count = static_cast<double>(total_pos + total_neg);
  double min_count = bin_cutoff * total_count;
  
  // Identify all low-frequency bins at once
  std::vector<size_t> low_freq_bins;
  
  for (size_t i = 0; i < bins.size(); ++i) {
    if (bins[i].total_count() < min_count) {
      low_freq_bins.push_back(i);
    }
  }
  
  // Sort bins by frequency (ascending) for better merging strategy
  std::sort(low_freq_bins.begin(), low_freq_bins.end(),
            [this](size_t a, size_t b) { 
              return bins[a].total_count() < bins[b].total_count(); 
            });
  
  // Process low-frequency bins efficiently
  for (size_t idx : low_freq_bins) {
    if (static_cast<int>(bins.size()) <= min_bins) {
      break; // Never go below min_bins
    }
    
    // Check if bin still exists and is still below cutoff
    if (idx >= bins.size() || bins[idx].total_count() >= min_count) {
      continue;
    }
    
    // Find best merge candidate based on event rate similarity
    size_t best_candidate = find_best_merge_candidate(idx);
    
    // Try to merge with best candidate
    if (best_candidate != idx && best_candidate < bins.size()) {
      // Create a new temporary bin
      Bin merged_bin = bins[idx];
      merged_bin.merge_with(bins[best_candidate]);
      
      // Calculate metrics for merged bin
      calculate_woe_iv(merged_bin);
      
      // Replace the bin with lower IV with the merged bin
      size_t replace_idx = (std::fabs(bins[idx].iv) <= std::fabs(bins[best_candidate].iv)) 
        ? idx : best_candidate;
      size_t remove_idx = (replace_idx == idx) ? best_candidate : idx;
      
      bins[replace_idx] = std::move(merged_bin);
      bins.erase(bins.begin() + remove_idx);
      
      // Adjust indices for remaining bins
      for (auto& remaining_idx : low_freq_bins) {
        if (remaining_idx == remove_idx) {
          remaining_idx = replace_idx;
        } else if (remaining_idx > remove_idx) {
          remaining_idx--;
        }
      }
    }
  }
}

void OptimalBinningCategoricalMILP::enforce_monotonicity() {
  if (bins.size() <= 2) return;  // Already monotonic
  
  // Calculate average WoE gap for adaptive threshold
  double total_woe_gap = 0.0;
  for (size_t i = 1; i < bins.size(); ++i) {
    total_woe_gap += std::fabs(bins[i].woe - bins[i-1].woe);
  }
  
  double avg_gap = total_woe_gap / (bins.size() - 1);
  double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);
  
  // Determine monotonicity direction (increasing or decreasing)
  bool increasing = true;
  int increasing_violations = 0;
  int decreasing_violations = 0;
  
  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
      increasing_violations++;
    }
    if (bins[i].woe > bins[i-1].woe + monotonicity_threshold) {
      decreasing_violations++;
    }
  }
  
  // Determine preferred direction (fewer violations)
  increasing = (increasing_violations <= decreasing_violations);
  
  // Fix violations with a maximum number of attempts
  int attempts = 0;
  const int max_attempts = static_cast<int>(bins.size() * 3);
  
  while (!is_monotonic() && static_cast<int>(bins.size()) > min_bins && attempts < max_attempts) {
    // Find all violations and their severity
    std::vector<std::pair<size_t, double>> violations;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      bool violation = (increasing && bins[i].woe < bins[i-1].woe - monotonicity_threshold) ||
        (!increasing && bins[i].woe > bins[i-1].woe + monotonicity_threshold);
      
      if (violation) {
        double severity = std::fabs(bins[i].woe - bins[i-1].woe);
        violations.push_back(std::make_pair(i, severity));
      }
    }
    
    // If no violations, we're done
    if (violations.empty()) break;
    
    // Sort violations by severity (descending)
    std::sort(violations.begin(), violations.end(),
              [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) { 
                return a.second > b.second; 
              });
    
    // Fix the most severe violation
    if (!violations.empty()) {
      size_t i = violations[0].first;
      size_t j = i - 1;
      
      // Create a new temporary bin
      Bin merged_bin = bins[j];
      merged_bin.merge_with(bins[i]);
      
      // Calculate metrics for merged bin
      calculate_woe_iv(merged_bin);
      
      // Replace the bin with lower IV with the merged bin
      bins[j] = std::move(merged_bin);
      bins.erase(bins.begin() + i);
    }
    
    attempts++;
  }
  
  if (attempts >= max_attempts) {
    Rcpp::warning("Could not ensure monotonicity in %d attempts. Using best solution found.", max_attempts);
  }
  
  // Final sort by WoE to ensure consistent ordering
  if (increasing) {
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return a.woe < b.woe;
    });
  } else {
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return a.woe > b.woe;
    });
  }
}

void OptimalBinningCategoricalMILP::merge_bins() {
  const size_t min_bins_size = static_cast<size_t>(min_bins);
  const size_t max_bins_size = static_cast<size_t>(std::min(max_bins, static_cast<int>(bins.size())));
  const size_t max_n_prebins_size = static_cast<size_t>(std::max(max_n_prebins, min_bins));
  
  // Reduce pre-bins if necessary
  if (bins.size() > max_n_prebins_size) {
    // Sort by count (ascending) for better merging strategy
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return (a.count_pos + a.count_neg) < (b.count_pos + b.count_neg);
    });
    
    while (bins.size() > max_n_prebins_size && bins.size() > min_bins_size) {
      // Find best pair to merge based on event rate similarity
      double best_similarity = -1.0;
      size_t merge_idx1 = 0;
      size_t merge_idx2 = 0;
      
      for (size_t i = 0; i < bins.size(); ++i) {
        for (size_t j = i + 1; j < bins.size(); ++j) {
          double rate_diff = std::fabs(bins[i].event_rate - bins[j].event_rate);
          double similarity = 1.0 / (rate_diff + EPSILON);
          
          if (similarity > best_similarity) {
            best_similarity = similarity;
            merge_idx1 = i;
            merge_idx2 = j;
          }
        }
      }
      
      // Merge the best pair
      bins[merge_idx1].merge_with(bins[merge_idx2]);
      calculate_woe_iv(bins[merge_idx1]);
      bins.erase(bins.begin() + merge_idx2);
    }
  }
  
  // Handle rare categories
  merge_rare_categories();
  
  // Enforce monotonicity first
  enforce_monotonicity();
  
  // Now optimize the number of bins if still needed
  bool merging = true;
  double prev_total_iv = calculate_total_iv();
  
  // Track best solution seen so far
  double best_total_iv = prev_total_iv;
  std::vector<Bin> best_bins = bins;
  
  while (merging && iterations_run < max_iterations) {
    merging = false;
    iterations_run++;
    
    // If too many bins, merge least informative bins
    if (bins.size() > max_bins_size && bins.size() > min_bins_size) {
      // Sort by absolute IV (ascending)
      std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
        return std::fabs(a.iv) < std::fabs(b.iv);
      });
      
      // Create a new merged bin
      Bin merged_bin = bins[0];
      merged_bin.merge_with(bins[1]);
      calculate_woe_iv(merged_bin);
      
      // Replace and remove
      bins[0] = std::move(merged_bin);
      bins.erase(bins.begin() + 1);
      
      merging = true;
    }
    
    // Check for monotonicity after merging
    if (!is_monotonic() && bins.size() > min_bins_size) {
      enforce_monotonicity();
      merging = true;
    }
    
    // Calculate current IV
    double total_iv = calculate_total_iv();
    
    // Track best solution
    if (is_monotonic() && bins.size() <= max_bins_size && bins.size() >= min_bins_size && 
        total_iv > best_total_iv) {
      best_total_iv = total_iv;
      best_bins = bins;
    }
    
    // Check for convergence
    if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
      converged = true;
      break;
    }
    
    prev_total_iv = total_iv;
  }
  
  // Restore best solution if found
  if (best_total_iv > 0.0 && best_bins.size() <= max_bins_size && best_bins.size() >= min_bins_size) {
    bins = std::move(best_bins);
  }
  
  // Ensure final monotonicity
  if (!is_monotonic() && bins.size() > min_bins_size) {
    enforce_monotonicity();
  }
}

std::string OptimalBinningCategoricalMILP::join_categories(const std::vector<std::string>& categories) const {
  // Efficient concatenation with uniqueness check
  if (categories.empty()) return "";
  
  std::unordered_set<std::string> unique_categories;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());
  
  for (const auto& cat : categories) {
    if (unique_categories.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }
  
  size_t total_length = 0;
  for (const auto& c : unique_vec) total_length += c.size() + bin_separator.size();
  total_length = (total_length > bin_separator.size()) ? total_length - bin_separator.size() : total_length;
  
  std::string result;
  result.reserve(total_length);
  for (size_t i = 0; i < unique_vec.size(); ++i) {
    if (i > 0) result += bin_separator;
    result += unique_vec[i];
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
    Rcpp::NumericVector ids(num_bins);
    
    double total_iv = 0.0;
    
    for (size_t i = 0; i < num_bins; ++i) {
      const Bin& bin = bins[i];
      bin_names[i] = join_categories(bin.categories);
      bin_woe[i] = bin.woe;
      bin_iv[i] = bin.iv;
      bin_count[i] = bin.count_pos + bin.count_neg;
      bin_count_pos[i] = bin.count_pos;
      bin_count_neg[i] = bin.count_neg;
      ids[i] = i + 1;
      
      total_iv += std::fabs(bin.iv);
    }
    
    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = bin_woe,
      Rcpp::Named("iv") = bin_iv,
      Rcpp::Named("count") = bin_count,
      Rcpp::Named("count_pos") = bin_count_pos,
      Rcpp::Named("count_neg") = bin_count_neg,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  } catch (const std::exception& e) {
    Rcpp::stop("Error in optimal binning: %s", e.what());
  }
}


//' @title Optimal Binning for Categorical Variables using MILP
//'
//' @description
//' Performs optimal binning for categorical variables using a Mixed Integer Linear Programming (MILP) 
//' inspired approach with enhanced statistical robustness. It creates optimal bins for a categorical 
//' feature based on its relationship with a binary target variable, maximizing the predictive power 
//' while respecting user-defined constraints. This implementation includes Bayesian smoothing for 
//' improved stability with small samples and sophisticated merging strategies.
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
//'   \item id: Numeric vector of bin identifiers.
//'   \item bin: Character vector of bin categories.
//'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector of Information Value (IV) for each bin.
//'   \item count: Integer vector of total observations in each bin.
//'   \item count_pos: Integer vector of positive target observations in each bin.
//'   \item count_neg: Integer vector of negative target observations in each bin.
//'   \item total_iv: Total Information Value of the binning.
//'   \item converged: Logical indicating whether the algorithm converged.
//'   \item iterations: Integer indicating the number of iterations run.
//' }
//'
//' @details
//' This enhanced version of the Optimal Binning algorithm for categorical variables implements 
//' several key improvements over traditional approaches:
//' 
//' \strong{Mathematical Framework:}
//' 
//' The Weight of Evidence (WoE) with Bayesian smoothing is calculated as:
//' 
//' \deqn{WoE_i = \ln\left(\frac{p_i^*}{q_i^*}\right)}
//' 
//' where:
//' \itemize{
//'   \item \eqn{p_i^* = \frac{n_i^+ + \alpha \cdot \pi}{N^+ + \alpha}} is the smoothed proportion of 
//'         events in bin i
//'   \item \eqn{q_i^* = \frac{n_i^- + \alpha \cdot (1-\pi)}{N^- + \alpha}} is the smoothed proportion of 
//'         non-events in bin i
//'   \item \eqn{\pi = \frac{N^+}{N^+ + N^-}} is the overall event rate
//'   \item \eqn{\alpha} is the prior strength parameter (default: 0.5)
//'   \item \eqn{n_i^+} is the count of events in bin i
//'   \item \eqn{n_i^-} is the count of non-events in bin i
//'   \item \eqn{N^+} is the total number of events
//'   \item \eqn{N^-} is the total number of non-events
//' }
//'
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV_i = (p_i^* - q_i^*) \times WoE_i}
//'
//' The total IV is:
//' 
//' \deqn{IV_{total} = \sum_{i=1}^{k} |IV_i|}
//'
//' \strong{Algorithm Phases:}
//' \enumerate{
//'   \item \strong{Initialization:} Create bins for each unique category with comprehensive statistics.
//'   \item \strong{Pre-binning:} Reduce to max_n_prebins by merging similar bins based on event rates.
//'   \item \strong{Rare Category Merging:} Combine categories with frequency below bin_cutoff using similarity-based strategy.
//'   \item \strong{Monotonicity Enforcement:} Ensure monotonic relationship in WoE across bins using adaptive thresholds.
//'   \item \strong{Bin Optimization:} Iteratively merge bins to maximize IV while respecting constraints.
//'   \item \strong{Solution Tracking:} Maintain the best solution found during optimization.
//' }
//'
//' \strong{Key Enhancements:}
//' \itemize{
//'   \item Bayesian smoothing for robust estimation of WoE with small samples
//'   \item Similarity-based bin merging rather than just adjacent bins
//'   \item Adaptive monotonicity enforcement with violation severity prioritization
//'   \item Best solution tracking to ensure optimal results
//'   \item Comprehensive handling of edge cases and rare categories
//' }
//'
//' @references
//' \itemize{
//'   \item Belotti, P., Kirches, C., Leyffer, S., Linderoth, J., Luedtke, J., & Mahajan, A. (2013). Mixed-integer nonlinear optimization. Acta Numerica, 22, 1-131.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. SSRN Electronic Journal. doi:10.2139/ssrn.2978774
//'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//'   \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and its Applications. SIAM.
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
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
//'
//' # Handle rare categories with lower threshold
//' result2 <- optimal_binning_categorical_milp(
//'   target, feature, 
//'   bin_cutoff = 0.02,
//'   min_bins = 2, 
//'   max_bins = 5
//' )
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
 // Preliminary validation
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature and target cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length.");
 }
 
 // Handle NA values
 std::vector<int> target_vec;
 std::vector<std::string> feature_vec;
 
 target_vec.reserve(target.size());
 feature_vec.reserve(feature.size());
 
 int na_feature_count = 0;
 int na_target_count = 0;
 
 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   // Handle NA in feature
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
     na_feature_count++;
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
   
   // Check for NA in target
   if (IntegerVector::is_na(target[i])) {
     na_target_count++;
     Rcpp::stop("Target cannot contain missing values at position %d.", i+1);
   } else {
     target_vec.push_back(target[i]);
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
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

