// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <numeric>

using namespace Rcpp;

// Constants for better readability and numerical stability
static constexpr double EPSILON = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
static constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

class OptimalBinningCategoricalMOB {
private:
  std::vector<std::string> feature;
  std::vector<bool> target;
  
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  
  std::unordered_map<std::string, double> category_counts;
  std::unordered_map<std::string, double> category_good;
  std::unordered_map<std::string, double> category_bad;
  double total_good;
  double total_bad;
  
  struct Bin {
    std::vector<std::string> categories;
    std::unordered_set<std::string> category_set; // For uniqueness checking
    double good_count;
    double bad_count;
    double woe;
    double iv;
    double event_rate; // Cache for event rate
    
    Bin() : good_count(0), bad_count(0), woe(0), iv(0), event_rate(0) {
      categories.reserve(8); // Pre-allocate for efficiency
    }
    
    // Add category ensuring uniqueness
    void addCategory(const std::string& cat, double good, double bad) {
      if (category_set.insert(cat).second) { // Only add if not already present
        categories.push_back(cat);
      }
      good_count += good;
      bad_count += bad;
      updateEventRate();
    }
    
    // Merge with another bin
    void mergeWith(const Bin& other) {
      for (const auto& cat : other.categories) {
        if (category_set.insert(cat).second) {
          categories.push_back(cat);
        }
      }
      good_count += other.good_count;
      bad_count += other.bad_count;
      updateEventRate();
    }
    
    // Update event rate
    void updateEventRate() {
      double total = good_count + bad_count;
      event_rate = total > 0 ? good_count / total : 0.0;
    }
    
    // Total count
    double totalCount() const {
      return good_count + bad_count;
    }
    
    // Calculate Bayesian smoothed WoE and IV
    void calculateMetrics(double total_good, double total_bad) {
      // Calculate Bayesian prior based on overall prevalence
      double prior_weight = BAYESIAN_PRIOR_STRENGTH;
      double overall_event_rate = total_good / (total_good + total_bad);
      
      double prior_good = prior_weight * overall_event_rate;
      double prior_bad = prior_weight * (1.0 - overall_event_rate);
      
      // Apply Bayesian smoothing to rates
      double rate_good = (good_count + prior_good) / (total_good + prior_weight);
      double rate_bad = (bad_count + prior_bad) / (total_bad + prior_weight);
      
      // Calculate WoE and IV with numerical stability
      if (rate_good < EPSILON && rate_bad < EPSILON) {
        woe = 0.0;
        iv = 0.0;
      } else {
        woe = std::log(std::max(rate_good, EPSILON) / std::max(rate_bad, EPSILON));
        iv = (rate_good - rate_bad) * woe;
      }
      
      // Handle non-finite values
      if (!std::isfinite(woe)) woe = 0.0;
      if (!std::isfinite(iv)) iv = 0.0;
    }
  };
  
  std::vector<Bin> bins;
  std::vector<Bin> best_bins; // Track best solution
  
  void calculateCategoryStats();
  void calculateInitialBins();
  bool isMonotonic(const std::vector<Bin>& bins_to_check) const;
  void enforceMonotonicity();
  void limitBins();
  void computeWoEandIV();
  size_t findBestMergeCandidate(size_t bin_idx) const;
  double calculateTotalIV() const;
  bool validateInputs() const;
  void handleRareCategories();
  
public:
  OptimalBinningCategoricalMOB(std::vector<std::string>&& feature_,
                               std::vector<bool>&& target_,
                               int min_bins_,
                               int max_bins_,
                               double bin_cutoff_,
                               int max_n_prebins_,
                               std::string bin_separator_,
                               double convergence_threshold_,
                               int max_iterations_);
  
  List fit();
};

OptimalBinningCategoricalMOB::OptimalBinningCategoricalMOB(
  std::vector<std::string>&& feature_,
  std::vector<bool>&& target_,
  int min_bins_,
  int max_bins_,
  double bin_cutoff_,
  int max_n_prebins_,
  std::string bin_separator_,
  double convergence_threshold_,
  int max_iterations_)
  : feature(std::move(feature_)),
    target(std::move(target_)),
    min_bins(min_bins_),
    max_bins(max_bins_),
    bin_cutoff(bin_cutoff_),
    max_n_prebins(max_n_prebins_),
    bin_separator(bin_separator_),
    convergence_threshold(convergence_threshold_),
    max_iterations(max_iterations_),
    total_good(0),
    total_bad(0) {
  
  if (!validateInputs()) {
    throw std::invalid_argument("Invalid input parameters");
  }
}

bool OptimalBinningCategoricalMOB::validateInputs() const {
  if (feature.size() != target.size()) {
    throw std::invalid_argument("Feature and target vectors must have the same length");
  }
  if (feature.empty()) {
    throw std::invalid_argument("Feature and target vectors cannot be empty");
  }
  if (min_bins <= 0 || max_bins <= 0 || min_bins > max_bins) {
    throw std::invalid_argument("Invalid min_bins or max_bins values");
  }
  if (bin_cutoff <= 0 || bin_cutoff >= 1) {
    throw std::invalid_argument("bin_cutoff must be between 0 and 1");
  }
  if (max_n_prebins <= 0) {
    throw std::invalid_argument("max_n_prebins must be positive");
  }
  if (convergence_threshold <= 0) {
    throw std::invalid_argument("convergence_threshold must be positive");
  }
  if (max_iterations <= 0) {
    throw std::invalid_argument("max_iterations must be positive");
  }
  
  // Check for empty strings in feature
  if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) { 
    return s.empty(); 
  })) {
    throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
  }
  
  return true;
}

void OptimalBinningCategoricalMOB::calculateCategoryStats() {
  category_counts.clear();
  category_good.clear();
  category_bad.clear();
  total_good = 0;
  total_bad = 0;
  
  for (size_t i = 0; i < feature.size(); ++i) {
    const auto& cat = feature[i];
    category_counts[cat]++;
    if (target[i]) {
      category_good[cat]++;
      total_good++;
    } else {
      category_bad[cat]++;
      total_bad++;
    }
  }
  
  // Check for extremely imbalanced datasets
  if (total_good < 5 || total_bad < 5) {
    Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
  }
}

void OptimalBinningCategoricalMOB::calculateInitialBins() {
  struct CategoryStats {
    std::string category;
    double woe;
    double good;
    double bad;
    double event_rate;
    double count;
    
    // Constructor for more direct initialization
    CategoryStats(const std::string& cat, double g, double b, double w) 
      : category(cat), woe(w), good(g), bad(b), count(g + b) {
      event_rate = count > 0 ? good / count : 0.0;
    }
  };
  
  std::vector<CategoryStats> cat_stats_vec;
  cat_stats_vec.reserve(category_counts.size());
  
  // Collect category statistics with Bayesian smoothing
  for (const auto& [cat, count] : category_counts) {
    double good = category_good[cat];
    double bad = category_bad[cat];
    
    // Calculate Bayesian prior based on overall prevalence
    double prior_weight = BAYESIAN_PRIOR_STRENGTH;
    double overall_event_rate = total_good / (total_good + total_bad);
    
    double prior_good = prior_weight * overall_event_rate;
    double prior_bad = prior_weight * (1.0 - overall_event_rate);
    
    // Apply Bayesian smoothing to rates
    double rate_good = (good + prior_good) / (total_good + prior_weight);
    double rate_bad = (bad + prior_bad) / (total_bad + prior_weight);
    
    // Calculate WoE with numerical stability
    double woe = std::log(std::max(rate_good, EPSILON) / std::max(rate_bad, EPSILON));
    
    // Handle non-finite values
    if (!std::isfinite(woe)) woe = 0.0;
    
    cat_stats_vec.emplace_back(cat, good, bad, woe);
  }
  
  // Sort categories by WoE
  std::sort(cat_stats_vec.begin(), cat_stats_vec.end(),
            [](const CategoryStats& a, const CategoryStats& b) {
              return a.woe < b.woe;
            });
  
  bins.clear();
  bins.reserve(cat_stats_vec.size());
  
  // Initialize bins with category stats
  for (const auto& stats : cat_stats_vec) {
    Bin bin;
    bin.addCategory(stats.category, stats.good, stats.bad);
    bin.woe = stats.woe; // Pre-computed WoE
    bins.push_back(std::move(bin));
  }
  
  // Handle rare categories first
  handleRareCategories();
  
  // Merge bins to limit the number of prebins using similarity-based approach
  while (bins.size() > static_cast<size_t>(max_n_prebins) && bins.size() > static_cast<size_t>(min_bins)) {
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
    
    // Merge the most similar bins
    bins[merge_idx1].mergeWith(bins[merge_idx2]);
    bins[merge_idx1].calculateMetrics(total_good, total_bad);
    bins.erase(bins.begin() + merge_idx2);
  }
  
  // Compute WoE and IV for all bins
  computeWoEandIV();
  
  // Sort bins by WoE for consistent ordering
  std::sort(bins.begin(), bins.end(),
            [](const Bin& a, const Bin& b) {
              return a.woe < b.woe;
            });
  
  // Force reduction to max_bins if needed
  limitBins();
}

void OptimalBinningCategoricalMOB::handleRareCategories() {
  // Calculate minimum count based on bin_cutoff
  double total_count = total_good + total_bad;
  double min_count = bin_cutoff * total_count;
  
  // Identify rare categories
  std::vector<size_t> rare_bin_indices;
  for (size_t i = 0; i < bins.size(); ++i) {
    if (bins[i].totalCount() < min_count) {
      rare_bin_indices.push_back(i);
    }
  }
  
  // Sort rare bins by count (ascending) for better merging strategy
  std::sort(rare_bin_indices.begin(), rare_bin_indices.end(),
            [this](size_t a, size_t b) { 
              return bins[a].totalCount() < bins[b].totalCount(); 
            });
  
  // Process rare categories
  for (size_t idx : rare_bin_indices) {
    // Skip if we already reached minimum bins
    if (bins.size() <= static_cast<size_t>(min_bins)) {
      break;
    }
    
    // Skip if bin no longer exists or is now above threshold
    if (idx >= bins.size() || bins[idx].totalCount() >= min_count) {
      continue;
    }
    
    // Find best merge candidate based on event rate similarity
    size_t best_candidate = findBestMergeCandidate(idx);
    
    // Merge with best candidate
    if (best_candidate != idx && best_candidate < bins.size()) {
      bins[best_candidate].mergeWith(bins[idx]);
      bins[best_candidate].calculateMetrics(total_good, total_bad);
      bins.erase(bins.begin() + idx);
      
      // Update indices of remaining rare bins
      for (auto& bin_idx : rare_bin_indices) {
        if (bin_idx > idx) {
          bin_idx--;
        }
      }
    }
  }
}

size_t OptimalBinningCategoricalMOB::findBestMergeCandidate(size_t bin_idx) const {
  if (bin_idx >= bins.size()) return bin_idx;
  
  // Find most similar bin based on event rate
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

bool OptimalBinningCategoricalMOB::isMonotonic(const std::vector<Bin>& bins_to_check) const {
  if (bins_to_check.size() <= 2) {
    return true; // Trivially monotonic
  }
  
  // Calculate average WoE gap for adaptive threshold
  double total_woe_gap = 0.0;
  for (size_t i = 1; i < bins_to_check.size(); ++i) {
    total_woe_gap += std::fabs(bins_to_check[i].woe - bins_to_check[i-1].woe);
  }
  
  double avg_gap = total_woe_gap / (bins_to_check.size() - 1);
  double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);
  
  // Check for monotonicity (either strictly increasing or strictly decreasing)
  bool is_increasing = true;
  bool is_decreasing = true;
  
  for (size_t i = 1; i < bins_to_check.size(); ++i) {
    if (bins_to_check[i].woe < bins_to_check[i-1].woe - monotonicity_threshold) {
      is_increasing = false;
    }
    if (bins_to_check[i].woe > bins_to_check[i-1].woe + monotonicity_threshold) {
      is_decreasing = false;
    }
    if (!is_increasing && !is_decreasing) {
      return false;
    }
  }
  
  return true;
}

void OptimalBinningCategoricalMOB::enforceMonotonicity() {
  if (isMonotonic(bins)) {
    return; // Already monotonic
  }
  
  // Determine monotonicity direction (prefer direction with fewer violations)
  bool is_increasing = true;
  int increasing_violations = 0;
  int decreasing_violations = 0;
  
  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i-1].woe) {
      increasing_violations++;
    }
    if (bins[i].woe > bins[i-1].woe) {
      decreasing_violations++;
    }
  }
  
  is_increasing = (increasing_violations <= decreasing_violations);
  
  // Fix monotonicity with a maximum number of attempts
  int attempts = 0;
  const int max_attempts = static_cast<int>(bins.size() * 3);
  
  while (!isMonotonic(bins) && bins.size() > static_cast<size_t>(min_bins) && attempts < max_attempts) {
    // Find all violations and their severity
    std::vector<std::pair<size_t, double>> violations;
    
    for (size_t i = 1; i < bins.size(); ++i) {
      bool violation = (is_increasing && bins[i].woe < bins[i-1].woe) ||
        (!is_increasing && bins[i].woe > bins[i-1].woe);
      
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
      
      // Determine which bin to keep based on count
      if (bins[i].totalCount() > bins[j].totalCount()) {
        // Merge j into i
        bins[i].mergeWith(bins[j]);
        bins[i].calculateMetrics(total_good, total_bad);
        bins.erase(bins.begin() + j);
      } else {
        // Merge i into j
        bins[j].mergeWith(bins[i]);
        bins[j].calculateMetrics(total_good, total_bad);
        bins.erase(bins.begin() + i);
      }
      
      computeWoEandIV(); // Recalculate after merging
    }
    
    attempts++;
  }
  
  if (attempts >= max_attempts) {
    Rcpp::warning("Could not ensure monotonicity in %d attempts. Using best solution found.", max_attempts);
  }
  
  // Final sort by WoE
  if (is_increasing) {
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return a.woe < b.woe;
    });
  } else {
    std::sort(bins.begin(), bins.end(), [](const Bin& a, const Bin& b) {
      return a.woe > b.woe;
    });
  }
}

// Fixed limitBins method to strictly enforce max_bins
void OptimalBinningCategoricalMOB::limitBins() {
  // Continue merging until we have EXACTLY max_bins
  while (bins.size() > static_cast<size_t>(max_bins)) {
    // Find two adjacent bins with minimum difference in WoE
    double min_diff = std::numeric_limits<double>::max();
    size_t min_index = 0;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      double diff = std::fabs(bins[i + 1].woe - bins[i].woe);
      if (diff < min_diff) {
        min_diff = diff;
        min_index = i;
      }
    }
    
    // Merge bins[min_index] and bins[min_index + 1]
    bins[min_index].mergeWith(bins[min_index + 1]);
    bins[min_index].calculateMetrics(total_good, total_bad);
    bins.erase(bins.begin() + min_index + 1);
  }
}

void OptimalBinningCategoricalMOB::computeWoEandIV() {
  for (auto& bin : bins) {
    bin.calculateMetrics(total_good, total_bad);
  }
}

double OptimalBinningCategoricalMOB::calculateTotalIV() const {
  double total_iv = 0.0;
  for (const auto& bin : bins) {
    total_iv += std::fabs(bin.iv);
  }
  return total_iv;
}

List OptimalBinningCategoricalMOB::fit() {
  calculateCategoryStats();
  
  int ncat = category_counts.size();
  bool converged = true;
  int iterations_run = 0;
  
  // If we have too many categories, apply the algorithm
  if (ncat > max_bins) {
    calculateInitialBins(); // This now includes a call to limitBins()
    
    // Track best solution
    double best_total_iv = calculateTotalIV();
    best_bins = bins;
    
    double prev_total_iv = best_total_iv;
    int iterations = 0;
    bool monotonic = false;
    
    while (!monotonic && iterations < max_iterations) {
      enforceMonotonicity();
      limitBins(); // This ensures we have exactly max_bins
      computeWoEandIV();
      
      double total_iv = calculateTotalIV();
      
      // Track best solution
      if (isMonotonic(bins) && total_iv > best_total_iv) {
        best_total_iv = total_iv;
        best_bins = bins;
      }
      
      if (std::abs(total_iv - prev_total_iv) < convergence_threshold) {
        monotonic = true;
      }
      
      prev_total_iv = total_iv;
      iterations++;
    }
    
    converged = monotonic;
    iterations_run = iterations;
    
    // Restore best solution if found
    if (best_total_iv > 0.0 && !best_bins.empty()) {
      bins = best_bins;
    }
    
    // Final check to ensure exactly max_bins
    limitBins();
  } else {
    // If we have fewer unique categories than max_bins, just create one bin per category
    bins.clear();
    for (const auto& [cat, count] : category_counts) {
      double good = category_good[cat];
      double bad = category_bad[cat];
      
      Bin bin;
      bin.addCategory(cat, good, bad);
      bin.calculateMetrics(total_good, total_bad);
      bins.push_back(std::move(bin));
    }
    
    // Sort bins by WoE
    std::sort(bins.begin(), bins.end(),
              [](const Bin& a, const Bin& b) {
                return a.woe < b.woe;
              });
    
    // Even in this case, we might still need to limit to max_bins
    // if there are more unique categories than max_bins
    if (bins.size() > static_cast<size_t>(max_bins)) {
      limitBins();
    }
  }
  
  // Double check we have at most max_bins bins
  if (bins.size() > static_cast<size_t>(max_bins)) {
    limitBins();
  }
  
  // At this point, bins.size() should be <= max_bins
  // If it's < min_bins, that's a problem that should have been handled earlier
  if (bins.size() < static_cast<size_t>(min_bins)) {
    Rcpp::warning("Could not create the minimum number of bins requested (%d). Created %d bins instead.", 
                  min_bins, bins.size());
  }
  
  // Prepare output
  std::vector<std::string> bin_names;
  std::vector<double> woe_values;
  std::vector<double> iv_values;
  std::vector<int> count_values;
  std::vector<int> count_pos_values;
  std::vector<int> count_neg_values;
  double total_iv = 0.0;
  
  for (auto& bin : bins) {
    // Use sorted categories for consistent output
    std::vector<std::string> sorted_categories = bin.categories;
    std::sort(sorted_categories.begin(), sorted_categories.end());
    
    std::string bin_name = sorted_categories[0];
    for (size_t i = 1; i < sorted_categories.size(); ++i) {
      bin_name += bin_separator + sorted_categories[i];
    }
    
    bin_names.push_back(bin_name);
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    count_values.push_back(static_cast<int>(bin.good_count + bin.bad_count));
    count_pos_values.push_back(static_cast<int>(bin.good_count));
    count_neg_values.push_back(static_cast<int>(bin.bad_count));
    
    total_iv += std::fabs(bin.iv);
  }
  
  // Create IDs
  Rcpp::NumericVector ids(bin_names.size());
  for(int i = 0; i < bin_names.size(); i++) {
    ids[i] = i + 1;
  }
  
  return Rcpp::List::create(
    Named("id") = ids,
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = iv_values,
    Named("count") = count_values,
    Named("count_pos") = count_pos_values,
    Named("count_neg") = count_neg_values,
    Named("total_iv") = total_iv,
    Named("converged") = converged,
    Named("iterations") = iterations_run
  );
}

//' @title Optimal Binning for Categorical Variables using Monotonic Optimal Binning (MOB)
//'
//' @description
//' Performs optimal binning for categorical variables using the Monotonic Optimal Binning (MOB) 
//' approach with enhanced statistical robustness. This implementation includes Bayesian smoothing 
//' for better stability with small samples, adaptive monotonicity enforcement, and sophisticated 
//' bin merging strategies.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param bin_separator Separator used for merging category names (default: "%;%").
//' @param convergence_threshold Convergence threshold for the algorithm (default: 1e-6).
//' @param max_iterations Maximum number of iterations for the algorithm (default: 1000).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item id: Numeric vector of bin identifiers.
//'   \item bin: Character vector of bin names (merged categories).
//'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector of Information Value (IV) for each bin.
//'   \item count: Integer vector of total counts for each bin.
//'   \item count_pos: Integer vector of positive target counts for each bin.
//'   \item count_neg: Integer vector of negative target counts for each bin.
//'   \item total_iv: Total Information Value of the binning.
//'   \item converged: Logical value indicating whether the algorithm converged.
//'   \item iterations: Integer value indicating the number of iterations run.
//' }
//'
//' @details
//' This enhanced version of the Monotonic Optimal Binning (MOB) algorithm implements several 
//' key improvements over traditional approaches:
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
//' \strong{Algorithm Phases:}
//' \enumerate{
//'   \item \strong{Initialization:} Calculate statistics for each category with Bayesian smoothing.
//'   \item \strong{Pre-binning:} Create initial bins sorted by WoE.
//'   \item \strong{Rare Category Handling:} Merge categories with frequency below bin_cutoff using a similarity-based approach.
//'   \item \strong{Monotonicity Enforcement:} Ensure monotonic WoE across bins using adaptive thresholds and severity-based prioritization.
//'   \item \strong{Bin Optimization:} Reduce number of bins to max_bins while maintaining monotonicity.
//'   \item \strong{Solution Tracking:} Maintain the best solution found during optimization.
//' }
//'
//' \strong{Key Features:}
//' \itemize{
//'   \item Bayesian smoothing for robust WoE estimation with small samples
//'   \item Similarity-based bin merging rather than just adjacent bins
//'   \item Adaptive monotonicity enforcement with violation severity prioritization
//'   \item Best solution tracking to ensure optimal results
//'   \item Efficient uniqueness handling for categories
//'   \item Comprehensive edge case handling
//'   \item Strict enforcement of max_bins parameter
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
//' result <- optimal_binning_categorical_mob(target, feature)
//'
//' # View results
//' print(result)
//'
//' # Force exactly 2 bins
//' result2 <- optimal_binning_categorical_mob(
//'   target, feature, 
//'   min_bins = 2, 
//'   max_bins = 2
//' )
//' }
//'
//' @references
//' \itemize{
//'    \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis.
//'          *Journal of the Operational Research Society*, 60(12), 1699-1707.
//'    \item Mironchyk, P., Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling.
//'          *arXiv preprint* arXiv:1711.05095.
//'    \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior 
//'          distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//'    \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary 
//'          classification. arXiv preprint arXiv:2001.08025.
//'    \item Thomas, L.C., Edelman, D.B., & Crook, J.N. (2002). Credit Scoring and its Applications. SIAM.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mob(Rcpp::IntegerVector target,
                                          Rcpp::CharacterVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 // Preliminary validations
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Feature and target vectors cannot be empty.");
 }
 
 if (feature.size() != target.size()) {
   Rcpp::stop("Target and feature vectors must have the same length.");
 }
 
 // Convert target to boolean vector with validation
 std::vector<bool> target_vec;
 target_vec.reserve(target.size());
 
 int na_target_count = 0;
 
 for (int i = 0; i < target.size(); ++i) {
   if (IntegerVector::is_na(target[i])) {
     na_target_count++;
     Rcpp::stop("Target cannot contain missing values at position %d.", i+1);
   } else if (target[i] != 0 && target[i] != 1) {
     Rcpp::stop("Target vector must be binary (0 and 1).");
   } else {
     target_vec.push_back(target[i] == 1);
   }
 }
 
 // Convert feature to string vector with NA handling
 std::vector<std::string> feature_vec;
 feature_vec.reserve(feature.size());
 
 int na_feature_count = 0;
 
 for (int i = 0; i < feature.size(); ++i) {
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
     na_feature_count++;
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }
 }
 
 // Warn about NA values in feature
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.", 
                 na_feature_count);
 }
 
 // Adjust max_bins based on unique categories
 std::unordered_set<std::string> unique_categories(feature_vec.begin(), feature_vec.end());
 max_bins = std::min(max_bins, static_cast<int>(unique_categories.size()));
 min_bins = std::min(min_bins, max_bins);
 
 try {
   OptimalBinningCategoricalMOB mob(std::move(feature_vec), std::move(target_vec),
                                    min_bins, max_bins, bin_cutoff, max_n_prebins,
                                    bin_separator, convergence_threshold, max_iterations);
   return mob.fit();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in OptimalBinningCategoricalMOB: %s", e.what());
 }
}

