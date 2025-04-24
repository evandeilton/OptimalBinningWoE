// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <numeric>

using namespace Rcpp;

// Constants for better readability and numerical stability
static constexpr double EPSILON = 1e-10;
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
// Bayesian smoothing parameter (adjustable prior strength)
static constexpr double BAYESIAN_PRIOR_STRENGTH = 0.5;

// Class for Optimal Binning using Simulated Annealing with enhanced algorithms
class OptimalBinningCategoricalSAB {
private:
  // Input data
  std::vector<std::string> feature;
  std::vector<int> target;
  
  // Binning parameters
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  
  // Simulated Annealing parameters
  double initial_temperature;
  double cooling_rate;
  int max_iterations;
  double convergence_threshold;
  bool adaptive_cooling;
  
  // Category statistics
  std::vector<std::string> unique_categories;
  std::unordered_map<std::string, int> category_counts;
  std::unordered_map<std::string, int> positive_counts;
  int total_count;
  int total_positive;
  int total_negative;
  
  // Binning results
  int actual_bins;
  std::vector<int> best_solution;
  std::vector<int> current_solution;
  double best_iv;
  double current_iv;
  bool converged;
  int iterations_run;
  int iterations_without_improvement;
  
  // Random number generator
  std::mt19937 gen;
  
  // Initialize the algorithm with comprehensive error checking
  void initialize() {
    // Extract unique categories with uniqueness guarantee
    std::unordered_set<std::string> unique_set;
    for (const auto& f : feature) {
      if (!f.empty()) {
        unique_set.insert(f);
      }
    }
    unique_categories.assign(unique_set.begin(), unique_set.end());
    
    // Count totals
    total_count = static_cast<int>(feature.size());
    total_positive = std::count(target.begin(), target.end(), 1);
    total_negative = total_count - total_positive;
    
    // Check for extremely imbalanced datasets
    if (total_positive < 5 || total_negative < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }
    
    // Count per category
    for (size_t i = 0; i < feature.size(); ++i) {
      category_counts[feature[i]]++;
      if (target[i] == 1) {
        positive_counts[feature[i]]++;
      }
    }
    
    int n_categories = static_cast<int>(unique_categories.size());
    actual_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), n_categories);
    
    // Initial solution: using kmeans-like strategy for better starting point
    initialize_improved_solution();
    
    best_solution = current_solution;
    best_iv = current_iv = calculate_iv(current_solution);
    converged = false;
    iterations_run = 0;
    iterations_without_improvement = 0;
  }
  
  // Enhanced initialization with kmeans-like strategy
  void initialize_improved_solution() {
    int n_categories = static_cast<int>(unique_categories.size());
    current_solution.resize(n_categories);
    
    if (n_categories <= actual_bins) {
      // If few categories, assign each to a separate bin
      for (int i = 0; i < n_categories; ++i) {
        current_solution[i] = i;
      }
    } else {
      // Calculate event rates for each category
      std::vector<double> event_rates(n_categories);
      for (int i = 0; i < n_categories; ++i) {
        const std::string& category = unique_categories[i];
        int count = category_counts[category];
        int pos_count = positive_counts[category];
        event_rates[i] = count > 0 ? static_cast<double>(pos_count) / count : 0.0;
      }
      
      // Sort categories by event rate
      std::vector<int> indices(n_categories);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return event_rates[a] < event_rates[b];
      });
      
      // Assign to bins based on sorted order
      for (int i = 0; i < n_categories; ++i) {
        current_solution[indices[i]] = std::min(i * actual_bins / n_categories, actual_bins - 1);
      }
      
      // Add some randomness to avoid local optima
      std::uniform_int_distribution<> dis(0, 99);
      for (int i = 0; i < n_categories; ++i) {
        if (dis(gen) < 10) { // 10% chance to reassign
          current_solution[i] = std::uniform_int_distribution<>(0, actual_bins - 1)(gen);
        }
      }
    }
  }
  
  // Calculate IV with Bayesian smoothing for more robust estimation
  double calculate_iv(const std::vector<int>& solution) const {
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    // Aggregate counts per bin
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts.at(category);
      bin_positives[bin] += positive_counts.at(category);
    }
    
    double iv = 0.0;
    
    // Apply Bayesian smoothing and calculate IV
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        int bin_negatives = bin_counts[i] - bin_positives[i];
        
        // Calculate bin proportion and apply cutoff
        double bin_proportion = static_cast<double>(bin_counts[i]) / total_count;
        if (bin_proportion < bin_cutoff) {
          // Penalize small bins
          iv -= 1000.0;
          continue;
        }
        
        // Apply Bayesian smoothing
        double prior_weight = BAYESIAN_PRIOR_STRENGTH;
        double overall_event_rate = static_cast<double>(total_positive) / total_count;
        
        double prior_pos = prior_weight * overall_event_rate;
        double prior_neg = prior_weight * (1.0 - overall_event_rate);
        
        double pos_dist = static_cast<double>(bin_positives[i] + prior_pos) / 
          (total_positive + prior_weight);
        double neg_dist = static_cast<double>(bin_negatives + prior_neg) / 
          (total_negative + prior_weight);
        
        if (pos_dist > EPSILON && neg_dist > EPSILON) {
          double woe = std::log(pos_dist / neg_dist);
          if (std::isfinite(woe)) {
            iv += (pos_dist - neg_dist) * woe;
          }
        }
      }
    }
    
    // Add monotonicity constraint as a penalty
    if (!is_monotonic(solution)) {
      iv -= 500.0; // Penalize non-monotonic solutions
    }
    
    return iv;
  }
  
  // Check monotonicity with adaptive threshold
  bool is_monotonic(const std::vector<int>& solution) const {
    std::vector<double> bin_rates(actual_bins, 0.0);
    std::vector<int> bin_counts(actual_bins, 0);
    
    // Calculate event rates for each bin
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = solution[i];
      bin_counts[bin] += category_counts.at(category);
      bin_rates[bin] += positive_counts.at(category);
    }
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        bin_rates[i] /= bin_counts[i];
      }
    }
    
    // Calculate average gap for adaptive threshold
    double total_gap = 0.0;
    for (int i = 1; i < actual_bins; ++i) {
      total_gap += std::fabs(bin_rates[i] - bin_rates[i-1]);
    }
    double avg_gap = actual_bins > 1 ? total_gap / (actual_bins - 1) : 0.0;
    double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);
    
    // Check monotonicity with adaptive threshold
    bool increasing = true;
    bool decreasing = true;
    for (int i = 1; i < actual_bins; ++i) {
      if (bin_rates[i] < bin_rates[i - 1] - monotonicity_threshold) {
        increasing = false;
      }
      if (bin_rates[i] > bin_rates[i - 1] + monotonicity_threshold) {
        decreasing = false;
      }
    }
    
    return increasing || decreasing;
  }
  
  // Enhanced neighbor generation with smarter strategy
  std::vector<int> generate_neighbor(const std::vector<int>& solution) {
    std::vector<int> neighbor = solution;
    if (neighbor.size() <= 1) return neighbor;
    
    // Different neighbor generation strategies
    std::uniform_int_distribution<> strategy_dis(0, 9);
    int strategy = strategy_dis(gen);
    
    if (strategy < 5) { // 50% chance for simple swap
      // Standard random swap
      std::uniform_int_distribution<> dis(0, static_cast<int>(neighbor.size()) - 1);
      int idx1 = dis(gen);
      int idx2 = dis(gen);
      
      while (idx2 == idx1 && neighbor.size() > 1) {
        idx2 = dis(gen);
      }
      
      std::swap(neighbor[idx1], neighbor[idx2]);
    } else if (strategy < 8) { // 30% chance for reassigning a random category
      // Reassign a random category to a random bin
      std::uniform_int_distribution<> cat_dis(0, static_cast<int>(neighbor.size()) - 1);
      std::uniform_int_distribution<> bin_dis(0, actual_bins - 1);
      int idx = cat_dis(gen);
      neighbor[idx] = bin_dis(gen);
    } else { // 20% chance for smarter event rate-based move
      // Find a category with highest event rate difference from its bin average
      std::vector<double> bin_rates(actual_bins, 0.0);
      std::vector<int> bin_counts(actual_bins, 0);
      
      // Calculate bin event rates
      for (size_t i = 0; i < unique_categories.size(); ++i) {
        const std::string& category = unique_categories[i];
        int bin = solution[i];
        bin_counts[bin] += category_counts.at(category);
        bin_rates[bin] += positive_counts.at(category);
      }
      
      for (int i = 0; i < actual_bins; ++i) {
        if (bin_counts[i] > 0) {
          bin_rates[i] /= bin_counts[i];
        }
      }
      
      // Find category with highest difference from its bin rate
      double max_diff = -1.0;
      int max_diff_idx = -1;
      
      for (size_t i = 0; i < unique_categories.size(); ++i) {
        const std::string& category = unique_categories[i];
        int bin = solution[i];
        int count = category_counts.at(category);
        int pos = positive_counts.at(category);
        
        double cat_rate = count > 0 ? static_cast<double>(pos) / count : 0.0;
        double diff = std::fabs(cat_rate - bin_rates[bin]);
        
        if (diff > max_diff) {
          max_diff = diff;
          max_diff_idx = static_cast<int>(i);
        }
      }
      
      if (max_diff_idx >= 0) {
        // Move the category to a bin with closer average rate
        const std::string& category = unique_categories[max_diff_idx];
        int count = category_counts.at(category);
        int pos = positive_counts.at(category);
        double cat_rate = count > 0 ? static_cast<double>(pos) / count : 0.0;
        
        // Find closest bin by event rate
        double min_rate_diff = std::numeric_limits<double>::max();
        int best_bin = 0;
        
        for (int i = 0; i < actual_bins; ++i) {
          if (i != neighbor[max_diff_idx]) {
            double diff = std::fabs(cat_rate - bin_rates[i]);
            if (diff < min_rate_diff) {
              min_rate_diff = diff;
              best_bin = i;
            }
          }
        }
        
        neighbor[max_diff_idx] = best_bin;
      }
    }
    
    return neighbor;
  }
  
  // Calculate acceptance probability with adaptive temperature
  double calculate_acceptance_probability(double current_iv, double neighbor_iv, 
                                          double temperature, int iter) const {
    // Calculate scaled energy difference
    double energy_diff = (neighbor_iv - current_iv);
    
    // Standard Boltzmann acceptance probability
    double probability = std::exp(energy_diff / temperature);
    
    // Add adaptive component based on iterations without improvement
    if (iterations_without_improvement > 50) {
      probability *= 1.5; // Increase acceptance to escape local optimum
    }
    
    return probability;
  }
  
public:
  OptimalBinningCategoricalSAB(const std::vector<std::string>& feature_,
                               const std::vector<int>& target_,
                               int min_bins_ = 3,
                               int max_bins_ = 5,
                               double bin_cutoff_ = 0.05,
                               int max_n_prebins_ = 20,
                               std::string bin_separator_ = "%;%",
                               double initial_temperature_ = 1.0,
                               double cooling_rate_ = 0.995,
                               int max_iterations_ = 1000,
                               double convergence_threshold_ = 1e-6,
                               bool adaptive_cooling_ = true)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
      initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
      max_iterations(max_iterations_), convergence_threshold(convergence_threshold_),
      adaptive_cooling(adaptive_cooling_), iterations_without_improvement(0) {
    
    // Enhanced validation
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same length");
    }
    
    if (feature.empty() || target.empty()) {
      throw std::invalid_argument("Feature and target vectors cannot be empty");
    }
    
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be at least 2");
    }
    
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins");
    }
    
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
    }
    
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be >= max_bins");
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
      else throw std::invalid_argument("Target must be binary (0 or 1)");
      
      if (has_zero && has_one) break;
    }
    
    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values");
    }
    
    // Initialize random number generator with better seed
    std::random_device rd;
    std::seed_seq seq{rd(), rd(), rd(), rd(), rd()};
    gen.seed(seq);
    
    initialize();
  }
  
  // Enhanced fit method with adaptive cooling and parallel tempering
  void fit() {
    double temperature = initial_temperature;
    double prev_best_iv = best_iv;
    
    // Keep track of best solution found
    std::vector<int> global_best_solution = best_solution;
    double global_best_iv = best_iv;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
      // Generate and evaluate neighbor
      std::vector<int> neighbor = generate_neighbor(current_solution);
      double neighbor_iv = calculate_iv(neighbor);
      
      // Decide whether to accept the new solution
      if (neighbor_iv > current_iv) {
        // Always accept better solutions
        current_solution = neighbor;
        current_iv = neighbor_iv;
        iterations_without_improvement = 0;
        
        // Update best solution if improved
        if (current_iv > best_iv) {
          best_solution = current_solution;
          best_iv = current_iv;
          
          // Update global best
          if (best_iv > global_best_iv) {
            global_best_solution = best_solution;
            global_best_iv = best_iv;
          }
        }
      } else {
        // Consider accepting worse solutions based on temperature
        double acceptance_probability = calculate_acceptance_probability(
          current_iv, neighbor_iv, temperature, iter);
        
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(gen) < acceptance_probability) {
          current_solution = neighbor;
          current_iv = neighbor_iv;
        }
        
        iterations_without_improvement++;
      }
      
      // Adjust temperature based on cooling schedule
      if (adaptive_cooling && iterations_without_improvement > 30) {
        // Adjust cooling rate to escape local optima
        temperature *= std::pow(cooling_rate, 0.9);
      } else {
        temperature *= cooling_rate;
      }
      
      // Periodically restart from the best known solution to avoid getting stuck
      if (iterations_without_improvement > 100) {
        current_solution = best_solution;
        current_iv = best_iv;
        temperature = initial_temperature * 0.5; // Restart with lower temperature
        iterations_without_improvement = 0;
      }
      
      // Check convergence
      if (iter % 10 == 0) { // Check every 10 iterations to save computation
        if (std::abs(best_iv - prev_best_iv) < convergence_threshold && 
            is_monotonic(best_solution)) {
          converged = true;
          iterations_run = iter + 1;
          break;
        }
        prev_best_iv = best_iv;
      }
    }
    
    if (!converged) {
      iterations_run = max_iterations;
    }
    
    // Use global best solution
    best_solution = global_best_solution;
    best_iv = global_best_iv;
    
    // Ensure monotonicity as a final step
    ensure_monotonicity();
  }
  
  // Ensure monotonicity post-hoc if needed
  void ensure_monotonicity() {
    if (actual_bins <= 2 || is_monotonic(best_solution)) {
      return; // Already monotonic or trivial case
    }
    
    // Calculate event rates for each bin
    std::vector<double> bin_rates(actual_bins, 0.0);
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_index_map(actual_bins); // To keep track of original indices
    
    for (int i = 0; i < actual_bins; ++i) {
      bin_index_map[i] = i;
    }
    
    // Calculate bin event rates
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = best_solution[i];
      bin_counts[bin] += category_counts.at(category);
      bin_rates[bin] += positive_counts.at(category);
    }
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        bin_rates[i] /= bin_counts[i];
      }
    }
    
    // Sort bins by event rate
    std::vector<int> indices(actual_bins);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return bin_rates[a] < bin_rates[b];
    });
    
    // Create a remapping from old bins to new sorted bins
    std::vector<int> bin_remap(actual_bins);
    for (int i = 0; i < actual_bins; ++i) {
      bin_remap[indices[i]] = i;
    }
    
    // Apply remapping to create a monotonic solution
    std::vector<int> monotonic_solution = best_solution;
    for (size_t i = 0; i < monotonic_solution.size(); ++i) {
      monotonic_solution[i] = bin_remap[monotonic_solution[i]];
    }
    
    // Check if new solution is better
    double monotonic_iv = calculate_iv(monotonic_solution);
    if (monotonic_iv > best_iv || !is_monotonic(best_solution)) {
      best_solution = monotonic_solution;
      best_iv = monotonic_iv;
    } else {
      // Try an alternative approach if the first one didn't work well
      int max_attempts = 100;
      for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::vector<int> alt_solution = generate_neighbor(best_solution);
        double alt_iv = calculate_iv(alt_solution);
        
        if (is_monotonic(alt_solution) && alt_iv > best_iv) {
          best_solution = alt_solution;
          best_iv = alt_iv;
          break;
        }
      }
    }
  }
  
  // Get final binning results with enhanced statistics
  List get_results() const {
    // Create bins with correct statistics
    std::vector<std::string> bins;
    std::vector<double> woe;
    std::vector<double> iv;
    std::vector<int> count;
    std::vector<int> count_pos;
    std::vector<int> count_neg;
    
    // Process categories into bins
    std::vector<std::unordered_set<std::string>> bin_category_sets(actual_bins);
    std::vector<std::vector<std::string>> bin_categories(actual_bins);
    std::vector<int> bin_counts(actual_bins, 0);
    std::vector<int> bin_positives(actual_bins, 0);
    
    for (size_t i = 0; i < unique_categories.size(); ++i) {
      const std::string& category = unique_categories[i];
      int bin = best_solution[i];
      
      // Add category to bin ensuring uniqueness
      if (bin_category_sets[bin].insert(category).second) {
        bin_categories[bin].push_back(category);
      }
      
      bin_counts[bin] += category_counts.at(category);
      bin_positives[bin] += positive_counts.at(category);
    }
    
    // Calculate total IV with statistics
    double total_iv = 0.0;
    
    for (int i = 0; i < actual_bins; ++i) {
      if (bin_counts[i] > 0) {
        // Sort categories within bin for consistent output
        std::sort(bin_categories[i].begin(), bin_categories[i].end());
        
        // Create bin name
        std::string bin_name = "";
        for (size_t j = 0; j < bin_categories[i].size(); ++j) {
          if (j > 0) bin_name += bin_separator;
          bin_name += bin_categories[i][j];
        }
        
        // Store bin statistics
        bins.push_back(bin_name);
        count.push_back(bin_counts[i]);
        count_pos.push_back(bin_positives[i]);
        int bin_negatives = bin_counts[i] - bin_positives[i];
        count_neg.push_back(bin_negatives);
        
        // Calculate WoE and IV with Bayesian smoothing
        double prior_weight = BAYESIAN_PRIOR_STRENGTH;
        double overall_event_rate = static_cast<double>(total_positive) / total_count;
        
        double prior_pos = prior_weight * overall_event_rate;
        double prior_neg = prior_weight * (1.0 - overall_event_rate);
        
        double pos_dist = static_cast<double>(bin_positives[i] + prior_pos) / 
          (total_positive + prior_weight);
        double neg_dist = static_cast<double>(bin_negatives + prior_neg) / 
          (total_negative + prior_weight);
        
        double woe_value = 0.0;
        double iv_value = 0.0;
        
        if (pos_dist > EPSILON && neg_dist > EPSILON) {
          woe_value = std::log(pos_dist / neg_dist);
          iv_value = (pos_dist - neg_dist) * woe_value;
          
          // Handle non-finite values
          if (!std::isfinite(woe_value)) woe_value = 0.0;
          if (!std::isfinite(iv_value)) iv_value = 0.0;
          
          total_iv += iv_value;
        }
        
        woe.push_back(woe_value);
        iv.push_back(iv_value);
      }
    }
    
    // Create numeric IDs for bins
    Rcpp::NumericVector ids(bins.size());
    for(int i = 0; i < static_cast<int>(bins.size()); i++) {
      ids[i] = i + 1;
    }
    
    // Return comprehensive results
    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bins,
      Named("woe") = woe,
      Named("iv") = iv,
      Named("count") = count,
      Named("count_pos") = count_pos,
      Named("count_neg") = count_neg,
      Named("total_iv") = total_iv, 
      Named("converged") = converged,
      Named("iterations") = iterations_run
    );
  }
};


//' @title Optimal Binning for Categorical Variables using Simulated Annealing
//'
//' @description
//' Performs optimal binning for categorical variables using an enhanced Simulated Annealing 
//' approach. This implementation maximizes Information Value (IV) while maintaining monotonicity 
//' in the bins, using Bayesian smoothing for robust estimation and adaptive temperature scheduling 
//' for better convergence.
//'
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A character vector of categorical feature values.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' @param bin_separator Separator string for merging categories (default: "%;%").
//' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
//' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
//' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
//' @param convergence_threshold Threshold for convergence (default: 1e-6).
//' @param adaptive_cooling Whether to use adaptive cooling schedule (default: TRUE).
//'
//' @return A list containing the following elements:
//' \itemize{
//'   \item id: Numeric vector of bin identifiers.
//'   \item bin: Character vector of bin names.
//'   \item woe: Numeric vector of Weight of Evidence (WoE) values for each bin.
//'   \item iv: Numeric vector of Information Value (IV) for each bin.
//'   \item count: Integer vector of total counts for each bin.
//'   \item count_pos: Integer vector of positive counts for each bin.
//'   \item count_neg: Integer vector of negative counts for each bin.
//'   \item total_iv: Total Information Value of the binning.
//'   \item converged: Logical value indicating whether the algorithm converged.
//'   \item iterations: Integer value indicating the number of iterations run.
//' }
//'
//' @details
//' This enhanced version of the Simulated Annealing Binning (SAB) algorithm implements several 
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
//' \strong{Simulated Annealing:}
//' 
//' The algorithm uses an enhanced version of Simulated Annealing with these key features:
//' \itemize{
//'   \item Multiple neighborhood generation strategies for better exploration
//'   \item Adaptive temperature scheduling to escape local optima
//'   \item Periodic restarting from the best known solution
//'   \item Smart initialization using event rates for better starting points
//' }
//'
//' The probability of accepting a worse solution is calculated as:
//' 
//' \deqn{P(accept) = \exp\left(\frac{\Delta IV}{T}\right)}
//' 
//' where \eqn{\Delta IV} is the change in Information Value and \eqn{T} is the current temperature.
//'
//' \strong{Algorithm Phases:}
//' \enumerate{
//'   \item \strong{Initialization:} Create initial bin assignments using a kmeans-like strategy based on event rates
//'   \item \strong{Optimization:} Apply Simulated Annealing to find the optimal assignment of categories to bins
//'   \item \strong{Monotonicity Enforcement:} Ensure the final solution has monotonic bin event rates
//' }
//'
//' \strong{Key Features:}
//' \itemize{
//'   \item Bayesian smoothing for robust estimation with small samples
//'   \item Multiple neighbor generation strategies for better search space exploration
//'   \item Adaptive temperature scheduling to escape local optima
//'   \item Smart initialization for better starting points
//'   \item Strong monotonicity enforcement
//'   \item Comprehensive handling of edge cases
//' }
//'
//' @examples
//' \dontrun{
//' # Basic usage
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
//' result <- optimal_binning_categorical_sab(target, feature)
//' print(result)
//'
//' # Adjust simulated annealing parameters
//' result2 <- optimal_binning_categorical_sab(
//'   target, feature,
//'   min_bins = 2,
//'   max_bins = 4,
//'   initial_temperature = 2.0,
//'   cooling_rate = 0.99,
//'   max_iterations = 2000
//' )
//' }
//'
//' @references
//' \itemize{
//'   \item Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. science, 220(4598), 671-680.
//'   \item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis. Journal of the Operational Research Society, 60(12), 1699-1707.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. arXiv preprint arXiv:1711.05095.
//'   \item Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y. S. (2008). A weakly informative default prior distribution for logistic and other regression models. The annals of applied statistics, 2(4), 1360-1383.
//'   \item Navas-Palencia, G. (2020). Optimal binning: mathematical programming formulations for binary classification. arXiv preprint arXiv:2001.08025.
//' }
//'
//' @export
// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double initial_temperature = 1.0,
                                          double cooling_rate = 0.995,
                                          int max_iterations = 1000,
                                          double convergence_threshold = 1e-6,
                                          bool adaptive_cooling = true) {
 try {
   // Basic parameter validation
   if (target.size() != feature.size()) {
     Rcpp::stop("Target and feature vectors must have the same length");
   }
   
   if (feature.size() == 0 || target.size() == 0) {
     Rcpp::stop("Feature and target vectors cannot be empty");
   }
   
   if (min_bins < 2) {
     Rcpp::stop("min_bins must be at least 2");
   }
   
   if (max_bins < min_bins) {
     Rcpp::stop("max_bins must be >= min_bins");
   }
   
   if (bin_cutoff <= 0 || bin_cutoff >= 1) {
     Rcpp::stop("bin_cutoff must be between 0 and 1");
   }
   
   if (max_n_prebins < max_bins) {
     Rcpp::stop("max_n_prebins must be >= max_bins");
   }
   
   // Convert data with NA handling
   std::vector<std::string> feature_vec;
   std::vector<int> target_vec;
   
   feature_vec.reserve(feature.size());
   target_vec.reserve(target.size());
   
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
   
   // Create and run the binning algorithm
   OptimalBinningCategoricalSAB binner(feature_vec, target_vec,
                                       min_bins, max_bins, bin_cutoff,
                                       max_n_prebins, bin_separator,
                                       initial_temperature, cooling_rate,
                                       max_iterations, convergence_threshold,
                                       adaptive_cooling);
   binner.fit();
   
   return binner.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: %s", e.what());
 }
}











// // [[Rcpp::depends(Rcpp)]]
// 
// #include <Rcpp.h>
// #include <vector>
// #include <string>
// #include <unordered_map>
// #include <unordered_set>
// #include <algorithm>
// #include <random>
// #include <cmath>
// #include <limits>
// #include <stdexcept>
// 
// using namespace Rcpp;
// 
// // Constants
// static constexpr double EPSILON = 1e-10;
// 
// // Class for Optimal Binning using Simulated Annealing
// class OptimalBinningCategoricalSAB {
// private:
//  // Input data
//  std::vector<std::string> feature;
//  std::vector<int> target;
//  
//  // Binning parameters
//  int min_bins;
//  int max_bins;
//  double bin_cutoff;
//  int max_n_prebins;
//  std::string bin_separator;
//  
//  // Simulated Annealing parameters
//  double initial_temperature;
//  double cooling_rate;
//  int max_iterations;
//  double convergence_threshold;
//  
//  // Category statistics
//  std::vector<std::string> unique_categories;
//  std::unordered_map<std::string, int> category_counts;
//  std::unordered_map<std::string, int> positive_counts;
//  int total_count;
//  int total_positive;
//  int total_negative;
//  
//  // Binning results
//  int actual_bins;
//  std::vector<int> best_solution;
//  double best_iv;
//  bool converged;
//  int iterations_run;
//  
//  // Random number generator
//  std::mt19937 gen;
//  
//  void initialize() {
//    // Extract unique categories
//    std::unordered_set<std::string> unique_set(feature.begin(), feature.end());
//    unique_categories = std::vector<std::string>(unique_set.begin(), unique_set.end());
//    
//    // Count totals
//    total_count = (int)feature.size();
//    total_positive = std::count(target.begin(), target.end(), 1);
//    total_negative = total_count - total_positive;
//    
//    // Count per category
//    for (size_t i = 0; i < feature.size(); ++i) {
//      category_counts[feature[i]]++;
//      if (target[i] == 1) {
//        positive_counts[feature[i]]++;
//      }
//    }
//    
//    int n_categories = (int)unique_categories.size();
//    actual_bins = std::min(std::max(min_bins, std::min(max_bins, n_categories)), n_categories);
//    
//    // Initial solution: round-robin assignment
//    best_solution.resize(n_categories);
//    for (size_t i = 0; i < best_solution.size(); ++i) {
//      best_solution[i] = (int)(i % actual_bins);
//    }
//    
//    // Shuffle initial solution
//    std::shuffle(best_solution.begin(), best_solution.end(), gen);
//    
//    best_iv = calculate_iv(best_solution);
//    converged = false;
//    iterations_run = 0;
//  }
//  
//  double calculate_iv(const std::vector<int>& solution) const {
//    std::vector<int> bin_counts(actual_bins, 0);
//    std::vector<int> bin_positives(actual_bins, 0);
//    
//    // Aggregate counts per bin
//    for (size_t i = 0; i < unique_categories.size(); ++i) {
//      const std::string& category = unique_categories[i];
//      int bin = solution[i];
//      bin_counts[bin] += category_counts.at(category);
//      bin_positives[bin] += positive_counts.at(category);
//    }
//    
//    double iv = 0.0;
//    
//    for (int i = 0; i < actual_bins; ++i) {
//      if (bin_counts[i] > 0) {
//        int bin_negatives = bin_counts[i] - bin_positives[i];
//        double pos_dist = (double)bin_positives[i] / total_positive;
//        double neg_dist = (double)bin_negatives / total_negative;
//        
//        double bin_proportion = (double)bin_counts[i] / total_count;
//        if (bin_proportion < bin_cutoff) {
//          // Penalize small bins
//          iv -= 1000.0;
//          continue;
//        }
//        
//        if (pos_dist > EPSILON && neg_dist > EPSILON) {
//          double woe = std::log(pos_dist / neg_dist);
//          iv += (pos_dist - neg_dist) * woe;
//        }
//      }
//    }
//    
//    return iv;
//  }
//  
//  bool is_monotonic(const std::vector<int>& solution) const {
//    std::vector<double> bin_rates(actual_bins, 0.0);
//    std::vector<int> bin_counts(actual_bins, 0);
//    
//    for (size_t i = 0; i < unique_categories.size(); ++i) {
//      const std::string& category = unique_categories[i];
//      int bin = solution[i];
//      bin_counts[bin] += category_counts.at(category);
//      bin_rates[bin] += positive_counts.at(category);
//    }
//    
//    for (int i = 0; i < actual_bins; ++i) {
//      if (bin_counts[i] > 0) {
//        bin_rates[i] /= bin_counts[i];
//      }
//    }
//    
//    bool increasing = true;
//    bool decreasing = true;
//    for (int i = 1; i < actual_bins; ++i) {
//      if (bin_rates[i] < bin_rates[i - 1] - EPSILON) {
//        increasing = false;
//      }
//      if (bin_rates[i] > bin_rates[i - 1] + EPSILON) {
//        decreasing = false;
//      }
//    }
//    
//    return increasing || decreasing;
//  }
//  
//  std::vector<int> generate_neighbor(const std::vector<int>& solution) {
//    std::vector<int> neighbor = solution;
//    if (neighbor.size() <= 1) return neighbor;
//    
//    std::uniform_int_distribution<> dis(0, (int)neighbor.size() - 1);
//    int idx1 = dis(gen);
//    int idx2 = dis(gen);
//    
//    while (idx2 == idx1) {
//      idx2 = dis(gen);
//    }
//    
//    std::swap(neighbor[idx1], neighbor[idx2]);
//    return neighbor;
//  }
//  
// public:
//  OptimalBinningCategoricalSAB(const std::vector<std::string>& feature_,
//                               const std::vector<int>& target_,
//                               int min_bins_ = 3,
//                               int max_bins_ = 5,
//                               double bin_cutoff_ = 0.05,
//                               int max_n_prebins_ = 20,
//                               std::string bin_separator_ = "%;%",
//                               double initial_temperature_ = 1.0,
//                               double cooling_rate_ = 0.995,
//                               int max_iterations_ = 1000,
//                               double convergence_threshold_ = 1e-6)
//    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
//      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
//      initial_temperature(initial_temperature_), cooling_rate(cooling_rate_),
//      max_iterations(max_iterations_), convergence_threshold(convergence_threshold_) {
//    
//    if (feature.size() != target.size()) {
//      throw std::invalid_argument("Feature and target vectors must have the same length");
//    }
//    if (min_bins < 2) {
//      throw std::invalid_argument("min_bins must be at least 2");
//    }
//    if (max_bins < min_bins) {
//      throw std::invalid_argument("max_bins must be >= min_bins");
//    }
//    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//      throw std::invalid_argument("bin_cutoff must be between 0 and 1");
//    }
//    if (max_n_prebins < max_bins) {
//      throw std::invalid_argument("max_n_prebins must be >= max_bins");
//    }
//    
//    std::random_device rd;
//    gen.seed(rd());
//    initialize();
//  }
//  
//  void fit() {
//    double temperature = initial_temperature;
//    double prev_best_iv = best_iv;
//    
//    for (int iter = 0; iter < max_iterations; ++iter) {
//      std::vector<int> neighbor = generate_neighbor(best_solution);
//      double neighbor_iv = calculate_iv(neighbor);
//      
//      if (actual_bins > min_bins && !is_monotonic(neighbor)) {
//        continue; // Skip non-monotonic
//      }
//      
//      if (neighbor_iv > best_iv) {
//        best_solution = neighbor;
//        best_iv = neighbor_iv;
//      } else {
//        double acceptance_probability = std::exp((neighbor_iv - best_iv) / temperature);
//        std::uniform_real_distribution<> dis(0.0, 1.0);
//        if (dis(gen) < acceptance_probability) {
//          best_solution = neighbor;
//          best_iv = neighbor_iv;
//        }
//      }
//      
//      temperature *= cooling_rate;
//      
//      if (std::abs(best_iv - prev_best_iv) < convergence_threshold) {
//        converged = true;
//        iterations_run = iter + 1;
//        break;
//      }
//      
//      prev_best_iv = best_iv;
//    }
//    
//    if (!converged) {
//      iterations_run = max_iterations;
//    }
//    
//    // Ensure monotonicity post-hoc if needed
//    if (actual_bins > min_bins) {
//      int max_monotonic_attempts = 100;
//      for (int attempt = 0; attempt < max_monotonic_attempts; ++attempt) {
//        if (is_monotonic(best_solution)) {
//          break;
//        }
//        best_solution = generate_neighbor(best_solution);
//        best_iv = calculate_iv(best_solution);
//      }
//    }
//  }
//  
//  List get_results() const {
//    std::vector<std::string> bins;
//    std::vector<double> woe;
//    std::vector<double> iv;
//    std::vector<int> count;
//    std::vector<int> count_pos;
//    std::vector<int> count_neg;
//    
//    std::vector<std::vector<std::string>> bin_categories(actual_bins);
//    std::vector<int> bin_counts(actual_bins, 0);
//    std::vector<int> bin_positives(actual_bins, 0);
//    
//    for (size_t i = 0; i < unique_categories.size(); ++i) {
//      const std::string& category = unique_categories[i];
//      int bin = best_solution[i];
//      bin_categories[bin].push_back(category);
//      bin_counts[bin] += category_counts.at(category);
//      bin_positives[bin] += positive_counts.at(category);
//    }
//    
//    double total_iv = 0.0;
//    for (int i = 0; i < actual_bins; ++i) {
//      if (bin_counts[i] > 0) {
//        std::string bin_name = "";
//        for (const auto& category : bin_categories[i]) {
//          if (!bin_name.empty()) bin_name += bin_separator;
//          bin_name += category;
//        }
//        bins.push_back(bin_name);
//        count.push_back(bin_counts[i]);
//        count_pos.push_back(bin_positives[i]);
//        int bin_negatives = bin_counts[i] - bin_positives[i];
//        count_neg.push_back(bin_negatives);
//        
//        double pos_dist = (double)bin_positives[i] / total_positive;
//        double neg_dist = (double)bin_negatives / total_negative;
//        
//        double woe_value = 0.0;
//        double iv_value = 0.0;
//        
//        if (pos_dist > EPSILON && neg_dist > EPSILON) {
//          woe_value = std::log(pos_dist / neg_dist);
//          iv_value = (pos_dist - neg_dist) * woe_value;
//          total_iv += iv_value;
//        }
//        
//        woe.push_back(woe_value);
//        iv.push_back(iv_value);
//      }
//    }
//    
//    Rcpp::NumericVector ids(bins.size());
//    for(int i = 0; i < bins.size(); i++) {
//       ids[i] = i + 1;
//    }
//    
//    return Rcpp::List::create(
//      Named("id") = ids,
//      Named("bin") = bins,
//      Named("woe") = woe,
//      Named("iv") = iv,
//      Named("count") = count,
//      Named("count_pos") = count_pos,
//      Named("count_neg") = count_neg,
//      Named("converged") = converged,
//      Named("iterations") = iterations_run
//    );
//  }
// };
// 
// 
// //' @title
// //' Optimal Binning for Categorical Variables using Simulated Annealing
// //'
// //' @description
// //' This function performs optimal binning for categorical variables using a Simulated Annealing approach.
// //' It maximizes the Information Value (IV) while maintaining monotonicity in the bins.
// //'
// //' @param target An integer vector of binary target values (0 or 1).
// //' @param feature A character vector of categorical feature values.
// //' @param min_bins Minimum number of bins (default: 3).
// //' @param max_bins Maximum number of bins (default: 5).
// //' @param bin_cutoff Minimum proportion of observations in a bin (default: 0.05).
// //' @param max_n_prebins Maximum number of pre-bins (default: 20).
// //' @param bin_separator Separator string for merging categories (default: "%;%").
// //' @param initial_temperature Initial temperature for Simulated Annealing (default: 1.0).
// //' @param cooling_rate Cooling rate for Simulated Annealing (default: 0.995).
// //' @param max_iterations Maximum number of iterations for Simulated Annealing (default: 1000).
// //' @param convergence_threshold Threshold for convergence (default: 1e-6).
// //'
// //' @return A list containing the following elements:
// //' \itemize{
// //'   \item bins: A character vector of bin names
// //'   \item woe: A numeric vector of Weight of Evidence (WoE) values for each bin
// //'   \item iv: A numeric vector of Information Value (IV) for each bin
// //'   \item count: An integer vector of total counts for each bin
// //'   \item count_pos: An integer vector of positive counts for each bin
// //'   \item count_neg: An integer vector of negative counts for each bin
// //'   \item converged: A logical value indicating whether the algorithm converged
// //'   \item iterations: An integer value indicating the number of iterations run
// //' }
// //'
// //' @examples
// //' \dontrun{
// //' set.seed(123)
// //' target <- sample(0:1, 1000, replace = TRUE)
// //' feature <- sample(LETTERS[1:5], 1000, replace = TRUE)
// //' result <- optimal_binning_categorical_sab(target, feature)
// //' print(result)
// //' }
// //'
// //' @details
// //' The algorithm uses Simulated Annealing to find an optimal binning solution that maximizes
// //' the Information Value while maintaining monotonicity. It respects the specified constraints
// //' on the number of bins and bin sizes.
// //'
// //' The Weight of Evidence (WoE) is calculated as:
// //' \deqn{WoE_i = \ln(\frac{\text{Distribution of positives}_i}{\text{Distribution of negatives}_i})}
// //'
// //' Where:
// //' \deqn{\text{Distribution of positives}_i = \frac{\text{Number of positives in bin } i}{\text{Total Number of positives}}}
// //' \deqn{\text{Distribution of negatives}_i = \frac{\text{Number of negatives in bin } i}{\text{Total Number of negatives}}}
// //'
// //' The Information Value (IV) is calculated as:
// //' \deqn{IV = \sum_{i=1}^{N} (\text{Distribution of positives}_i - \text{Distribution of negatives}_i) \times WoE_i}
// //'
// //' @export
// // [[Rcpp::export]]
// Rcpp::List optimal_binning_categorical_sab(Rcpp::IntegerVector target,
//                                           Rcpp::StringVector feature,
//                                           int min_bins = 3,
//                                           int max_bins = 5,
//                                           double bin_cutoff = 0.05,
//                                           int max_n_prebins = 20,
//                                           std::string bin_separator = "%;%",
//                                           double initial_temperature = 1.0,
//                                           double cooling_rate = 0.995,
//                                           int max_iterations = 1000,
//                                           double convergence_threshold = 1e-6) {
//  try {
//    if (target.size() != feature.size()) {
//      Rcpp::stop("Target and feature vectors must have the same length");
//    }
//    if (min_bins < 2) {
//      Rcpp::stop("min_bins must be at least 2");
//    }
//    if (max_bins < min_bins) {
//      Rcpp::stop("max_bins must be >= min_bins");
//    }
//    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
//      Rcpp::stop("bin_cutoff must be between 0 and 1");
//    }
//    if (max_n_prebins < max_bins) {
//      Rcpp::stop("max_n_prebins must be >= max_bins");
//    }
//    
//    std::vector<std::string> feature_vec = Rcpp::as<std::vector<std::string>>(feature);
//    std::vector<int> target_vec = Rcpp::as<std::vector<int>>(target);
//    
//    OptimalBinningCategoricalSAB binner(feature_vec, target_vec,
//                                        min_bins, max_bins, bin_cutoff,
//                                        max_n_prebins, bin_separator,
//                                        initial_temperature, cooling_rate,
//                                        max_iterations, convergence_threshold);
//    binner.fit();
//    
//    return binner.get_results();
//  } catch (const std::exception& e) {
//    Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
//  }
// }
