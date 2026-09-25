// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <numeric>

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

/**
 * Core class implementing Optimal Binning for numerical variables using Branch and Bound algorithm.
 * 
 * This class transforms continuous numerical variables into optimal discrete bins 
 * based on their relationship with a binary target variable, maximizing predictive power
 * while maintaining statistical stability and interpretability constraints.
 */
class OBN_BB {
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
  
  std::vector<NumericalBin> bins; // Use shared NumericalBin structure
  bool converged;
  int iterations_run;
  
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
   * One bin per distinct value: (-Inf, u0], (u0, u1], ..., (u_{k-2}, +Inf].
   * A non-finite value (-Inf) is never used as a boundary: it produced a bin
   * "(-Inf;-Inf]" and a cutpoint vector one element shorter than the bins.
   *
   * @param unique_values Distinct values in ascending order
   */
  void bins_from_unique_values(const std::vector<double>& unique_values) {
    bins.clear();
    double lower = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < unique_values.size(); ++i) {
      if (!std::isfinite(unique_values[i])) continue;
      NumericalBin bin;
      bin.lower_bound = lower;
      bin.upper_bound = unique_values[i];
      bins.push_back(bin);
      lower = unique_values[i];
    }
    NumericalBin last;
    last.lower_bound = lower;
    last.upper_bound = std::numeric_limits<double>::infinity();
    bins.push_back(last);
  }

  /**
   * Handle the special case where feature has <= 2 unique values
   * This creates bins directly without optimization
   * 
   * @param unique_values Vector of unique values in the feature
   */
  void handle_two_or_fewer_unique_values(const std::vector<double>& unique_values) {
    // Construct bins based on unique values
    bins_from_unique_values(unique_values);
    
    // Assign observations to bins: first bin whose upper bound is >= the
    // value. The former scan widened every bin by an absolute 1e-10, so for a
    // feature whose two values are closer than that (e.g. 1e-11 and 2e-11)
    // every observation landed in the first bin and the second stayed empty.
    for (size_t i = 0; i < feature.size(); ++i) {
      double val = feature[i];
      if (std::isnan(val)) continue;
      int tgt = target[i];
      size_t idx = 0;
      while (idx + 1 < bins.size() && val > bins[idx].upper_bound) ++idx;
      NumericalBin &bin = bins[idx];
      if (tgt == 1) bin.count_pos++;
      else bin.count_neg++;
      bin.count++;
    }
    
    compute_woe_iv();
    converged = true;
    iterations_run = 0;
  }
  
  /**
   * Compute a quantile from an ALREADY SORTED vector (no copy/sort overhead).
   *
   * @param sorted_data Sorted vector of values
   * @param q Quantile value between 0 and 1
   * @return The q-th quantile value
   */
  static double quantile(const std::vector<double>& sorted_data, double q) {
    if (sorted_data.empty()) return 0.0;
    if (q <= 0.0) return sorted_data.front();
    if (q >= 1.0) return sorted_data.back();

    double idx_exact = q * static_cast<double>(sorted_data.size() - 1);
    size_t idx_lower = static_cast<size_t>(std::floor(idx_exact));
    size_t idx_upper = static_cast<size_t>(std::ceil(idx_exact));

    if (idx_lower == idx_upper) return sorted_data[idx_lower];

    double weight_upper = idx_exact - static_cast<double>(idx_lower);
    double weight_lower = 1.0 - weight_upper;
    return weight_lower * sorted_data[idx_lower] + weight_upper * sorted_data[idx_upper];
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
    
    // Sort once; distinct values and quantiles are both read from this copy
    std::vector<double> sorted_feature = clean_feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());

    // Get unique values
    std::vector<double> unique_values = sorted_feature;
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), unique_values.end());
    
    // Special case for few unique values
    if (unique_values.size() <= 2) {
      handle_two_or_fewer_unique_values(unique_values);
      return;
    }
    
    // If limited unique values, create a bin for each value
    if (unique_values.size() <= static_cast<size_t>(min_bins)) {
      bins_from_unique_values(unique_values);
    } else {
      // Use quantile-based initial binning for better distribution.
      int n_prebins = std::min(static_cast<int>(unique_values.size()), max_n_prebins);

      std::vector<double> quantiles;
      for (int i = 1; i < n_prebins; ++i) {
        double q = static_cast<double>(i) / n_prebins;
        double qval = quantile(sorted_feature, q);
        quantiles.push_back(qval);
      }
      
      // Remove duplicate quantiles (can happen with skewed distributions).
      // Non-finite quantiles (from -Inf/+Inf observations) are dropped: as a
      // boundary they only produce bins such as (-Inf;-Inf] or (+Inf;+Inf].
      quantiles.erase(std::remove_if(quantiles.begin(), quantiles.end(),
                                     [](double v) { return !std::isfinite(v); }),
                      quantiles.end());
      std::sort(quantiles.begin(), quantiles.end());
      quantiles.erase(std::unique(quantiles.begin(), quantiles.end()), quantiles.end());
      
      // Create bins based on quantiles
      bins.clear();
      bins.resize(quantiles.size() + 1);

      for (size_t i = 0; i < bins.size(); ++i) {
        if (bins.size() == 1) {
          bins[i].lower_bound = -std::numeric_limits<double>::infinity();
          bins[i].upper_bound = std::numeric_limits<double>::infinity();
        } else if (i == 0) {
          bins[i].lower_bound = -std::numeric_limits<double>::infinity();
          bins[i].upper_bound = quantiles[i];
        } else if (i == bins.size() - 1) {
          bins[i].lower_bound = quantiles[i - 1];
          bins[i].upper_bound = std::numeric_limits<double>::infinity();
        } else {
          bins[i].lower_bound = quantiles[i - 1];
          bins[i].upper_bound = quantiles[i];
        }
        
        bins[i].count_pos = 0;
        bins[i].count_neg = 0;
        bins[i].count = 0;
        bins[i].woe = 0.0;
        bins[i].iv = 0.0;
      }
    }
    
    // Extract upper boundaries for binary search
    std::vector<double> uppers;
    uppers.reserve(bins.size());
    for (auto &b : bins) {
      uppers.push_back(b.upper_bound);
    }
    
    // Assign observations to bins using binary search for performance
    for (size_t i = 0; i < clean_feature.size(); ++i) {
      double val = clean_feature[i];
      int tgt = clean_target[i];
      
      // Binary search for the correct bin:
      // Find the first bin with upper boundary >= value.
      // The "+ EPSILON" that used to be added here turned the comparison into a
      // strict one, so a value landing exactly on a bin's upper boundary was
      // pushed into the NEXT bin -- i.e. [a, b) behaviour, contradicting both
      // this comment and the emitted "(a;b]" labels.
      auto it = std::lower_bound(uppers.begin(), uppers.end(), val);
      size_t idx = it - uppers.begin();
      if (idx >= bins.size()) idx = bins.size() - 1;

      // Count by target value
      if (tgt == 1) {
        bins[idx].count_pos++;
      } else {
        bins[idx].count_neg++;
      }
      bins[idx].count++;
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
      total_count += bin.total();
    }
    
    double cutoff_count = bin_cutoff * total_count;
    
    // Iteratively merge rare bins with neighbors
    for (auto it = bins.begin(); it != bins.end();) {
      int bin_count = it->total();
      
      // Check if bin is too small and we can still merge (respecting min_bins)
      if (bin_count < cutoff_count && bins.size() > static_cast<size_t>(min_bins)) {
        if (it != bins.begin()) {
          // Merge with previous bin (preferred)
          auto prev = std::prev(it);
          prev->merge_with(*it); // Uses shared merge_with logic
          it = bins.erase(it);
        } else if (std::next(it) != bins.end()) {
          // Merge with next bin if this is the first bin
          auto nxt = std::next(it);
          // Note for next merge: numerical merge generally extends range.
          // Shared merge_with logic: upper_bound = other.upper_bound, etc.
          // Merging it into nxt: nxt extended downwards.
          // The shared merge_with assumes merging 'other' into 'this'.
          // So nxt->merge_with(*it) would mean nxt.upper_bound updated to it.upper_bound (wrong for numerical merge if 'it' is below 'nxt').
          // Actually numerical merge logic in header: upper_bound = other.upper_bound.
          // So if we merge it (lower) into nxt (higher), we want nxt.lower_bound = it.lower_bound.
          // Shared struct doesn't handle lower bound merge automatically for all cases, mainly upper.
          // Let's do it manually to be safe for Numerical logic.
          
          nxt->lower_bound = it->lower_bound; // Extend lower bound down
          nxt->count_pos += it->count_pos;
          nxt->count_neg += it->count_neg;
          nxt->count += it->count;
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
    double pos_denom = total_pos + static_cast<double>(bins.size()) * 0.5;
    double neg_denom = total_neg + static_cast<double>(bins.size()) * 0.5;
    
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
   */
  void enforce_monotonicity(bool complete = false) {
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
    std::vector<NumericalBin> temp_bins = bins;
    
    // Simulate increasing monotonicity
    for (size_t i = 1; i < temp_bins.size(); ++i) {
      if (temp_bins[i].woe < temp_bins[i-1].woe - EPSILON) {
        // Merge current bin into previous
        temp_bins[i-1].merge_with(temp_bins[i]); // NumericalBin merge extends upper bound
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
        temp_bins[i-1].merge_with(temp_bins[i]);
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
        std::prev(it)->merge_with(*it);
        it = bins.erase(it);
      } else {
        ++it;
      }
    }

    // Recompute WoE/IV after merges
    compute_woe_iv();

    // The pass above compares the WoE a merged bin had BEFORE it absorbed its
    // neighbour, so a single pass can leave violations behind (e.g. WoE
    // -0.76, -0.26, 0.91, 0.76 was returned for is_monotonic = TRUE). On the
    // final call (complete = true) keep merging the first violating pair, with
    // fresh WoE, until the sequence is monotone in the chosen direction or
    // min_bins is reached. Intermediate calls keep the single pass, so a fit
    // whose result was already monotone is unchanged.
    while (complete && bins.size() > static_cast<size_t>(min_bins)) {
      size_t v = 0;
      for (size_t i = 1; i < bins.size(); ++i) {
        if ((prefer_increasing && bins[i].woe < bins[i - 1].woe - EPSILON) ||
            (!prefer_increasing && bins[i].woe > bins[i - 1].woe + EPSILON)) {
          v = i;
          break;
        }
      }
      if (v == 0) break;
      bins[v - 1].merge_with(bins[v]);
      bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(v));
      compute_woe_iv();
    }
  }
  

  /**
   * Remove bins that received no observation.
   *
   * Interpolated quantiles on heavily tied data can put two boundaries between
   * the same pair of consecutive values, and min_bins then kept the resulting
   * empty bin alive to the output, where its WoE/IV are pure smoothing
   * artefacts. An empty bin is folded into its left neighbour (the first one
   * into its right neighbour), which leaves every other bin untouched.
   */
  void drop_empty_bins() {
    for (size_t i = 0; i < bins.size() && bins.size() > 1; ) {
      if (bins[i].count_pos + bins[i].count_neg > 0) { ++i; continue; }
      if (i > 0) {
        bins[i - 1].upper_bound = bins[i].upper_bound;
      } else {
        bins[1].lower_bound = bins[0].lower_bound;
      }
      bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));
    }
  }

public:
  /**
   * Constructor for the OBN_BB algorithm
   */
  OBN_BB(const std::vector<double> &feature_,
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
   */
  Rcpp::List fit() {
    // Step 1: Initial prebinning
    prebinning();
    
    if (!converged) {  // Only proceed if not already converged
      // Step 2: Merge rare bins
      merge_rare_bins();
      compute_woe_iv();
      
      // Step 3: Enforce monotonicity if requested
      if (is_monotonic) {
        enforce_monotonicity();
      }
      
      iterations_run = 0;
      
      // Step 4: Branch and Bound optimization
      while (bins.size() > static_cast<size_t>(max_bins) && iterations_run < max_iterations) {
        // Find bin with minimum IV contribution
        auto min_iv_it = std::min_element(bins.begin(), bins.end(),
                                          [](const NumericalBin &a, const NumericalBin &b) { return a.iv < b.iv; });
        
        // Merge the minimum IV bin with an adjacent bin
        if (min_iv_it != bins.begin()) {
          // Prefer merging with previous bin
          auto prev = std::prev(min_iv_it);
          prev->merge_with(*min_iv_it);
          bins.erase(min_iv_it);
        } else {
          // If it's the first bin, merge with the next one
          auto nxt = std::next(min_iv_it);
          // Manually handle lower bound extension for next bin
          nxt->lower_bound = min_iv_it->lower_bound;
          nxt->count_pos += min_iv_it->count_pos;
          nxt->count_neg += min_iv_it->count_neg;
          nxt->count += min_iv_it->count;
          bins.erase(min_iv_it);
        }
        
        // Recompute metrics after merge
        compute_woe_iv();
        
        // Re-enforce monotonicity if needed
        if (is_monotonic) {
          enforce_monotonicity();
        }
        
        // The loop only runs while there are more than max_bins bins, so the
        // former exit on |delta IV| < convergence_threshold could only ever
        // return too many bins; merging continues until max_bins is met.
        iterations_run++;
      }

      // Reaching the bin-count target is a successful stopping state, exactly
      // like meeting the IV tolerance above. Only exhausting max_iterations
      // leaves converged == false. This also covers the few-unique-values path
      // in prebinning(), which produces one bin per value and never enters the
      // loop at all.
      converged = converged || (bins.size() <= static_cast<size_t>(max_bins));

      // Guarantee the documented monotonic WoE on the returned bins.
      if (is_monotonic) {
        enforce_monotonicity(true);
      }

      const size_t before = bins.size();
      drop_empty_bins();
      if (bins.size() != before) compute_woe_iv();
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
      std::string lower_str = std::isinf(bin.lower_bound) ? "-Inf" : std::to_string(bin.lower_bound);
      std::string upper_str = std::isinf(bin.upper_bound) ? "+Inf" : std::to_string(bin.upper_bound);
      std::string bin_label = "(" + lower_str + ";" + upper_str + "]";
      
      bin_labels.push_back(bin_label);
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      counts.push_back(bin.total());
      counts_pos.push_back(bin.count_pos);
      counts_neg.push_back(bin.count_neg);
      
      // Store cutpoints (excluding infinity)
      if (!std::isinf(bin.upper_bound)) {
        cutpoints.push_back(bin.upper_bound);
      }
    }
    
    // Create bin IDs (1-based indexing for R)
    Rcpp::NumericVector ids(bin_labels.size());
    for(size_t i = 0; i < bin_labels.size(); i++) {
      ids[i] = static_cast<double>(i + 1);
    }
    
    // Calculate total IV
    double total_iv = 0.0;
    for (R_xlen_t i = 0; i < iv_values.size(); i++) {
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
  OBN_BB obb(
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
  Rcpp::stop("Error in optimal binning: " + std::string(e.what()));
 }
}