/**
 * @file bin_structures.h
 * @brief Unified bin structures for numerical and categorical binning
 */

#ifndef OPTIMAL_BINNING_BIN_STRUCTURES_H
#define OPTIMAL_BINNING_BIN_STRUCTURES_H

#include "optimal_binning_common.h"
#include <vector>
#include <string>

namespace OptimalBinning {

// =============================================================================
// NUMERICAL BIN STRUCTURE
// =============================================================================

/**
 * @brief Bin structure for numerical variables
 */
struct NumericalBin {
  double lower_bound = -std::numeric_limits<double>::infinity();
  double upper_bound = std::numeric_limits<double>::infinity();
  int count = 0;
  int count_pos = 0;
  int count_neg = 0;
  double woe = 0.0;
  double iv = 0.0;
  double entropy = 0.0;
  double divergence = 0.0; // For DMIV algorithms
  double centroid = 0.0;   // For KMB (K-Means) algorithms

  // Default constructor
  NumericalBin() = default;

  // Main constructor with all parameters - delegates to this one
  NumericalBin(double lower, double upper, int c, int p, int n, double w, double i)
    : lower_bound(lower), upper_bound(upper), count(c), count_pos(p), count_neg(n),
      woe(w), iv(i), entropy(0.0), divergence(0.0), centroid((lower + upper) / 2.0) {}

  // 2-arg Constructor (lower, upper) - delegates to main
  NumericalBin(double lower, double upper)
    : NumericalBin(lower, upper, 0, 0, 0, 0.0, 0.0) {}

  // 5-arg Constructor (lower, upper, count, pos, neg) - delegates to main
  NumericalBin(double lower, double upper, int c, int p, int n)
    : NumericalBin(lower, upper, c, p, n, 0.0, 0.0) {}

  /// Get total count (convenience method)
  int total() const { return count_pos + count_neg; }
  
  /// Get event rate
  double event_rate() const {
    return count > 0 ? static_cast<double>(count_pos) / count : 0.0;
  }
  
  /// Get bin name (interval)
  std::string name() const {
    std::string l = (std::isinf(lower_bound) && lower_bound < 0) ? "-Inf" : std::to_string(lower_bound);
    std::string u = (std::isinf(upper_bound) && upper_bound > 0) ? "Inf" : std::to_string(upper_bound);
    return "[" + l + ", " + u + ")";
  }
  
  /// Calculate WoE and IV with smoothing
  void calculate_metrics(long total_pos, long total_neg) {
    if (total_pos <= 0 || total_neg <= 0) return;
    
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    double dist_pos = (count_pos + prior_pos) / (total_pos + BAYESIAN_PRIOR_STRENGTH);
    double dist_neg = (count_neg + prior_neg) / (total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    woe = std::log(dist_pos / std::max(dist_neg, EPSILON));
    iv = (dist_pos - dist_neg) * woe;
  }

  /// Validate bin integrity
  bool is_valid() const {
    return count >= 0 && count_pos >= 0 && count_neg >= 0 &&
           count == count_pos + count_neg && lower_bound <= upper_bound;
  }
  
  /// Merge with another bin
  void merge_with(const NumericalBin& other) {
    // Weighted centroid update
    double total_count = count + other.count;
    if (total_count > 0) {
      centroid = (centroid * count + other.centroid * other.count) / total_count;
    }

    upper_bound = other.upper_bound;
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    // Note: divergence/entropy need recalculation after merge
  }

  // Compatibility methods for Sketch
  void add_value(int is_positive) {
    count++;
    if (is_positive) count_pos++;
    else count_neg++;
    // implicit update of event_rate if accessed via method
  }
  
  void add_counts(int p_count, int n_count) {
    count += (p_count + n_count);
    count_pos += p_count;
    count_neg += n_count;
  }
  
  void update_event_rate() {
     // No-op: event_rate() is calculated on-the-fly via method
  }

  /// Check if value is in bin [lower, upper)
  /// Note: Half-open interval - includes lower, excludes upper
  /// Exception: Last bin typically includes upper bound
  bool contains(double value) const {
    return value >= lower_bound && value < upper_bound;
  }

  /// Check if value is in bin [lower, upper] (inclusive on both ends)
  /// Use this for last bin to capture maximum value
  bool contains_inclusive(double value) const {
    return value >= lower_bound && value <= upper_bound;
  }
};

// =============================================================================
// CATEGORICAL BIN STRUCTURE
// =============================================================================

/**
 * @brief Bin structure for categorical variables
 */
struct CategoricalBin {
  std::vector<std::string> categories;
  int count = 0;
  int count_pos = 0;
  int count_neg = 0;
  double woe = 0.0;
  double iv = 0.0;
  double divergence = 0.0; // For DMIV algorithms
  
  CategoricalBin() : 
    count(0), count_pos(0), count_neg(0), woe(0.0), iv(0.0), divergence(0.0) {}

  // Constructor for compatibility with DP/other algorithms (single category)
  CategoricalBin(const std::string& cat, double w, double i, int c, int p, int n)
    : categories({cat}), count(c), count_pos(p), count_neg(n), woe(w), iv(i), divergence(0.0) {}

  // Constructor for compatibility (vector of categories)
  CategoricalBin(const std::vector<std::string>& cats, double w, double i, int c, int p, int n)
    : categories(cats), count(c), count_pos(p), count_neg(n), woe(w), iv(i), divergence(0.0) {}

  /// Get total count
  int total() const { return count_pos + count_neg; }
  
  /// Get event rate
  double event_rate() const {
    return count > 0 ? static_cast<double>(count_pos) / count : 0.0;
  }
  
  /// Update total count from pos + neg
  void update_count() {
    count = count_pos + count_neg;
  }
  
  /// Merge with another bin
  void merge_with(const CategoricalBin& other, const std::string& separator = "%;%") {
    categories.insert(categories.end(), other.categories.begin(), other.categories.end());
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    update_count();
  }
  
  /// Get bin name as joined categories
  std::string name(const std::string& separator = "%;%") const {
    std::string result;
    for (size_t i = 0; i < categories.size(); ++i) {
      if (i > 0) result += separator;
      result += categories[i];
    }
    return result;
  }
  
  /// Calculate WoE and IV with smoothing
  void calculate_metrics(long total_pos, long total_neg) {
    if (total_pos <= 0 || total_neg <= 0) return;
    
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) / (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;
    
    double dist_pos = (count_pos + prior_pos) / (total_pos + BAYESIAN_PRIOR_STRENGTH);
    double dist_neg = (count_neg + prior_neg) / (total_neg + BAYESIAN_PRIOR_STRENGTH);
    
    woe = std::log(dist_pos / std::max(dist_neg, EPSILON));
    iv = (dist_pos - dist_neg) * woe;
  }

  /// Calculate Jensen-Shannon divergence from another bin
  /// @param other Other bin to compare with
  /// @param total_pos Total positive count across all bins
  /// @param total_neg Total negative count across all bins
  /// @param alpha Laplace smoothing parameter (default: DEFAULT_LAPLACE_ALPHA)
  /// @return Jensen-Shannon divergence value
  double divergence_from(const CategoricalBin& other, long total_pos, long total_neg,
                        double alpha = DEFAULT_LAPLACE_ALPHA) const {
    double p1 = (count_pos + alpha) / (total_pos + alpha * 2);
    double n1 = (count_neg + alpha) / (total_neg + alpha * 2);
    double p2 = (other.count_pos + alpha) / (total_pos + alpha * 2);
    double n2 = (other.count_neg + alpha) / (total_neg + alpha * 2);

    double p_avg = (p1 + p2) / 2;
    double n_avg = (n1 + n2) / 2;

    double div_p1 = p1 > EPSILON ? p1 * std::log(p1 / p_avg) : 0;
    double div_n1 = n1 > EPSILON ? n1 * std::log(n1 / n_avg) : 0;
    double div_p2 = p2 > EPSILON ? p2 * std::log(p2 / p_avg) : 0;
    double div_n2 = n2 > EPSILON ? n2 * std::log(n2 / n_avg) : 0;

    return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
  }
};

// =============================================================================
// CATEGORY STATISTICS (for preprocessing)
// =============================================================================

/**
 * @brief Statistics for a single category
 */
struct CategoryStats {
  std::string category;
  int count = 0;
  int count_pos = 0;
  int count_neg = 0;
  double event_rate = 0.0;
  double woe = 0.0;
  double iv = 0.0;

  // Default constructor
  CategoryStats() = default;

  // Constructor with category and WoE
  CategoryStats(const std::string& c, int p, int n, double w)
      : category(c), count(p+n), count_pos(p), count_neg(n),
        event_rate(count > 0 ? static_cast<double>(p) / (p+n) : 0.0),
        woe(w), iv(0.0) {}

  /// Update with a new observation
  void update(int is_positive) {
    count++;
    if (is_positive) {
      count_pos++;
    } else {
      count_neg++;
    }
    event_rate = static_cast<double>(count_pos) / count;
  }

  /// Merge with another category
  void merge_with(const CategoryStats& other, const std::string& separator = "%;%") {
    if (!category.empty() && !other.category.empty()) {
      category += separator + other.category;
    } else if (category.empty()) {
      category = other.category;
    }
    count += other.count;
    count_pos += other.count_pos;
    count_neg += other.count_neg;
    if (count > 0) {
      event_rate = static_cast<double>(count_pos) / count;
    }
  }

  /// Calculate WoE and IV using shared totals
  void calculate_metrics(int total_pos, int total_neg) {
    double total_p = std::max(static_cast<double>(total_pos), 1.0);
    double total_n = std::max(static_cast<double>(total_neg), 1.0);

    double ev_rate = std::max(count_pos / total_p, EPSILON);
    double nonev_rate = std::max(count_neg / total_n, EPSILON);

    woe = std::log(ev_rate / nonev_rate);
    iv = (ev_rate - nonev_rate) * woe;
  }
  
  // Compatibility methods
  void add_value(int is_positive) {
    update(is_positive);
  }
  
  void add_counts(int p_count, int n_count) {
    count += (p_count + n_count);
    count_pos += p_count;
    count_neg += n_count;
    // event_rate updated implicitly? No, `update` updates it.
    // Need to update event_rate manually here or use a helper
    if (count > 0) event_rate = (double)count_pos / count;
  }
  
  void update_event_rate() {
     if (count > 0) event_rate = (double)count_pos / count;
     else event_rate = 0.0;
  }
};

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_BIN_STRUCTURES_H
