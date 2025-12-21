#ifndef OPTIMAL_BINNING_BIN_STRUCTURES_H
#define OPTIMAL_BINNING_BIN_STRUCTURES_H

#include <vector>
#include <string>
#include <cmath>
#include <limits>

namespace OptimalBinning {

/**
 * @brief Base structure for a bin in numerical binning
 * 
 * Contains boundaries and statistics for a bin
 */
struct NumericalBin {
  double lower_bound;    // Lower bound of the bin (inclusive)
  double upper_bound;    // Upper bound of the bin (inclusive)
  int count;             // Total observations in bin
  int count_pos;         // Count of positive class (target=1)
  int count_neg;         // Count of negative class (target=0)
  double woe;            // Weight of Evidence
  double iv;             // Information Value contribution
  
  NumericalBin() : lower_bound(0.0), upper_bound(0.0), count(0), 
    count_pos(0), count_neg(0), woe(0.0), iv(0.0) {}
  
  // Validate bin integrity
  bool is_valid() const {
    return count >= 0 && count_pos >= 0 && count_neg >= 0 && 
      count == count_pos + count_neg && lower_bound <= upper_bound;
  }
  
  // Get total count
  int get_total_count() const {
    return count;
  }
  
  // Update count from pos/neg
  void update_count() {
    count = count_pos + count_neg;
  }
  
  // Get positive rate with safety
  double get_positive_rate() const {
    return (count > 0) ? static_cast<double>(count_pos) / count : 0.0;
  }
};

/**
 * @brief Base structure for a bin in categorical binning
 * 
 * Contains categories and statistics for a bin
 */
struct CategoricalBin {
  std::vector<std::string> categories;  // Categories in this bin
  int count_pos;                        // Count of positive class
  int count_neg;                        // Count of negative class
  double woe;                           // Weight of Evidence
  double iv;                            // Information Value contribution
  int total_count;                      // Total observations (cached)
  
  CategoricalBin() : count_pos(0), count_neg(0), woe(0.0), iv(0.0), total_count(0) {
    categories.reserve(10);
  }
  
  // Copy constructor
  CategoricalBin(const CategoricalBin& other) = default;
  
  // Move constructor
  CategoricalBin(CategoricalBin&& other) noexcept = default;
  
  // Assignment operators
  CategoricalBin& operator=(const CategoricalBin& other) = default;
  CategoricalBin& operator=(CategoricalBin&& other) noexcept = default;
  
  // Helper function to get total count
  int get_total_count() const { 
    return total_count;
  }
  
  // Update total count (call after modifying counts)
  void update_total_count() {
    total_count = count_pos + count_neg;
  }
  
  // Get positive rate with safety
  double get_positive_rate() const {
    return (total_count > 0) ? static_cast<double>(count_pos) / total_count : 0.0;
  }
  
  // Validate bin integrity
  bool is_valid() const {
    return !categories.empty() && total_count >= 0 && count_pos >= 0 && count_neg >= 0 &&
      (count_pos + count_neg == total_count);
  }
};

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_BIN_STRUCTURES_H

