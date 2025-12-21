#ifndef OPTIMAL_BINNING_BASE_ALGORITHM_H
#define OPTIMAL_BINNING_BASE_ALGORITHM_H

#include "BinStructures.h"
#include "Utilities.h"
#include <vector>
#include <string>

namespace OptimalBinning {

/**
 * @brief Base class for numerical binning algorithms
 * 
 * Provides common functionality and interface for all numerical binning algorithms
 */
class BaseNumericalBinning {
protected:
  // Common parameters
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  double laplace_smoothing_;
  bool is_monotonic_;
  
  // Common state
  int total_pos_;
  int total_neg_;
  bool converged_;
  int iterations_run_;
  std::vector<std::string> warnings_;
  
  // Bins
  std::vector<NumericalBin> bins_;
  
public:
  BaseNumericalBinning(int min_bins, int max_bins, double bin_cutoff,
                      int max_n_prebins, double laplace_smoothing = DEFAULT_LAPLACE_SMOOTHING,
                      bool is_monotonic = true)
    : min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
      max_n_prebins_(max_n_prebins), laplace_smoothing_(laplace_smoothing),
      is_monotonic_(is_monotonic), total_pos_(0), total_neg_(0),
      converged_(false), iterations_run_(0) {}
  
  virtual ~BaseNumericalBinning() = default;
  
  // Pure virtual methods to be implemented by derived classes
  virtual void fit() = 0;
  
  // Common methods that can be overridden
  virtual void calculate_woe_iv() {
    // Count totals
    total_pos_ = 0;
    total_neg_ = 0;
    for (const auto& bin : bins_) {
      total_pos_ += bin.count_pos;
      total_neg_ += bin.count_neg;
    }
    
    if (total_pos_ == 0 || total_neg_ == 0) {
      return; // Cannot compute WoE/IV
    }
    
    // Calculate WoE and IV for each bin
    for (auto& bin : bins_) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, 
                             total_pos_, total_neg_, laplace_smoothing_);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg,
                           total_pos_, total_neg_, laplace_smoothing_);
    }
  }
  
  virtual bool check_monotonicity() const {
    if (bins_.size() <= 1) return true;
    
    std::vector<double> woe_values;
    woe_values.reserve(bins_.size());
    for (const auto& bin : bins_) {
      woe_values.push_back(bin.woe);
    }
    
    return is_monotonic(woe_values);
  }
  
  virtual void enforce_monotonicity() {
    if (bins_.size() <= 2 || check_monotonicity()) {
      return;
    }
    
    // Simple monotonicity enforcement: merge adjacent bins with opposite trends
    // This is a basic implementation; derived classes can override for better algorithms
    bool changed = true;
    while (changed && bins_.size() > min_bins_) {
      changed = false;
      for (size_t i = 0; i < bins_.size() - 1; ++i) {
        if ((bins_[i].woe > bins_[i+1].woe && bins_[i].woe > 0) ||
            (bins_[i].woe < bins_[i+1].woe && bins_[i].woe < 0)) {
          // Merge bins
          bins_[i].upper_bound = bins_[i+1].upper_bound;
          bins_[i].count_pos += bins_[i+1].count_pos;
          bins_[i].count_neg += bins_[i+1].count_neg;
          bins_[i].update_count();
          bins_.erase(bins_.begin() + i + 1);
          changed = true;
          break;
        }
      }
      if (changed) {
        calculate_woe_iv();
      }
    }
  }
  
  // Getters
  const std::vector<NumericalBin>& get_bins() const { return bins_; }
  bool is_converged() const { return converged_; }
  int get_iterations() const { return iterations_run_; }
  const std::vector<std::string>& get_warnings() const { return warnings_; }
};

/**
 * @brief Base class for categorical binning algorithms
 * 
 * Provides common functionality and interface for all categorical binning algorithms
 */
class BaseCategoricalBinning {
protected:
  // Common parameters
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double laplace_smoothing_;
  bool is_monotonic_;
  
  // Common state
  int total_pos_;
  int total_neg_;
  bool converged_;
  int iterations_run_;
  std::vector<std::string> warnings_;
  
  // Bins
  std::vector<CategoricalBin> bins_;
  
public:
  BaseCategoricalBinning(int min_bins, int max_bins, double bin_cutoff,
                        int max_n_prebins, const std::string& bin_separator,
                        double laplace_smoothing = DEFAULT_LAPLACE_SMOOTHING,
                        bool is_monotonic = true)
    : min_bins_(min_bins), max_bins_(max_bins), bin_cutoff_(bin_cutoff),
      max_n_prebins_(max_n_prebins), bin_separator_(bin_separator),
      laplace_smoothing_(laplace_smoothing), is_monotonic_(is_monotonic),
      total_pos_(0), total_neg_(0), converged_(false), iterations_run_(0) {}
  
  virtual ~BaseCategoricalBinning() = default;
  
  // Pure virtual methods to be implemented by derived classes
  virtual void fit() = 0;
  
  // Common methods that can be overridden
  virtual void calculate_woe_iv() {
    // Count totals
    total_pos_ = 0;
    total_neg_ = 0;
    for (const auto& bin : bins_) {
      total_pos_ += bin.count_pos;
      total_neg_ += bin.count_neg;
    }
    
    if (total_pos_ == 0 || total_neg_ == 0) {
      return; // Cannot compute WoE/IV
    }
    
    // Calculate WoE and IV for each bin
    for (auto& bin : bins_) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, 
                             total_pos_, total_neg_, laplace_smoothing_);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg,
                          total_pos_, total_neg_, laplace_smoothing_);
    }
  }
  
  virtual bool check_monotonicity() const {
    if (bins_.size() <= 1) return true;
    
    std::vector<double> woe_values;
    woe_values.reserve(bins_.size());
    for (const auto& bin : bins_) {
      woe_values.push_back(bin.woe);
    }
    
    return is_monotonic(woe_values);
  }
  
  virtual void enforce_monotonicity() {
    if (bins_.size() <= 2 || check_monotonicity()) {
      return;
    }
    
    // Simple monotonicity enforcement: merge adjacent bins with opposite trends
    bool changed = true;
    while (changed && bins_.size() > min_bins_) {
      changed = false;
      for (size_t i = 0; i < bins_.size() - 1; ++i) {
        if ((bins_[i].woe > bins_[i+1].woe && bins_[i].woe > 0) ||
            (bins_[i].woe < bins_[i+1].woe && bins_[i].woe < 0)) {
          // Merge bins
          bins_[i].categories.insert(bins_[i].categories.end(),
                                   bins_[i+1].categories.begin(),
                                   bins_[i+1].categories.end());
          bins_[i].count_pos += bins_[i+1].count_pos;
          bins_[i].count_neg += bins_[i+1].count_neg;
          bins_[i].update_total_count();
          bins_.erase(bins_.begin() + i + 1);
          changed = true;
          break;
        }
      }
      if (changed) {
        calculate_woe_iv();
      }
    }
  }
  
  // Helper to create bin label from categories
  std::string create_bin_label(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    if (categories.size() == 1) return categories[0];
    
    std::string label = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      label += bin_separator_ + categories[i];
    }
    return label;
  }
  
  // Getters
  const std::vector<CategoricalBin>& get_bins() const { return bins_; }
  bool is_converged() const { return converged_; }
  int get_iterations() const { return iterations_run_; }
  const std::vector<std::string>& get_warnings() const { return warnings_; }
};

} // namespace OptimalBinning

#endif // OPTIMAL_BINNING_BASE_ALGORITHM_H

