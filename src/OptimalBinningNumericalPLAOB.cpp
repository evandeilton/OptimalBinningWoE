#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <omp.h>

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalPLAOB {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  struct Bin {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<Bin> bins;
  
  double calculate_woe(int count_pos, int count_neg, int total_pos, int total_neg) {
    double pos_rate = static_cast<double>(count_pos) / total_pos;
    double neg_rate = static_cast<double>(count_neg) / total_neg;
    if (pos_rate == 0 || neg_rate == 0) {
      return 0.0; // Avoid log(0)
    }
    return std::log(pos_rate / neg_rate);
  }
  
  double calculate_iv(double woe, int count_pos, int count_neg, int total_pos, int total_neg) {
    double pos_rate = static_cast<double>(count_pos) / total_pos;
    double neg_rate = static_cast<double>(count_neg) / total_neg;
    return (pos_rate - neg_rate) * woe;
  }
  
  void create_initial_bins() {
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    int n = feature.size();
    int initial_bins = std::min(max_bins, max_n_prebins);
    int bin_size = n / initial_bins;
    
    for (int i = 0; i < initial_bins; ++i) {
      int start = i * bin_size;
      int end = (i == initial_bins - 1) ? n : (i + 1) * bin_size;
      
      Bin bin;
      bin.lower = (i == 0) ? -std::numeric_limits<double>::infinity() : sorted_feature[start];
      bin.upper = (i == initial_bins - 1) ? std::numeric_limits<double>::infinity() : sorted_feature[end - 1];
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (int j = 0; j < n; ++j) {
        if (feature[j] > bin.lower && feature[j] <= bin.upper) {
          bin.count++;
          if (target[j] == 1) bin.count_pos++;
          else bin.count_neg++;
        }
      }
      
      if (bin.count > 0) {
        bins.push_back(bin);
      }
    }
  }
  
  void merge_low_frequency_bins() {
    int total_count = feature.size();
    double cutoff_count = bin_cutoff * total_count;
    
    for (size_t i = 0; i < bins.size() - 1; ++i) {
      if (bins[i].count < cutoff_count && static_cast<int>(bins.size()) > min_bins) {
        bins[i + 1].lower = bins[i].lower;
        bins[i + 1].count += bins[i].count;
        bins[i + 1].count_pos += bins[i].count_pos;
        bins[i + 1].count_neg += bins[i].count_neg;
        bins.erase(bins.begin() + i);
        --i;
      }
    }
  }
  
  void calculate_woe_iv() {
    int total_pos = 0;
    int total_neg = 0;
    for (const auto& bin : bins) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    for (auto& bin : bins) {
      bin.woe = calculate_woe(bin.count_pos, bin.count_neg, total_pos, total_neg);
      bin.iv = calculate_iv(bin.woe, bin.count_pos, bin.count_neg, total_pos, total_neg);
    }
  }
  
  void enforce_monotonicity() {
    if (bins.size() < 2) return;
    
    bool is_increasing = bins[1].woe > bins[0].woe;
    
    for (size_t i = 1; i < bins.size() - 1; ++i) {
      if ((is_increasing && bins[i + 1].woe < bins[i].woe) ||
          (!is_increasing && bins[i + 1].woe > bins[i].woe)) {
        // Merge bins i and i+1
        if (static_cast<int>(bins.size()) > min_bins) {
          bins[i].upper = bins[i + 1].upper;
          bins[i].count += bins[i + 1].count;
          bins[i].count_pos += bins[i + 1].count_pos;
          bins[i].count_neg += bins[i + 1].count_neg;
          bins.erase(bins.begin() + i + 1);
          --i;
        }
      }
    }
  }
  
  void adjust_boundaries() {
    if (bins.size() < 3) return;
    
#pragma omp parallel for
    for (size_t i = 1; i < bins.size() - 1; ++i) {
      double left = bins[i - 1].upper;
      double right = bins[i + 1].lower;
      double step = (right - left) / 100.0;
      
      double best_iv = bins[i - 1].iv + bins[i].iv;
      double best_boundary = bins[i].lower;
      
      for (double boundary = left + step; boundary < right; boundary += step) {
        Bin left_bin = bins[i - 1];
        Bin right_bin = bins[i];
        
        // Adjust counts
        for (size_t j = 0; j < feature.size(); ++j) {
          if (feature[j] > left_bin.upper && feature[j] <= boundary) {
            if (target[j] == 1) {
              left_bin.count_pos++;
              right_bin.count_pos--;
            } else {
              left_bin.count_neg++;
              right_bin.count_neg--;
            }
            left_bin.count++;
            right_bin.count--;
          }
        }
        
        // Recalculate WoE and IV
        int total_pos = left_bin.count_pos + right_bin.count_pos;
        int total_neg = left_bin.count_neg + right_bin.count_neg;
        
        double left_woe = calculate_woe(left_bin.count_pos, left_bin.count_neg, total_pos, total_neg);
        double right_woe = calculate_woe(right_bin.count_pos, right_bin.count_neg, total_pos, total_neg);
        double left_iv = calculate_iv(left_woe, left_bin.count_pos, left_bin.count_neg, total_pos, total_neg);
        double right_iv = calculate_iv(right_woe, right_bin.count_pos, right_bin.count_neg, total_pos, total_neg);
        
        double total_iv = left_iv + right_iv;
        
        if (total_iv > best_iv) {
          best_iv = total_iv;
          best_boundary = boundary;
        }
      }
      
#pragma omp critical
{
  bins[i - 1].upper = best_boundary;
  bins[i].lower = best_boundary;
}
    }
  }
  
  void ensure_min_bins() {
    while (static_cast<int>(bins.size()) < min_bins) {
      // Find the bin with the highest IV
      auto max_iv_bin = std::max_element(bins.begin(), bins.end(),
                                         [](const Bin& a, const Bin& b) { return a.iv < b.iv; });
      
      int index = std::distance(bins.begin(), max_iv_bin);
      double mid = (max_iv_bin->lower + max_iv_bin->upper) / 2;
      
      Bin new_bin;
      new_bin.lower = mid;
      new_bin.upper = max_iv_bin->upper;
      max_iv_bin->upper = mid;
      
      // Redistribute counts
      new_bin.count = 0;
      new_bin.count_pos = 0;
      new_bin.count_neg = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] > mid && feature[i] <= new_bin.upper) {
          if (target[i] == 1) {
            new_bin.count_pos++;
            max_iv_bin->count_pos--;
          } else {
            new_bin.count_neg++;
            max_iv_bin->count_neg--;
          }
          new_bin.count++;
          max_iv_bin->count--;
        }
      }
      
      bins.insert(bins.begin() + index + 1, new_bin);
    }
  }
  
  void ensure_max_bins() {
    while (static_cast<int>(bins.size()) > max_bins) {
      // Find the pair of adjacent bins with the lowest combined IV
      double min_combined_iv = std::numeric_limits<double>::max();
      size_t merge_index = 0;
      
      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double combined_iv = bins[i].iv + bins[i + 1].iv;
        if (combined_iv < min_combined_iv) {
          min_combined_iv = combined_iv;
          merge_index = i;
        }
      }
      
      // Merge the bins
      bins[merge_index].upper = bins[merge_index + 1].upper;
      bins[merge_index].count += bins[merge_index + 1].count;
      bins[merge_index].count_pos += bins[merge_index + 1].count_pos;
      bins[merge_index].count_neg += bins[merge_index + 1].count_neg;
      bins.erase(bins.begin() + merge_index + 1);
    }
  }
  
public:
  OptimalBinningNumericalPLAOB(const std::vector<double>& feature, const std::vector<int>& target,
                               int min_bins = 3, int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20)
    : feature(feature), target(target), min_bins(std::max(2, min_bins)), max_bins(std::max(min_bins, max_bins)),
      bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {}
  
  Rcpp::List fit() {
    create_initial_bins();
    merge_low_frequency_bins();
    
    int iterations = 0;
    double prev_total_iv = 0;
    
    while (iterations < 100) {
      calculate_woe_iv();
      enforce_monotonicity();
      ensure_min_bins();
      ensure_max_bins();
      
      adjust_boundaries();
      calculate_woe_iv();
      
      double total_iv = 0;
      for (const auto& bin : bins) {
        total_iv += bin.iv;
      }
      
      if (std::abs(total_iv - prev_total_iv) < 1e-6) {
        break;
      }
      
      prev_total_iv = total_iv;
      iterations++;
    }
    
    // Prepare output
    std::vector<std::string> bin_labels;
    std::vector<double> woe_values;
    std::vector<double> iv_values;
    std::vector<int> count_values;
    std::vector<int> count_pos_values;
    std::vector<int> count_neg_values;
    
    for (const auto& bin : bins) {
      std::ostringstream oss;
      if (bin.lower == -std::numeric_limits<double>::infinity()) {
        oss << "(-Inf;" << bin.upper << "]";
      } else if (bin.upper == std::numeric_limits<double>::infinity()) {
        oss << "(" << bin.lower << ";+Inf]";
      } else {
        oss << "(" << bin.lower << ";" << bin.upper << "]";
      }
      bin_labels.push_back(oss.str());
      woe_values.push_back(bin.woe);
      iv_values.push_back(bin.iv);
      count_values.push_back(bin.count);
      count_pos_values.push_back(bin.count_pos);
      count_neg_values.push_back(bin.count_neg);
    }
    
    // Apply WoE transformation to feature
    std::vector<double> woe_feature(feature.size());
#pragma omp parallel for
    for (size_t i = 0; i < feature.size(); ++i) {
      for (const auto& bin : bins) {
        if (feature[i] <= bin.upper) {
          woe_feature[i] = bin.woe;
          break;
        }
      }
    }
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woe_feature,
      Rcpp::Named("woebin") = Rcpp::DataFrame::create(
        Rcpp::Named("bin") = bin_labels,
        Rcpp::Named("woe") = woe_values,
        Rcpp::Named("iv") = iv_values,
        Rcpp::Named("count") = count_values,
        Rcpp::Named("count_pos") = count_pos_values,
        Rcpp::Named("count_neg") = count_neg_values
      )
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_plaob(Rcpp::IntegerVector target, Rcpp::NumericVector feature,
                                           int min_bins = 3, int max_bins = 5,
                                           double bin_cutoff = 0.05, int max_n_prebins = 20) {
  std::vector<int> target_vec(target.begin(), target.end());
  std::vector<double> feature_vec(feature.begin(), feature.end());
  OptimalBinningNumericalPLAOB binner(feature_vec, target_vec, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binner.fit();
}
