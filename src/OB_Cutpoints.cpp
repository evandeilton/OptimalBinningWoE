// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <sstream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;


// Helper function to calculate WoE and IV
void calculate_woe_iv(std::vector<int>& count_pos, std::vector<int>& count_neg, 
                      std::vector<double>& woe, std::vector<double>& iv,
                      int total_pos, int total_neg) {
  for (size_t i = 0; i < count_pos.size(); ++i) {
    double pos_rate = static_cast<double>(count_pos[i]) / total_pos;
    double neg_rate = static_cast<double>(count_neg[i]) / total_neg;
    
    // Handling edge cases to avoid log(0)
    if (pos_rate == 0) pos_rate = 0.0001;
    if (neg_rate == 0) neg_rate = 0.0001;
    
    woe[i] = std::log(pos_rate / neg_rate);
    iv[i] = (pos_rate - neg_rate) * woe[i];
  }
}

// Helper function to format bin ranges
std::string format_bin_range(double lower, double upper) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2);
  
  if (std::isinf(lower) && lower < 0) {
    oss << "[-Inf;";
  } else {
    oss << "[" << lower << ";";
  }
  
  if (std::isinf(upper) && upper > 0) {
    oss << "+Inf]";
  } else {
    oss << upper << ")";
  }
  
  return oss.str();
}

// [[Rcpp::export]]
List binning_numerical_cutpoints(NumericVector feature, IntegerVector target, 
                                 NumericVector cutpoints) {
  int n = feature.size();
  int num_bins = cutpoints.size() + 1;
  
  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  std::vector<double> bin_edges(num_bins + 1);
  
  int total_pos = 0, total_neg = 0;
  
  // Sort cutpoints
  std::sort(cutpoints.begin(), cutpoints.end());
  
  // Set bin edges
  bin_edges[0] = -INFINITY;
  for (int i = 0; i < cutpoints.size(); ++i) {
    bin_edges[i + 1] = cutpoints[i];
  }
  bin_edges[num_bins] = INFINITY;
  
  // Assign observations to bins
  for (int i = 0; i < n; ++i) {
    int bin = std::upper_bound(bin_edges.begin(), bin_edges.end(), feature[i]) - bin_edges.begin() - 1;
    count[bin]++;
    if (target[i] == 1) {
      count_pos[bin]++;
      total_pos++;
    } else {
      count_neg[bin]++;
      total_neg++;
    }
  }
  
  // Calculate WoE and IV
  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);
  
  // Format bin ranges
  CharacterVector bin_ranges(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    bin_ranges[i] = format_bin_range(bin_edges[i], bin_edges[i + 1]);
  }
  
  // Create woebin DataFrame
  DataFrame woebin = DataFrame::create(
    Named("bin") = bin_ranges,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );
  
  // Assign WoE to feature values
  NumericVector woefeature(n);
  for (int i = 0; i < n; ++i) {
    int bin = std::upper_bound(bin_edges.begin(), bin_edges.end(), feature[i]) - bin_edges.begin() - 1;
    woefeature[i] = woe[bin];
  }
  
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}

// [[Rcpp::export]]
List binning_categorical_cutpoints(CharacterVector feature, IntegerVector target, 
                                   CharacterVector cutpoints) {
  int n = feature.size();
  int num_bins = cutpoints.size();
  
  std::unordered_map<std::string, int> category_to_bin;
  for (int i = 0; i < num_bins; ++i) {
    std::string bin_categories = as<std::string>(cutpoints[i]);
    size_t pos = 0;
    while ((pos = bin_categories.find("+")) != std::string::npos) {
      std::string category = bin_categories.substr(0, pos);
      category_to_bin[category] = i;
      bin_categories.erase(0, pos + 1);
    }
    category_to_bin[bin_categories] = i;
  }
  
  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  
  int total_pos = 0, total_neg = 0;
  
  // Assign observations to bins
  for (int i = 0; i < n; ++i) {
    std::string cat = as<std::string>(feature[i]);
    int bin = category_to_bin[cat];
    count[bin]++;
    if (target[i] == 1) {
      count_pos[bin]++;
      total_pos++;
    } else {
      count_neg[bin]++;
      total_neg++;
    }
  }
  
  // Calculate WoE and IV
  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);
  
  // Create woebin DataFrame
  DataFrame woebin = DataFrame::create(
    Named("bin") = cutpoints,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );
  
  // Assign WoE to feature values
  NumericVector woefeature(n);
  for (int i = 0; i < n; ++i) {
    std::string cat = as<std::string>(feature[i]);
    int bin = category_to_bin[cat];
    woefeature[i] = woe[bin];
  }
  
  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}