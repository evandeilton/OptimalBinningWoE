#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// [[Rcpp::export]]
Rcpp::List OptimalBinningNumericMIP2Prep(Rcpp::IntegerVector target, Rcpp::NumericVector feature, 
                                         int max_n_prebins = 20) {
  int N = target.size();
  if (feature.size() != N) {
    Rcpp::stop("Length of target and feature must be the same.");
  }
  
  // Remove missing values and create index mapping
  std::vector<double> feature_clean;
  std::vector<int> target_clean;
  std::vector<int> original_indices;
  for (int i = 0; i < N; ++i) {
    if (!Rcpp::NumericVector::is_na(feature[i]) && !Rcpp::IntegerVector::is_na(target[i])) {
      feature_clean.push_back(feature[i]);
      target_clean.push_back(target[i]);
      original_indices.push_back(i);
    }
  }
  
  int N_clean = feature_clean.size();
  if (N_clean == 0) {
    Rcpp::stop("No valid data after removing missing values.");
  }
  
  // Create pre-bins
  std::vector<double> sorted_values = feature_clean;
  std::sort(sorted_values.begin(), sorted_values.end());
  auto last = std::unique(sorted_values.begin(), sorted_values.end());
  sorted_values.erase(last, sorted_values.end());
  
  if (sorted_values.size() == 1) {
    Rcpp::stop("All feature values are the same. Cannot create bins.");
  }
  
  int n_prebins = std::min((int)sorted_values.size() - 1, max_n_prebins);
  std::vector<double> candidate_cutpoints;
  
  if (n_prebins > 0) {
    int step = std::max(1, (int)sorted_values.size() / (n_prebins + 1));
    for (int i = step; i < (int)sorted_values.size() - 1; i += step) {
      candidate_cutpoints.push_back(sorted_values[i]);
    }
  } else {
    candidate_cutpoints = std::vector<double>(sorted_values.begin() + 1, sorted_values.end() - 1);
  }
  
  int n_cutpoints = candidate_cutpoints.size();
  
  // Prepare data for binning
  std::vector<int> pos_counts(n_cutpoints + 1, 0);
  std::vector<int> neg_counts(n_cutpoints + 1, 0);
  
  for (int i = 0; i < N_clean; ++i) {
    double value = feature_clean[i];
    int tgt = target_clean[i];
    
    // Determine bin index
    int bin_idx = std::lower_bound(candidate_cutpoints.begin(), candidate_cutpoints.end(), value) - candidate_cutpoints.begin();
    
    if (tgt == 1) {
      pos_counts[bin_idx]++;
    } else {
      neg_counts[bin_idx]++;
    }
  }
  
  int total_pos = std::accumulate(pos_counts.begin(), pos_counts.end(), 0);
  int total_neg = std::accumulate(neg_counts.begin(), neg_counts.end(), 0);
  
  return Rcpp::List::create(
    Rcpp::Named("candidate_cutpoints") = candidate_cutpoints,
    Rcpp::Named("pos_counts") = pos_counts,
    Rcpp::Named("neg_counts") = neg_counts,
    Rcpp::Named("total_pos") = total_pos,
    Rcpp::Named("total_neg") = total_neg,
    Rcpp::Named("original_indices") = original_indices,
    Rcpp::Named("feature_clean") = feature_clean
  );
}
