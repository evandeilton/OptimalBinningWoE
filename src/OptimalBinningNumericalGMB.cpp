// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]

#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif


using namespace Rcpp;

class OptimalBinningNumericalGMB {
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

  void validate_inputs();
  void initialize_bins();
  void merge_bins();
  void calculate_woe_iv();
  void enforce_monotonicity();

public:
  OptimalBinningNumericalGMB(const std::vector<double>& feature,
                             const std::vector<int>& target,
                             int min_bins, int max_bins,
                             double bin_cutoff, int max_n_prebins);

  List fit();
};

void OptimalBinningNumericalGMB::validate_inputs() {
  if (feature.size() != target.size()) {
    stop("Feature and target vectors must be of the same length.");
  }
  if (min_bins < 2) {
    stop("min_bins must be at least 2.");
  }
  if (max_bins < min_bins) {
    stop("max_bins must be greater than or equal to min_bins.");
  }
  if (bin_cutoff < 0.0 || bin_cutoff > 0.5) {
    stop("bin_cutoff must be between 0 and 0.5.");
  }
  if (max_n_prebins < min_bins) {
    stop("max_n_prebins must be greater than or equal to min_bins.");
  }
  for (int t : target) {
    if (t != 0 && t != 1) {
      stop("Target must be binary (0 or 1).");
    }
  }
}

void OptimalBinningNumericalGMB::initialize_bins() {
  // Create pre-bins based on quantiles
  std::vector<double> sorted_feature = feature;
  std::sort(sorted_feature.begin(), sorted_feature.end());

  std::vector<double> cut_points;
  int n = sorted_feature.size();
  int bin_size = n / max_n_prebins;

  for (int i = 1; i < max_n_prebins; ++i) {
    cut_points.push_back(sorted_feature[i * bin_size]);
  }

  // Remove duplicates
  cut_points.erase(std::unique(cut_points.begin(), cut_points.end()), cut_points.end());

  // Initialize bins
  bins.clear();
  double lower = -std::numeric_limits<double>::infinity();
  for (double cp : cut_points) {
    Bin bin = {lower, cp, 0, 0, 0, 0.0, 0.0};
    bins.push_back(bin);
    lower = cp;
  }
  // Last bin
  bins.push_back({lower, std::numeric_limits<double>::infinity(), 0, 0, 0, 0.0, 0.0});

  // Assign data to bins
  int num_bins = bins.size();
  int data_size = feature.size();

#pragma omp parallel for schedule(static) shared(bins)
  for (int i = 0; i < data_size; ++i) {
    double val = feature[i];
    int targ = target[i];
    for (int j = 0; j < num_bins; ++j) {
      if (val > bins[j].lower && val <= bins[j].upper) {
#pragma omp atomic
        bins[j].count += 1;
        if (targ == 1) {
#pragma omp atomic
          bins[j].count_pos += 1;
        } else {
#pragma omp atomic
          bins[j].count_neg += 1;
        }
        break;
      }
    }
  }

  // Remove empty bins
  bins.erase(std::remove_if(bins.begin(), bins.end(),
                            [](const Bin& b) { return b.count == 0; }), bins.end());
}

void OptimalBinningNumericalGMB::merge_bins() {
  int total_count = feature.size();
  double cutoff = bin_cutoff * total_count;

  // Merge bins with counts less than cutoff
  for (size_t i = 0; i < bins.size(); ) {
    if (bins[i].count < cutoff && bins.size() > min_bins) {
      if (i == 0) {
        // Merge with next bin
        bins[i+1].lower = bins[i].lower;
        bins[i+1].count += bins[i].count;
        bins[i+1].count_pos += bins[i].count_pos;
        bins[i+1].count_neg += bins[i].count_neg;
        bins.erase(bins.begin() + i);
      } else {
        // Merge with previous bin
        bins[i-1].upper = bins[i].upper;
        bins[i-1].count += bins[i].count;
        bins[i-1].count_pos += bins[i].count_pos;
        bins[i-1].count_neg += bins[i].count_neg;
        bins.erase(bins.begin() + i);
        --i;
      }
    } else {
      ++i;
    }
  }
}

void OptimalBinningNumericalGMB::calculate_woe_iv() {
  int total_pos = std::accumulate(target.begin(), target.end(), 0);
  int total_neg = target.size() - total_pos;

  for (auto& bin : bins) {
    double dist_pos = (double)bin.count_pos / total_pos;
    double dist_neg = (double)bin.count_neg / total_neg;
    if (dist_pos == 0) dist_pos = 1e-6;
    if (dist_neg == 0) dist_neg = 1e-6;
    bin.woe = std::log(dist_pos / dist_neg);
    bin.iv = (dist_pos - dist_neg) * bin.woe;
  }
}

void OptimalBinningNumericalGMB::enforce_monotonicity() {
  // Ensure WoE values are monotonic
  bool increasing = true;
  bool decreasing = true;

  for (size_t i = 1; i < bins.size(); ++i) {
    if (bins[i].woe < bins[i-1].woe) {
      increasing = false;
    }
    if (bins[i].woe > bins[i-1].woe) {
      decreasing = false;
    }
  }

  if (!increasing && !decreasing) {
    // Merge bins to enforce monotonicity
    // Simple heuristic: merge adjacent bins with minimal WoE difference
    while (bins.size() > min_bins) {
      double min_diff = std::numeric_limits<double>::infinity();
      size_t merge_idx = 0;

      for (size_t i = 0; i < bins.size() - 1; ++i) {
        double diff = std::abs(bins[i+1].woe - bins[i].woe);
        if (diff < min_diff) {
          min_diff = diff;
          merge_idx = i;
        }
      }

      // Merge bins at merge_idx and merge_idx + 1
      bins[merge_idx].upper = bins[merge_idx+1].upper;
      bins[merge_idx].count += bins[merge_idx+1].count;
      bins[merge_idx].count_pos += bins[merge_idx+1].count_pos;
      bins[merge_idx].count_neg += bins[merge_idx+1].count_neg;
      bins.erase(bins.begin() + merge_idx + 1);

      calculate_woe_iv();
      // Re-check monotonicity
      increasing = true;
      decreasing = true;
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i-1].woe) {
          increasing = false;
        }
        if (bins[i].woe > bins[i-1].woe) {
          decreasing = false;
        }
      }
      if (increasing || decreasing || bins.size() <= min_bins) {
        break;
      }
    }
  }
}

OptimalBinningNumericalGMB::OptimalBinningNumericalGMB(
  const std::vector<double>& feature,
  const std::vector<int>& target,
  int min_bins, int max_bins,
  double bin_cutoff, int max_n_prebins)
  : feature(feature), target(target),
    min_bins(min_bins), max_bins(max_bins),
    bin_cutoff(bin_cutoff), max_n_prebins(max_n_prebins) {
  validate_inputs();
}

List OptimalBinningNumericalGMB::fit() {
  initialize_bins();
  merge_bins();
  calculate_woe_iv();
  enforce_monotonicity();

  // Prepare outputs
  std::vector<double> woefeature(feature.size());
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < feature.size(); ++i) {
    double val = feature[i];
    for (const auto& bin : bins) {
      if (val > bin.lower && val <= bin.upper) {
        woefeature[i] = bin.woe;
        break;
      }
    }
  }

  // Prepare bin dataframe
  std::vector<std::string> bin_names;
  std::vector<double> woe_values, iv_values;
  std::vector<int> counts, count_poses, count_negs;

  for (const auto& bin : bins) {
    std::string bin_name = "(" + std::to_string(bin.lower) + ";" + std::to_string(bin.upper) + "]";
    bin_names.push_back(bin_name);
    woe_values.push_back(bin.woe);
    iv_values.push_back(bin.iv);
    counts.push_back(bin.count);
    count_poses.push_back(bin.count_pos);
    count_negs.push_back(bin.count_neg);
  }

  DataFrame woebin = DataFrame::create(
    Named("bin") = bin_names,
    Named("woe") = woe_values,
    Named("iv") = iv_values,
    Named("count") = counts,
    Named("count_pos") = count_poses,
    Named("count_neg") = count_negs
  );

  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}


//' @title Optimal Binning for Numerical Variables using Greedy Monotonic Binning
//' 
//' @description
//' This function implements an optimal binning algorithm for numerical variables using a greedy monotonic binning approach. It aims to find the best binning strategy that maximizes the predictive power while ensuring monotonicity in the Weight of Evidence (WoE) values.
//' 
//' @param target An integer vector of binary target values (0 or 1).
//' @param feature A numeric vector of feature values to be binned.
//' @param min_bins Minimum number of bins (default: 3).
//' @param max_bins Maximum number of bins (default: 5).
//' @param bin_cutoff Minimum fraction of total observations in each bin (default: 0.05).
//' @param max_n_prebins Maximum number of pre-bins (default: 20).
//' 
//' @return A list containing:
//' \item{woefeature}{A numeric vector of Weight of Evidence (WoE) values for each observation}
//' \item{woebin}{A data frame with binning information, including bin ranges, WoE, IV, and counts}
//' 
//' @details
//' The optimal binning algorithm using greedy monotonic binning consists of several steps:
//' 
//' 1. Initial binning: The feature is initially divided into a maximum number of bins specified by \code{max_n_prebins}.
//' 2. Merging low-frequency bins: Bins with a fraction of observations less than \code{bin_cutoff} are merged with adjacent bins.
//' 3. Calculating WoE and IV: The Weight of Evidence (WoE) and Information Value (IV) are calculated for each bin.
//' 4. Enforcing monotonicity: The algorithm ensures that the WoE values are either monotonically increasing or decreasing across the bins.
//' 
//' The Weight of Evidence (WoE) for each bin is calculated as:
//' 
//' \deqn{WoE = \ln\left(\frac{P(X|Y=1)}{P(X|Y=0)}\right)}
//' 
//' where \eqn{P(X|Y=1)} is the probability of the feature being in a particular bin given a positive target, and \eqn{P(X|Y=0)} is the probability given a negative target.
//' 
//' The Information Value (IV) for each bin is calculated as:
//' 
//' \deqn{IV = (P(X|Y=1) - P(X|Y=0)) * WoE}
//' 
//' The algorithm uses a greedy approach to enforce monotonicity:
//' 
//' 1. Check if the initial WoE values are monotonic (increasing or decreasing).
//' 2. If not monotonic, iteratively merge adjacent bins with the smallest WoE difference until monotonicity is achieved or the minimum number of bins is reached.
//' 3. After each merge, recalculate WoE and IV values and check for monotonicity.
//' 
//' This approach ensures that the final binning solution has monotonic WoE values, which is often desirable for interpretability and stability of the binning.
//' 
//' @examples
//' \dontrun{
//' set.seed(123)
//' target <- sample(0:1, 1000, replace = TRUE)
//' feature <- rnorm(1000)
//' result <- optimal_binning_numerical_gmb(target, feature)
//' print(result$woebin)
//' }
//' 
//' @references
//' \itemize{
//'   \item Belotti, P., & Carrasco, M. (2017). Optimal binning: mathematical programming formulation and solution approach. arXiv preprint arXiv:1705.03287.
//'   \item Mironchyk, P., & Tchistiakov, V. (2017). Monotone optimal binning algorithm for credit risk modeling. arXiv preprint arXiv:1711.06692.
//' }
//' 
//' @author Lopes, J. E.
//' @export
// [[Rcpp::export]]
List optimal_binning_numerical_gmb(IntegerVector target, NumericVector feature,
                                   int min_bins = 3, int max_bins = 5,
                                   double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalGMB obngmb(
      as<std::vector<double>>(feature),
      as<std::vector<int>>(target),
      min_bins, max_bins, bin_cutoff, max_n_prebins
  );

  return obngmb.fit();
}
