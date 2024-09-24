// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <Rcpp.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>
#include <set>

using namespace Rcpp;

class OptimalBinningNumericalSAB {
public:
  OptimalBinningNumericalSAB(const NumericVector& feature,
                             const IntegerVector& target,
                             int min_bins,
                             int max_bins,
                             double bin_cutoff,
                             int max_n_prebins)
    : feature(feature),
      target(target),
      min_bins(std::max(min_bins, 2)),
      max_bins(std::max(max_bins, this->min_bins)),
      bin_cutoff(bin_cutoff),
      max_n_prebins(max_n_prebins)
  {
    n = feature.size();
    ValidateInput();
    Initialize();
  }

  List Fit() {
    PreBinning();
    SimulatedAnnealingOptimization();
    MergeLowFrequencyBins();
    CalculateWoE();
    return PrepareOutput();
  }

private:
  NumericVector feature;
  IntegerVector target;
  int n;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;

  std::vector<double> cut_points;
  std::vector<double> final_cut_points;
  NumericVector woefeature;
  DataFrame woebin;

  void ValidateInput() {
    if (feature.size() != target.size()) {
      stop("Feature and target must have the same length.");
    }

    for (int i = 0; i < target.size(); ++i) {
      if (target[i] != 0 && target[i] != 1) {
        stop("Target must be binary (0 and 1).");
      }
    }

    if (min_bins < 2) {
      stop("min_bins must be at least 2.");
    }

    if (max_bins < min_bins) {
      stop("max_bins must be greater than or equal to min_bins.");
    }

    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      stop("bin_cutoff must be between 0 and 1.");
    }

    if (max_n_prebins < 2) {
      stop("max_n_prebins must be at least 2.");
    }
  }

  void Initialize() {
    woefeature = NumericVector(n);
  }

  void PreBinning() {
    NumericVector sorted_feature = clone(feature);
    std::sort(sorted_feature.begin(), sorted_feature.end());

    int step = std::max(1, n / max_n_prebins);
    std::set<double> cut_points_set;

    for (int i = 1; i < max_n_prebins && i * step < n; ++i) {
      cut_points_set.insert(sorted_feature[i * step]);
    }

    cut_points.assign(cut_points_set.begin(), cut_points_set.end());
  }

  double CalculateIV(const std::vector<double>& cuts) {
    int bin_count = static_cast<int>(cuts.size()) + 1;
    std::vector<double> count_pos(bin_count, 0.0);
    std::vector<double> count_neg(bin_count, 0.0);
    double total_pos = 0.0;
    double total_neg = 0.0;

#pragma omp parallel for reduction(+:total_pos,total_neg)
    for (int i = 0; i < n; ++i) {
      int bin_idx = GetBinIndex(feature[i], cuts);
      if (target[i] == 1) {
#pragma omp atomic
        count_pos[bin_idx] += 1.0;
        total_pos += 1.0;
      } else {
#pragma omp atomic
        count_neg[bin_idx] += 1.0;
        total_neg += 1.0;
      }
    }

    double iv = 0.0;
    for (int i = 0; i < bin_count; ++i) {
      if (count_pos[i] > 0 && count_neg[i] > 0) {
        double pos_rate = count_pos[i] / total_pos;
        double neg_rate = count_neg[i] / total_neg;
        double woe = std::log(pos_rate / neg_rate);
        iv += (pos_rate - neg_rate) * woe;
      }
    }
    return iv;
  }

  int GetBinIndex(double value, const std::vector<double>& cuts) {
    return static_cast<int>(std::lower_bound(cuts.begin(), cuts.end(), value) - cuts.begin());
  }

  void SimulatedAnnealingOptimization() {
    double T = 1.0;
    double T_min = 0.0001;
    double alpha = 0.9;
    int max_iter = 1000;

    final_cut_points = cut_points;
    double best_iv = CalculateIV(final_cut_points);

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> uni_dist(0.0, 1.0);

    while (T > T_min) {
      for (int i = 0; i < max_iter; ++i) {
        std::vector<double> new_cuts = GenerateNeighbor(final_cut_points);
        if (static_cast<int>(new_cuts.size()) < min_bins - 1 || static_cast<int>(new_cuts.size()) > max_bins - 1) {
          continue;
        }
        double new_iv = CalculateIV(new_cuts);
        double delta = new_iv - best_iv;
        if (delta > 0 || std::exp(delta / T) > uni_dist(rng)) {
          final_cut_points = new_cuts;
          best_iv = new_iv;
        }
      }
      T *= alpha;
    }
  }

  std::vector<double> GenerateNeighbor(const std::vector<double>& cuts) {
    std::vector<double> neighbor = cuts;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> idx_dist(0, static_cast<int>(neighbor.size()) - 1);
    std::uniform_real_distribution<double> value_dist(Rcpp::min(feature), Rcpp::max(feature));

    int action = rng() % 3; // 0: add cut, 1: remove cut, 2: modify cut
    if (action == 0 && static_cast<int>(neighbor.size()) < max_bins - 1) {
      double new_cut = value_dist(rng);
      neighbor.push_back(new_cut);
      std::sort(neighbor.begin(), neighbor.end());
    } else if (action == 1 && static_cast<int>(neighbor.size()) > min_bins - 1) {
      int idx = idx_dist(rng);
      neighbor.erase(neighbor.begin() + idx);
    } else if (action == 2) {
      int idx = idx_dist(rng);
      neighbor[idx] = value_dist(rng);
      std::sort(neighbor.begin(), neighbor.end());
    }
    return neighbor;
  }

  void MergeLowFrequencyBins() {
    std::vector<int> bin_counts(final_cut_points.size() + 1, 0);
    for (int i = 0; i < n; ++i) {
      int bin_idx = GetBinIndex(feature[i], final_cut_points);
      bin_counts[bin_idx]++;
    }

    double total_count = std::accumulate(bin_counts.begin(), bin_counts.end(), 0.0);
    std::vector<double> new_cut_points;

    for (size_t i = 0; i < final_cut_points.size(); ++i) {
      if (bin_counts[i] / total_count >= bin_cutoff) {
        new_cut_points.push_back(final_cut_points[i]);
      }
    }

    // Ensure we respect min_bins and max_bins
    while (static_cast<int>(new_cut_points.size()) + 1 < min_bins && !final_cut_points.empty()) {
      new_cut_points.push_back(final_cut_points[new_cut_points.size()]);
    }

    while (static_cast<int>(new_cut_points.size()) + 1 > max_bins) {
      new_cut_points.pop_back();
    }

    final_cut_points = new_cut_points;
  }

  void CalculateWoE() {
    int bin_count = static_cast<int>(final_cut_points.size()) + 1;
    std::vector<double> count_pos(bin_count, 0.0);
    std::vector<double> count_neg(bin_count, 0.0);
    double total_pos = 0.0;
    double total_neg = 0.0;

    std::vector<int> bin_indices(n);

#pragma omp parallel for reduction(+:total_pos,total_neg)
    for (int i = 0; i < n; ++i) {
      int bin_idx = GetBinIndex(feature[i], final_cut_points);
      bin_indices[i] = bin_idx;
      if (target[i] == 1) {
#pragma omp atomic
        count_pos[bin_idx] += 1.0;
        total_pos += 1.0;
      } else {
#pragma omp atomic
        count_neg[bin_idx] += 1.0;
        total_neg += 1.0;
      }
    }

    std::vector<double> woe(bin_count, 0.0);
    std::vector<double> iv(bin_count, 0.0);
    std::vector<int> count(bin_count, 0);

    for (int i = 0; i < bin_count; ++i) {
      count[i] = count_pos[i] + count_neg[i];
      if (count_pos[i] > 0 && count_neg[i] > 0) {
        double pos_rate = count_pos[i] / total_pos;
        double neg_rate = count_neg[i] / total_neg;
        woe[i] = std::log(pos_rate / neg_rate);
        iv[i] = (pos_rate - neg_rate) * woe[i];
      }
    }

#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      woefeature[i] = woe[bin_indices[i]];
    }

    CharacterVector bins(bin_count);
    NumericVector woe_vec(bin_count);
    NumericVector iv_vec(bin_count);
    IntegerVector count_vec(bin_count);
    IntegerVector count_pos_vec(bin_count);
    IntegerVector count_neg_vec(bin_count);

    for (int i = 0; i < bin_count; ++i) {
      bins[i] = GetBinLabel(i, final_cut_points);
      woe_vec[i] = woe[i];
      iv_vec[i] = iv[i];
      count_vec[i] = count[i];
      count_pos_vec[i] = count_pos[i];
      count_neg_vec[i] = count_neg[i];
    }

    woebin = DataFrame::create(
      Named("bin") = bins,
      Named("woe") = woe_vec,
      Named("iv") = iv_vec,
      Named("count") = count_vec,
      Named("count_pos") = count_pos_vec,
      Named("count_neg") = count_neg_vec
    );
  }

  std::string GetBinLabel(int bin_idx, const std::vector<double>& cuts) {
    std::ostringstream oss;
    oss.precision(4);
    oss << std::fixed;
    if (bin_idx == 0) {
      oss << "(-Inf, " << cuts[0] << "]";
    } else if (bin_idx == static_cast<int>(cuts.size())) {
      oss << "(" << cuts.back() << ", +Inf]";
    } else {
      oss << "(" << cuts[bin_idx - 1] << ", " << cuts[bin_idx] << "]";
    }
    return oss.str();
  }

  List PrepareOutput() {
    return List::create(
      Named("woefeature") = woefeature,
      Named("woebin") = woebin
    );
  }
};

// [[Rcpp::export]]
List optimal_binning_numerical_sab(IntegerVector target,
                                   NumericVector feature,
                                   int min_bins = 3,
                                   int max_bins = 5,
                                   double bin_cutoff = 0.05,
                                   int max_n_prebins = 20) {
  OptimalBinningNumericalSAB binning(feature, target, min_bins, max_bins, bin_cutoff, max_n_prebins);
  return binning.Fit();
}
