#include <Rcpp.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <numeric>

// [[Rcpp::plugins(openmp)]]

class OptimalBinningNumericalIR {
private:
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  
  std::vector<double> feature;
  std::vector<int> target;
  
  std::vector<double> bin_edges;
  std::vector<double> woe_values;
  double total_iv;
  
  struct BinInfo {
    double lower;
    double upper;
    int count;
    int count_pos;
    int count_neg;
    double woe;
    double iv;
  };
  
  std::vector<BinInfo> bin_info;
  
public:
  OptimalBinningNumericalIR(int min_bins_ = 2, int max_bins_ = 5,
                            double bin_cutoff_ = 0.05, int max_n_prebins_ = 20)
    : min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_) {}
  
  void fit(const std::vector<double>& feature_, const std::vector<int>& target_) {
    feature = feature_;
    target = target_;
    
    validateInputs();
    createInitialBins();
    mergeLowFrequencyBins();
    ensureMinMaxBins();
    applyIsotonicRegression();
    calculateWOEandIV();
  }
  
  Rcpp::List getResults() {
    std::vector<double> woefeature = applyWOEToFeature();
    Rcpp::DataFrame woebin = createWOEBinDataFrame();
    
    return Rcpp::List::create(
      Rcpp::Named("woefeature") = woefeature,
      Rcpp::Named("woebin") = woebin
    );
  }
  
private:
  void validateInputs() {
    if (feature.size() != target.size()) {
      Rcpp::stop("Feature and target must have the same length");
    }
    if (min_bins < 2) {
      Rcpp::stop("min_bins must be at least 2");
    }
    if (max_bins < min_bins) {
      Rcpp::stop("max_bins must be greater than or equal to min_bins");
    }
    if (*std::min_element(target.begin(), target.end()) < 0 || 
        *std::max_element(target.begin(), target.end()) > 1) {
        Rcpp::stop("Target must be binary (0 or 1)");
    }
  }
  
  void createInitialBins() {
    std::vector<double> sorted_feature = feature;
    std::sort(sorted_feature.begin(), sorted_feature.end());
    int n = sorted_feature.size();
    int n_prebins = std::min(max_n_prebins, n);
    
    bin_edges.resize(n_prebins + 1);
    bin_edges[0] = sorted_feature[0] - 1e-6;
    bin_edges[n_prebins] = sorted_feature[n - 1] + 1e-6;
    
    for (int i = 1; i < n_prebins; ++i) {
      int idx = (i * n) / n_prebins;
      bin_edges[i] = (sorted_feature[idx - 1] + sorted_feature[idx]) / 2;
    }
  }
  
  void mergeLowFrequencyBins() {
    std::vector<BinInfo> temp_bins;
    int total_count = feature.size();
    
    for (size_t i = 0; i < bin_edges.size() - 1; ++i) {
      BinInfo bin;
      bin.lower = bin_edges[i];
      bin.upper = bin_edges[i + 1];
      
      bin.count = 0;
      bin.count_pos = 0;
      bin.count_neg = 0;
      
      for (size_t j = 0; j < feature.size(); ++j) {
        if (feature[j] > bin.lower && feature[j] <= bin.upper) {
          bin.count++;
          bin.count_pos += target[j];
        }
      }
      bin.count_neg = bin.count - bin.count_pos;
      
      if (static_cast<double>(bin.count) / total_count >= bin_cutoff) {
        temp_bins.push_back(bin);
      } else if (!temp_bins.empty()) {
        temp_bins.back().upper = bin.upper;
        temp_bins.back().count += bin.count;
        temp_bins.back().count_pos += bin.count_pos;
        temp_bins.back().count_neg += bin.count_neg;
      } else {
        // If the first bin is below cutoff, we need to keep it to ensure we have at least one bin
        temp_bins.push_back(bin);
      }
    }
    
    bin_info = temp_bins;
  }
  
  void ensureMinMaxBins() {
    // If we have fewer than min_bins, split the largest bins
    while (bin_info.size() < min_bins) {
      auto it = std::max_element(bin_info.begin(), bin_info.end(),
                                 [](const BinInfo& a, const BinInfo& b) { return a.count < b.count; });
      
      size_t idx = std::distance(bin_info.begin(), it);
      double mid = (it->lower + it->upper) / 2;
      
      BinInfo new_bin = *it;
      it->upper = mid;
      new_bin.lower = mid;
      
      // Recalculate counts for split bins
      it->count = 0;
      it->count_pos = 0;
      new_bin.count = 0;
      new_bin.count_pos = 0;
      
      for (size_t i = 0; i < feature.size(); ++i) {
        if (feature[i] > it->lower && feature[i] <= it->upper) {
          it->count++;
          it->count_pos += target[i];
        } else if (feature[i] > new_bin.lower && feature[i] <= new_bin.upper) {
          new_bin.count++;
          new_bin.count_pos += target[i];
        }
      }
      
      it->count_neg = it->count - it->count_pos;
      new_bin.count_neg = new_bin.count - new_bin.count_pos;
      
      bin_info.insert(it + 1, new_bin);
    }
    
    // If we have more than max_bins, merge the most similar adjacent bins
    while (bin_info.size() > max_bins) {
      auto it = std::min_element(bin_info.begin(), bin_info.end() - 1,
                                 [](const BinInfo& a, const BinInfo& b) {
                                   double rate_a = static_cast<double>(a.count_pos) / a.count;
                                   double rate_b = static_cast<double>(b.count_pos) / b.count;
                                   double next_rate_a = static_cast<double>(a.count_pos + (*(std::next(&a))).count_pos) /
                                     (a.count + (*(std::next(&a))).count);
                                   double next_rate_b = static_cast<double>(b.count_pos + (*(std::next(&b))).count_pos) /
                                     (b.count + (*(std::next(&b))).count);
                                   return std::abs(rate_a - next_rate_a) < std::abs(rate_b - next_rate_b);
                                 });
      
      it->upper = (*(std::next(it))).upper;
      it->count += (*(std::next(it))).count;
      it->count_pos += (*(std::next(it))).count_pos;
      it->count_neg += (*(std::next(it))).count_neg;
      bin_info.erase(std::next(it));
    }
  }
  
  void applyIsotonicRegression() {
    int n = bin_info.size();
    std::vector<double> y(n), w(n);
    
    for (int i = 0; i < n; ++i) {
      double pos_rate = static_cast<double>(bin_info[i].count_pos) / bin_info[i].count;
      y[i] = pos_rate;
      w[i] = bin_info[i].count;
    }
    
    std::vector<double> isotonic_y = isotonic_regression(y, w);
    
    for (int i = 0; i < n; ++i) {
      bin_info[i].woe = std::log(isotonic_y[i] / (1 - isotonic_y[i]));
    }
  }
  
  std::vector<double> isotonic_regression(const std::vector<double>& y, const std::vector<double>& w) {
    int n = y.size();
    std::vector<double> result = y;
    
    for (int i = 1; i < n; ++i) {
      if (result[i] < result[i - 1]) {
        double numerator = 0, denominator = 0;
        int j = i;
        
        while (j > 0 && result[j] < result[j - 1]) {
          numerator += w[j] * y[j] + w[j - 1] * y[j - 1];
          denominator += w[j] + w[j - 1];
          double pooled = numerator / denominator;
          result[j] = result[j - 1] = pooled;
          j--;
        }
      }
    }
    
    return result;
  }
  
  void calculateWOEandIV() {
    double total_pos = 0, total_neg = 0;
    for (const auto& bin : bin_info) {
      total_pos += bin.count_pos;
      total_neg += bin.count_neg;
    }
    
    total_iv = 0;
    for (auto& bin : bin_info) {
      double pos_rate = bin.count_pos / total_pos;
      double neg_rate = bin.count_neg / total_neg;
      bin.woe = std::log(pos_rate / neg_rate);
      bin.iv = (pos_rate - neg_rate) * bin.woe;
      total_iv += bin.iv;
    }
  }
  
  std::vector<double> applyWOEToFeature() {
    int n = feature.size();
    std::vector<double> woefeature(n);
    
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      double x = feature[i];
      for (const auto& bin : bin_info) {
        if (x > bin.lower && x <= bin.upper) {
          woefeature[i] = bin.woe;
          break;
        }
      }
    }
    
    return woefeature;
  }
  
  Rcpp::DataFrame createWOEBinDataFrame() {
    int n_bins = bin_info.size();
    Rcpp::CharacterVector bin(n_bins);
    Rcpp::NumericVector woe(n_bins), iv(n_bins);
    Rcpp::IntegerVector count(n_bins), count_pos(n_bins), count_neg(n_bins);
    
    for (int i = 0; i < n_bins; ++i) {
      const auto& b = bin_info[i];
      bin[i] = "(" + std::to_string(b.lower) + ";" + std::to_string(b.upper) + "]";
      woe[i] = b.woe;
      iv[i] = b.iv;
      count[i] = b.count;
      count_pos[i] = b.count_pos;
      count_neg[i] = b.count_neg;
    }
    
    return Rcpp::DataFrame::create(
      Rcpp::Named("bin") = bin,
      Rcpp::Named("woe") = woe,
      Rcpp::Named("iv") = iv,
      Rcpp::Named("count") = count,
      Rcpp::Named("count_pos") = count_pos,
      Rcpp::Named("count_neg") = count_neg
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_ir(std::vector<int> target, std::vector<double> feature,
                                        int min_bins = 3, int max_bins = 5,
                                        double bin_cutoff = 0.05, int max_n_prebins = 20) {
  OptimalBinningNumericalIR binner(min_bins, max_bins, bin_cutoff, max_n_prebins);
  binner.fit(feature, target);
  return binner.getResults();
}
