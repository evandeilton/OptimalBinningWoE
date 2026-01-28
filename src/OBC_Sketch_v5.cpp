// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace Rcpp;

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Global constants
static constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
static constexpr double LAPLACE_ALPHA = 0.5;
static constexpr const char *MISSING_VALUE = "N/A";

// Utility functions
namespace utils {
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

inline std::string join(const std::vector<std::string> &v,
                        const std::string &delimiter) {
  if (v.empty())
    return "";
  if (v.size() == 1)
    return v[0];

  std::unordered_set<std::string> unique_values;
  std::vector<std::string> unique_vector;
  unique_vector.reserve(v.size());

  for (const auto &s : v) {
    if (unique_values.insert(s).second) {
      unique_vector.push_back(s);
    }
  }

  size_t total_length = 0;
  for (const auto &s : unique_vector) {
    total_length += s.length();
  }
  total_length += delimiter.length() * (unique_vector.size() - 1);

  std::string result;
  result.reserve(total_length);

  result = unique_vector[0];
  for (size_t i = 1; i < unique_vector.size(); ++i) {
    result += delimiter;
    result += unique_vector[i];
  }
  return result;
}

inline size_t string_hash(const std::string &str, size_t seed) {
  constexpr size_t FNV_PRIME = 1099511628211ULL;
  constexpr size_t FNV_OFFSET_BASIS = 14695981039346656037ULL;

  size_t hash = FNV_OFFSET_BASIS ^ seed;
  for (char c : str) {
    hash ^= static_cast<size_t>(c);
    hash *= FNV_PRIME;
  }
  return hash;
}

inline std::pair<double, double>
smoothed_proportions(int positive_count, int negative_count, int total_positive,
                     int total_negative, double alpha = LAPLACE_ALPHA) {
  double smoothed_pos_rate =
      (positive_count + alpha) / (total_positive + alpha * 2);
  double smoothed_neg_rate =
      (negative_count + alpha) / (total_negative + alpha * 2);

  return {smoothed_pos_rate, smoothed_neg_rate};
}

inline double calculate_woe(int positive_count, int negative_count,
                            int total_positive, int total_negative,
                            double alpha = LAPLACE_ALPHA) {
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
      positive_count, negative_count, total_positive, total_negative, alpha);

  return safe_log(smoothed_pos_rate / smoothed_neg_rate);
}

inline double calculate_iv(int positive_count, int negative_count,
                           int total_positive, int total_negative,
                           double alpha = LAPLACE_ALPHA) {
  auto [smoothed_pos_rate, smoothed_neg_rate] = smoothed_proportions(
      positive_count, negative_count, total_positive, total_negative, alpha);

  double woe = safe_log(smoothed_pos_rate / smoothed_neg_rate);
  return (smoothed_pos_rate - smoothed_neg_rate) * woe;
}

inline double bin_divergence(int bin1_pos, int bin1_neg, int bin2_pos,
                             int bin2_neg, int total_pos, int total_neg) {
  auto [p1, n1] =
      smoothed_proportions(bin1_pos, bin1_neg, total_pos, total_neg);
  auto [p2, n2] =
      smoothed_proportions(bin2_pos, bin2_neg, total_pos, total_neg);

  double p_avg = (p1 + p2) / 2;
  double n_avg = (n1 + n2) / 2;

  double div_p1 = p1 > EPSILON ? p1 * safe_log(p1 / p_avg) : 0;
  double div_n1 = n1 > EPSILON ? n1 * safe_log(n1 / n_avg) : 0;
  double div_p2 = p2 > EPSILON ? p2 * safe_log(p2 / p_avg) : 0;
  double div_n2 = n2 > EPSILON ? n2 * safe_log(n2 / n_avg) : 0;

  return (div_p1 + div_n1 + div_p2 + div_n2) / 2;
}
} // namespace utils

// Count-Min Sketch structure for frequency estimation
class CountMinSketch {
private:
  std::vector<std::vector<int>> table;
  std::vector<size_t> seeds;
  size_t width;
  size_t depth;

public:
  CountMinSketch(size_t width_param = 2000, size_t depth_param = 5)
      : width(width_param), depth(depth_param) {
    table.resize(depth);
    for (auto &row : table) {
      row.resize(width, 0);
    }

    std::mt19937 gen(42);
    std::uniform_int_distribution<size_t> dist(
        1, std::numeric_limits<size_t>::max());

    seeds.resize(depth);
    for (size_t i = 0; i < depth; ++i) {
      seeds[i] = dist(gen);
    }
  }

  void update(const std::string &item, int count = 1) {
    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      table[i][hash] += count;
    }
  }

  int estimate(const std::string &item) const {
    int min_count = std::numeric_limits<int>::max();

    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      min_count = std::min(min_count, table[i][hash]);
    }

    return min_count;
  }

  std::pair<int, int> estimate_with_bounds(const std::string &item) const {
    std::vector<int> counts(depth);

    for (size_t i = 0; i < depth; ++i) {
      size_t hash = utils::string_hash(item, seeds[i]) % width;
      counts[i] = table[i][hash];
    }

    std::sort(counts.begin(), counts.end());
    return {counts[0], counts[depth / 2]};
  }

  std::vector<std::string>
  heavy_hitters(const std::vector<std::string> &candidates,
                double threshold_ratio) const {
    int64_t total_count = 0;
    for (size_t i = 0; i < depth; ++i) {
      int64_t row_sum = std::accumulate(table[i].begin(), table[i].end(), 0LL);
      total_count += row_sum;
    }
    total_count /= static_cast<int64_t>(depth);

    int threshold = static_cast<int>(total_count * threshold_ratio);

    std::vector<std::string> result;
    result.reserve(candidates.size() / 4);

    for (const auto &candidate : candidates) {
      if (estimate(candidate) >= threshold) {
        result.push_back(candidate);
      }
    }

    return result;
  }

  int64_t estimate_total_elements() const {
    int64_t total_count = 0;
    for (size_t i = 0; i < depth; ++i) {
      int64_t row_sum = std::accumulate(table[i].begin(), table[i].end(), 0LL);
      total_count += row_sum;
    }
    return total_count / static_cast<int64_t>(depth);
  }
};

// Main class for Categorical Sketch Binning
// NOTE: MergeCache removed to fix UBSAN memory corruption issues
class OBC_Sketch {
private:
  std::vector<std::string> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  size_t sketch_width;
  size_t sketch_depth;
  bool use_divergence;

  int total_good;
  int total_bad;

  std::vector<CategoricalBin> bins;
  std::unique_ptr<CountMinSketch> sketch;
  std::unique_ptr<CountMinSketch> sketch_pos;
  std::unique_ptr<CountMinSketch> sketch_neg;

  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument(
          "Feature and target must have the same size.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature and target cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (max_n_prebins < max_bins) {
      throw std::invalid_argument("max_n_prebins must be >= max_bins.");
    }
    if (sketch_width < 100) {
      throw std::invalid_argument(
          "sketch_width must be >= 100 for reasonable accuracy.");
    }
    if (sketch_depth < 3) {
      throw std::invalid_argument(
          "sketch_depth must be >= 3 for reasonable accuracy.");
    }

    bool has_zero = false;
    bool has_one = false;

    for (int val : target) {
      if (val == 0)
        has_zero = true;
      else if (val == 1)
        has_one = true;
      else
        throw std::invalid_argument("Target must contain only 0 and 1.");

      if (has_zero && has_one)
        break;
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1.");
    }
  }

  void build_sketches() {
    sketch = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_pos = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_neg = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);

    total_good = 0;
    total_bad = 0;

    std::unordered_set<std::string> unique_categories;

    for (size_t i = 0; i < feature.size(); ++i) {
      const auto &cat = feature[i];
      int is_positive = target[i];

      unique_categories.insert(cat);
      sketch->update(cat, 1);

      if (is_positive) {
        sketch_pos->update(cat, 1);
        total_bad++;
      } else {
        sketch_neg->update(cat, 1);
        total_good++;
      }
    }

    int ncat = static_cast<int>(unique_categories.size());
    if (max_bins > ncat) {
      max_bins = ncat;
    }
    min_bins = std::min(min_bins, max_bins);
  }

  void prebinning() {
    std::unordered_set<std::string> unique_categories_set;
    for (const auto &cat : feature) {
      unique_categories_set.insert(cat);
    }

    std::vector<std::string> unique_categories(unique_categories_set.begin(),
                                               unique_categories_set.end());

    double heavy_hitter_threshold = bin_cutoff / 2.0;
    std::vector<std::string> heavy_hitters =
        sketch->heavy_hitters(unique_categories, heavy_hitter_threshold);

    bins.clear();
    bins.reserve(heavy_hitters.size());

    for (const auto &cat : heavy_hitters) {
      CategoricalBin bin;
      int pos_count = sketch_pos->estimate(cat);
      int neg_count = sketch_neg->estimate(cat);
      bin.categories.push_back(cat);
      bin.count_pos += pos_count;
      bin.count_neg += neg_count;
      bin.update_count();
      bins.push_back(bin);
    }

    CategoricalBin rare_bin;
    for (const auto &cat : unique_categories) {
      if (std::find(heavy_hitters.begin(), heavy_hitters.end(), cat) ==
          heavy_hitters.end()) {
        int pos_count = sketch_pos->estimate(cat);
        int neg_count = sketch_neg->estimate(cat);
        rare_bin.categories.push_back(cat);
        rare_bin.count_pos += pos_count;
        rare_bin.count_neg += neg_count;
        rare_bin.update_count();
      }
    }

    if (rare_bin.count > 0) {
      bins.push_back(rare_bin);
    }

    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.count > b.count;
              });

    while (static_cast<int>(bins.size()) > max_n_prebins &&
           static_cast<int>(bins.size()) > min_bins) {
      size_t min_idx1 = 0;
      size_t min_idx2 = 1;
      double min_divergence = std::numeric_limits<double>::max();

      for (size_t i = 0; i < bins.size(); ++i) {
        for (size_t j = i + 1; j < bins.size(); ++j) {
          int combined_count = bins[i].count + bins[j].count;
          double size_penalty = 1.0 + std::log(1.0 + combined_count);

          double div = bins[i].divergence_from(bins[j], total_good, total_bad) *
                       size_penalty;

          if (div < min_divergence) {
            min_divergence = div;
            min_idx1 = i;
            min_idx2 = j;
          }
        }
      }

      if (!try_merge_bins(min_idx1, min_idx2))
        break;
    }
  }

  void enforce_bin_cutoff() {
    int min_count = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    int min_count_pos = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(total_bad)));

    std::vector<size_t> low_freq_bins;

    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
        low_freq_bins.push_back(i);
      }
    }

    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }

      if (idx >= bins.size() || (bins[idx].count >= min_count &&
                                 bins[idx].count_pos >= min_count_pos)) {
        continue;
      }

      size_t merge_idx = idx;
      double min_divergence = std::numeric_limits<double>::max();

      for (size_t i = 0; i < bins.size(); ++i) {
        if (i == idx)
          continue;

        double div = bins[idx].divergence_from(bins[i], total_good, total_bad);
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx = i;
        }
      }

      if (idx == merge_idx) {
        if (idx > 0) {
          merge_idx = idx - 1;
        } else if (idx + 1 < bins.size()) {
          merge_idx = idx + 1;
        } else {
          continue;
        }
      }

      if (!try_merge_bins(std::min(idx, merge_idx), std::max(idx, merge_idx))) {
        continue;
      }

      for (auto &remaining_idx : low_freq_bins) {
        if (remaining_idx > merge_idx) {
          remaining_idx--;
        }
      }
    }
  }

  void calculate_initial_woe() {
    for (auto &bin : bins) {
      bin.calculate_metrics(total_good, total_bad);
    }
  }

  void enforce_monotonicity() {
    if (bins.empty()) {
      throw std::runtime_error("No bins available to enforce monotonicity.");
    }

    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.woe < b.woe;
              });

    bool increasing = true;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe - EPSILON) {
          increasing = false;
          break;
        }
      }
    }

    bool any_merge;
    do {
      any_merge = false;

      double max_violation = 0.0;
      size_t violation_idx = 0;

      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        if (static_cast<int>(bins.size()) <= min_bins) {
          break;
        }

        double violation_amount = 0.0;
        bool is_violation = false;

        if (increasing && bins[i].woe > bins[i + 1].woe + EPSILON) {
          violation_amount = bins[i].woe - bins[i + 1].woe;
          is_violation = true;
        } else if (!increasing && bins[i].woe < bins[i + 1].woe - EPSILON) {
          violation_amount = bins[i + 1].woe - bins[i].woe;
          is_violation = true;
        }

        if (is_violation && violation_amount > max_violation) {
          max_violation = violation_amount;
          violation_idx = i;
        }
      }

      if (max_violation > EPSILON) {
        if (try_merge_bins(violation_idx, violation_idx + 1)) {
          any_merge = true;
        }
      }

    } while (any_merge && static_cast<int>(bins.size()) > min_bins);
  }

  // Optimized bin optimization - calculates divergence on-the-fly (no cache)
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins) {
      return;
    }

    int iterations = 0;
    double prev_total_iv = 0.0;

    for (const auto &bin : bins) {
      prev_total_iv += std::fabs(bin.iv);
    }

    while (static_cast<int>(bins.size()) > max_bins &&
           iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }

      double min_score = std::numeric_limits<double>::max();
      size_t min_score_idx1 = 0;
      size_t min_score_idx2 = 0;

      for (size_t i = 0; i < bins.size(); ++i) {
        for (size_t j = i + 1; j < bins.size(); ++j) {
          double score;

          if (use_divergence) {
            // Calculate divergence on-the-fly (no cache)
            score = bins[i].divergence_from(bins[j], total_good, total_bad);
          } else {
            // Calculate IV loss on-the-fly (no cache)
            score = std::fabs(bins[i].iv) + std::fabs(bins[j].iv);
          }

          if (score < min_score) {
            min_score = score;
            min_score_idx1 = i;
            min_score_idx2 = j;
          }
        }
      }

      if (!try_merge_bins(min_score_idx1, min_score_idx2)) {
        break;
      }

      double total_iv = 0.0;
      for (const auto &bin : bins) {
        total_iv += std::fabs(bin.iv);
      }

      if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
        break;
      }

      prev_total_iv = total_iv;
      iterations++;

      if (iterations % 5 == 0) {
        use_divergence = !use_divergence;
      }
    }

    if (static_cast<int>(bins.size()) > max_bins) {
      Rcpp::warning("Could not reduce the number of bins to max_bins without "
                    "violating min_bins or convergence criteria. "
                    "Current bins: " +
                    std::to_string(bins.size()) +
                    ", max_bins: " + std::to_string(max_bins));
    }
  }

  bool try_merge_bins(size_t index1, size_t index2) {
    if (static_cast<int>(bins.size()) <= min_bins) {
      return false;
    }

    if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
      return false;
    }

    if (index2 < index1)
      std::swap(index1, index2);

    bins[index1].merge_with(bins[index2]);
    bins[index1].calculate_metrics(total_good, total_bad);

    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(index2));

    return true;
  }

  void check_consistency() const {
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;

    for (const auto &bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }

    double count_ratio =
        static_cast<double>(total_count) / static_cast<double>(feature.size());
    if (count_ratio < 0.95 || count_ratio > 1.05) {
      Rcpp::warning(
          "Possible inconsistency after binning due to sketch approximation. "
          "Total count: " +
          std::to_string(total_count) +
          ", expected: " + std::to_string(feature.size()) +
          ". Ratio: " + std::to_string(count_ratio));
    }

    double pos_ratio = total_bad > 0 ? static_cast<double>(total_count_pos) /
                                           static_cast<double>(total_bad)
                                     : 0.0;
    double neg_ratio = total_good > 0 ? static_cast<double>(total_count_neg) /
                                            static_cast<double>(total_good)
                                      : 0.0;
    if (pos_ratio < 0.95 || pos_ratio > 1.05 || neg_ratio < 0.95 ||
        neg_ratio > 1.05) {
      Rcpp::warning("Possible inconsistency in positive/negative counts after "
                    "binning due to sketch approximation. "
                    "Positives: " +
                    std::to_string(total_count_pos) + " vs " +
                    std::to_string(total_bad) +
                    ", Negatives: " + std::to_string(total_count_neg) + " vs " +
                    std::to_string(total_good));
    }
  }

public:
  OBC_Sketch(const std::vector<std::string> &feature_,
             const Rcpp::IntegerVector &target_, int min_bins_ = 3,
             int max_bins_ = 5, double bin_cutoff_ = 0.05,
             int max_n_prebins_ = 20, std::string bin_separator_ = "%;%",
             double convergence_threshold_ = 1e-6, int max_iterations_ = 1000,
             size_t sketch_width_ = 2000, size_t sketch_depth_ = 5)
      : feature(feature_), target(Rcpp::as<std::vector<int>>(target_)),
        min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_),
        max_n_prebins(max_n_prebins_), bin_separator(std::move(bin_separator_)),
        convergence_threshold(convergence_threshold_),
        max_iterations(max_iterations_), sketch_width(sketch_width_),
        sketch_depth(sketch_depth_), use_divergence(true), total_good(0),
        total_bad(0) {
    bins.reserve(std::min(max_n_prebins_, 1000));
  }

  Rcpp::List fit() {
    try {
      validate_inputs();
      build_sketches();
      prebinning();
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();

      bool converged_flag = false;
      int iterations_done = 0;

      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto &bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }

        for (int i = 0; i < max_iterations; ++i) {
          size_t start_bins = bins.size();

          optimize_bins();

          if (bins.size() == start_bins ||
              static_cast<int>(bins.size()) <= max_bins) {
            double total_iv = 0.0;
            for (const auto &bin : bins) {
              total_iv += std::fabs(bin.iv);
            }

            if (std::fabs(total_iv - prev_total_iv) < convergence_threshold) {
              converged_flag = true;
              iterations_done = i + 1;
              break;
            }

            prev_total_iv = total_iv;
          }

          iterations_done = i + 1;

          if (static_cast<int>(bins.size()) <= max_bins) {
            break;
          }
        }
      }

      check_consistency();

      const size_t n_bins = bins.size();

      CharacterVector bin_names(n_bins);
      NumericVector bin_woe(n_bins);
      NumericVector bin_iv(n_bins);
      IntegerVector bin_count(n_bins);
      IntegerVector bin_count_pos(n_bins);
      IntegerVector bin_count_neg(n_bins);
      NumericVector ids(n_bins);
      NumericVector event_rates(n_bins);

      for (size_t i = 0; i < n_bins; ++i) {
        bin_names[i] = utils::join(bins[i].categories, bin_separator);
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        event_rates[i] = bins[i].event_rate();
        ids[i] = static_cast<double>(i + 1);
      }

      double total_iv = 0.0;
      for (size_t i = 0; i < n_bins; ++i) {
        total_iv += std::fabs(bin_iv[i]);
      }

      return Rcpp::List::create(
          Named("id") = ids, Named("bin") = bin_names, Named("woe") = bin_woe,
          Named("iv") = bin_iv, Named("count") = bin_count,
          Named("count_pos") = bin_count_pos,
          Named("count_neg") = bin_count_neg, Named("event_rate") = event_rates,
          Named("converged") = converged_flag,
          Named("iterations") = iterations_done, Named("total_iv") = total_iv);
    } catch (const std::exception &e) {
      Rcpp::stop("Error in optimal binning with sketch: " +
                 std::string(e.what()));
    }
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_sketch(
    Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 3,
    int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
    std::string bin_separator = "%;%", double convergence_threshold = 1e-6,
    int max_iterations = 1000, int sketch_width = 2000, int sketch_depth = 5) {
  if (feature.size() == 0 || target.size() == 0) {
    Rcpp::stop("Feature and target cannot be empty.");
  }

  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same size.");
  }

  std::vector<std::string> feature_vec;
  feature_vec.reserve(static_cast<size_t>(feature.size()));

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    if (feature[i] == NA_STRING) {
      feature_vec.push_back(MISSING_VALUE);
    } else {
      feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
    }
  }

  for (R_xlen_t i = 0; i < target.size(); ++i) {
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target cannot contain missing values.");
    }
  }

  OBC_Sketch sketch_binner(feature_vec, target, min_bins, max_bins, bin_cutoff,
                           max_n_prebins, bin_separator, convergence_threshold,
                           max_iterations, static_cast<size_t>(sketch_width),
                           static_cast<size_t>(sketch_depth));

  return sketch_binner.fit();
}
