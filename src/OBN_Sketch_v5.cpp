// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// ============================================================================
// Utilities
// ============================================================================
namespace utils {
// Safe logarithm
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// A finite cut c with a <= c < b between two consecutive distinct finite values
// a < b (right-closed bins). The midpoint is used whenever it is valid; it
// overflows near the largest double and can round up to b for adjacent doubles.
inline double safe_cut(double a, double b) {
  double m = (a + b) / 2.0;
  if (!std::isfinite(m)) m = a / 2.0 + b / 2.0;
  if (m >= a && m < b) return m;
  return a;
}
} // namespace utils

// ============================================================================
// KLL-style sketch for approximate quantiles in a stream
// ============================================================================
//
// Every level holds at most k items; an overflowing level is sorted and its
// adjacent pairs are collapsed into one item carrying the pair's weight, kept
// at the lower element on even levels and at the upper one on odd levels (a
// deterministic alternative to KLL's random coin, so results do not depend on
// the RNG state). The weights are tracked explicitly, so the total weight
// always equals the number of items seen.
//
// Accuracy: a compaction at a level whose items weigh w moves any rank by at
// most w, and a level is compacted at most n / (w k / 2) times, so the rank
// error is at most about 2 n H / k with H = log2(n / k) levels (worst case);
// the alternating choice makes the errors largely cancel in practice. The
// quantiles are only used to propose candidate cutpoints: every bin statistic
// is computed exactly from the data afterwards.
class KLLSketch {
private:
  struct Item {
    double value;
    int weight;

    Item(double v, int w = 1) : value(v), weight(w) {}

    bool operator<(const Item &other) const { return value < other.value; }
  };

  using Compactor = std::vector<Item>;
  std::vector<Compactor> compactors;
  int k;          // Capacity of every level
  int n;          // Number of items processed
  int max_level;  // Levels at or above this one are never compacted

  // Non-recursive compaction of a level (and of the levels it overflows)
  void compact_level(size_t level) {
    std::queue<size_t> levels_to_compact;
    levels_to_compact.push(level);

    while (!levels_to_compact.empty()) {
      const size_t current_level = levels_to_compact.front();
      levels_to_compact.pop();

      if (current_level >= static_cast<size_t>(max_level)) {
        continue;
      }

      std::sort(compactors[current_level].begin(), compactors[current_level].end());

      // The next level must exist BEFORE a reference into `compactors` is taken
      if (current_level + 1 >= compactors.size()) {
        compactors.push_back(Compactor());
      }

      Compactor &compactor = compactors[current_level];
      Compactor &next = compactors[current_level + 1];
      const bool keep_lower = (current_level % 2 == 0);

      for (size_t i = 0; i + 1 < compactor.size(); i += 2) {
        const int w = compactor[i].weight + compactor[i + 1].weight;
        next.push_back(Item(keep_lower ? compactor[i].value : compactor[i + 1].value, w));
      }
      // Odd size: the remaining (largest) item is promoted unchanged
      if (compactor.size() % 2 == 1) {
        next.push_back(compactor.back());
      }
      compactor.clear();

      if (next.size() > static_cast<size_t>(k)) {
        levels_to_compact.push(current_level + 1);
      }
    }
  }

public:
  explicit KLLSketch(int k_param = 200)
    : k(k_param),
      n(0),
      max_level(20) {
    compactors.push_back(Compactor());
    compactors[0].reserve(static_cast<size_t>(k) + 1);
  }

  // Add a (finite) value to the sketch
  void update(double value) {
    compactors[0].push_back(Item(value));
    n++;

    if (compactors[0].size() > static_cast<size_t>(k)) {
      compact_level(0);
    }
  }

  // Estimate the q-th quantile, 0 < q < 1 (after at least one update)
  double get_quantile(double q) const {
    std::vector<Item> flattened;
    for (const auto &compactor : compactors) {
      flattened.insert(flattened.end(), compactor.begin(), compactor.end());
    }
    std::sort(flattened.begin(), flattened.end());

    int total_weight = 0;
    for (const auto &item : flattened) {
      total_weight += item.weight;
    }

    const int target_weight = static_cast<int>(q * total_weight);
    int cumulative_weight = 0;
    for (const auto &item : flattened) {
      cumulative_weight += item.weight;
      if (cumulative_weight >= target_weight) {
        return item.value;
      }
    }
    return flattened.back().value;
  }

  int count() const { return n; }
};

// ============================================================================
// Target statistics per candidate cutpoint
// ============================================================================
struct CutpointStats {
  double cutpoint;
  int count_below;
  int count_pos_below;
  int count_neg_below;
  int count_above;
  int count_pos_above;
  int count_neg_above;
  double iv;

  explicit CutpointStats(double cp = 0.0)
    : cutpoint(cp), count_below(0), count_pos_below(0), count_neg_below(0),
      count_above(0), count_pos_above(0), count_neg_above(0), iv(0.0) {}
};

// ============================================================================
// Exact dynamic programming for small samples
// ============================================================================
class DynamicProgramming {
private:
  std::vector<double> sorted_values;
  // prefix counts: cum_pos[i] = positives among the first i sorted values
  std::vector<int> cum_pos;
  std::vector<int> cum_neg;
  int total_pos;  // Total events (target=1)
  int total_neg;  // Total non-events (target=0)

  // |IV| of the bin made of sorted observations i..j (inclusive)
  double calculate_bin_iv(int i, int j) const {
    const int bin_count_pos = cum_pos[static_cast<size_t>(j) + 1] - cum_pos[static_cast<size_t>(i)];
    const int bin_count_neg = cum_neg[static_cast<size_t>(j) + 1] - cum_neg[static_cast<size_t>(i)];

    double prop_event = static_cast<double>(bin_count_pos) / std::max(total_pos, 1);
    double prop_non_event = static_cast<double>(bin_count_neg) / std::max(total_neg, 1);
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);

    const double woe = utils::safe_log(prop_event / prop_non_event);
    return std::fabs((prop_event - prop_non_event) * woe);
  }

public:
  DynamicProgramming(const std::vector<double> &values,
                     const std::vector<int> &targets)
    : total_pos(0), total_neg(0) {
    std::vector<std::pair<double, int>> paired_data;
    paired_data.reserve(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      paired_data.push_back(std::make_pair(values[i], targets[i]));
    }
    std::sort(paired_data.begin(), paired_data.end());

    sorted_values.reserve(paired_data.size());
    cum_pos.assign(1, 0);
    cum_neg.assign(1, 0);
    for (const auto &pair : paired_data) {
      sorted_values.push_back(pair.first);
      if (pair.second == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
      cum_pos.push_back(total_pos);
      cum_neg.push_back(total_neg);
    }
  }

  // Cutpoints of the partition into at most k bins that maximises the sum of
  // |IV| over bins. Splits are only placed between two distinct values: a
  // split inside a run of ties cannot be realised by any cutpoint, and the
  // earlier version produced duplicate cutpoints and empty bins from them.
  std::vector<double> optimize(int k) {
    const int n = static_cast<int>(sorted_values.size());

    // valid[l]: a split before sorted observation l separates two distinct
    // finite values (-Inf / +Inf stay with the lowest / highest finite values
    // and never produce a cutpoint)
    std::vector<char> valid(static_cast<size_t>(n) + 1, 0);
    int n_distinct = 1;  // 1 + number of valid split positions
    for (int l = 1; l < n; ++l) {
      const double lo = sorted_values[static_cast<size_t>(l) - 1];
      const double hi = sorted_values[static_cast<size_t>(l)];
      if (lo != hi && std::isfinite(lo) && std::isfinite(hi)) {
        valid[static_cast<size_t>(l)] = 1;
        n_distinct++;
      }
    }

    k = std::max(2, std::min(k, std::min(n - 1, 50)));
    k = std::min(k, n_distinct);

    const size_t rows = static_cast<size_t>(n) + 1;
    const size_t cols = static_cast<size_t>(k) + 1;
    std::vector<std::vector<double>> dp_table(rows, std::vector<double>(cols, -1.0));
    std::vector<std::vector<int>> split_points(rows, std::vector<int>(cols, -1));

    // Base case: 1 bin
    dp_table[0][1] = 0.0;
    for (int i = 1; i <= n; ++i) {
      dp_table[static_cast<size_t>(i)][1] = calculate_bin_iv(0, i - 1);
    }

    for (int j = 2; j <= k; ++j) {
      for (int i = j; i <= n; ++i) {
        double &best = dp_table[static_cast<size_t>(i)][static_cast<size_t>(j)];
        for (int l = j - 1; l < i; ++l) {
          if (!valid[static_cast<size_t>(l)]) continue;
          const double prev = dp_table[static_cast<size_t>(l)][static_cast<size_t>(j) - 1];
          if (prev < 0.0) continue;  // infeasible state

          const double current_iv = prev + calculate_bin_iv(l, i - 1);
          if (current_iv > best) {
            best = current_iv;
            split_points[static_cast<size_t>(i)][static_cast<size_t>(j)] = l;
          }
        }
      }
    }

    // Recover the optimal split positions
    std::vector<double> cutpoints;
    int i = n;
    int j = k;
    while (j > 1) {
      const int split = split_points[static_cast<size_t>(i)][static_cast<size_t>(j)];
      cutpoints.push_back(utils::safe_cut(sorted_values[static_cast<size_t>(split) - 1],
                                          sorted_values[static_cast<size_t>(split)]));
      i = split;
      j--;
    }

    std::sort(cutpoints.begin(), cutpoints.end());
    return cutpoints;
  }
};

// ============================================================================
// Optimal numerical binning with a quantile sketch
// ============================================================================
class OBN_Sketch {
private:
  std::vector<double> feature;
  std::vector<int> target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  bool monotonic;
  int max_iterations;
  int sketch_k;
  int dp_size_limit;

  int total_pos;  // Total events (target=1)
  int total_neg;  // Total non-events (target=0)
  double obs_min;  // smallest observation (may be -Inf)
  double obs_max;  // largest observation (may be +Inf)

  std::unique_ptr<KLLSketch> sketch;
  std::vector<NumericalBin> bins;
  std::vector<double> cutpoints;

  // Sizes, missing values and 0/1 targets are handled by the exported
  // wrapper before the class is built. One class may be absent once rows
  // with a missing feature are dropped; WoE and IV are then 0.
  void validate_inputs() const {
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0.0 || bin_cutoff >= 1.0) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1.");
    }
    if (sketch_k < 10) {
      throw std::invalid_argument("sketch_k must be >= 10.");
    }
    if (max_iterations < 1) {
      throw std::invalid_argument("max_iterations must be >= 1.");
    }
  }

  // The sketch summarises the finite values only, so candidate cutpoints are
  // always finite; +/-Inf observations still count in the first / last bin.
  void build_sketch() {
    sketch.reset(new KLLSketch(sketch_k));
    total_pos = 0;
    total_neg = 0;
    obs_min = feature[0];
    obs_max = feature[0];
    for (size_t i = 0; i < feature.size(); ++i) {
      obs_min = std::min(obs_min, feature[i]);
      obs_max = std::max(obs_max, feature[i]);
      if (std::isfinite(feature[i])) sketch->update(feature[i]);
      if (target[i] == 1) {
        total_pos++;
      } else {
        total_neg++;
      }
    }
  }

  // Candidate cutpoints: sketch quantiles, finer in the tails
  std::vector<double> extract_candidates() const {
    std::vector<double> candidates;
    candidates.reserve(40);

    for (double q = 0.01; q <= 0.1; q += 0.01) {
      candidates.push_back(sketch->get_quantile(q));
      candidates.push_back(sketch->get_quantile(1.0 - q));
    }
    for (double q = 0.1; q <= 0.9; q += 0.05) {
      candidates.push_back(sketch->get_quantile(q));
    }

    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
  }

  // Split statistics of every candidate (sorted, distinct) in one pass:
  // each observation is located by binary search among the candidates and
  // the per-slot counts are accumulated, instead of testing every candidate
  // against every observation.
  std::vector<CutpointStats> calculate_cutpoint_stats(const std::vector<double> &candidates) const {
    const size_t C = candidates.size();
    std::vector<int> slot_pos(C + 1, 0);
    std::vector<int> slot_neg(C + 1, 0);
    for (size_t i = 0; i < feature.size(); ++i) {
      // first candidate >= value: the value is "below" it and every later one
      const size_t s = static_cast<size_t>(
        std::lower_bound(candidates.begin(), candidates.end(), feature[i]) - candidates.begin());
      if (target[i]) slot_pos[s]++; else slot_neg[s]++;
    }

    std::vector<CutpointStats> stats;
    stats.reserve(C);
    int below_pos = 0;
    int below_neg = 0;
    const int n = static_cast<int>(feature.size());
    for (size_t c = 0; c < C; ++c) {
      below_pos += slot_pos[c];
      below_neg += slot_neg[c];
      CutpointStats cs(candidates[c]);
      cs.count_pos_below = below_pos;
      cs.count_neg_below = below_neg;
      cs.count_below = below_pos + below_neg;
      cs.count_pos_above = total_pos - below_pos;
      cs.count_neg_above = total_neg - below_neg;
      cs.count_above = n - cs.count_below;
      stats.push_back(cs);
    }

    for (auto &cs : stats) {
      if (cs.count_below == 0 || cs.count_above == 0) {
        cs.iv = 0.0;
        continue;
      }

      double prop_event_below = static_cast<double>(cs.count_pos_below) / std::max(total_pos, 1);
      double prop_non_event_below = static_cast<double>(cs.count_neg_below) / std::max(total_neg, 1);
      prop_event_below = std::max(prop_event_below, EPSILON);
      prop_non_event_below = std::max(prop_non_event_below, EPSILON);
      const double woe_below = utils::safe_log(prop_event_below / prop_non_event_below);
      const double iv_below = (prop_event_below - prop_non_event_below) * woe_below;

      double prop_event_above = static_cast<double>(cs.count_pos_above) / std::max(total_pos, 1);
      double prop_non_event_above = static_cast<double>(cs.count_neg_above) / std::max(total_neg, 1);
      prop_event_above = std::max(prop_event_above, EPSILON);
      prop_non_event_above = std::max(prop_non_event_above, EPSILON);
      const double woe_above = utils::safe_log(prop_event_above / prop_non_event_above);
      const double iv_above = (prop_event_above - prop_non_event_above) * woe_above;

      cs.iv = std::fabs(iv_below) + std::fabs(iv_above);
    }

    return stats;
  }

  // Exact DP for small samples, greedy IV ranking of the candidates otherwise
  void select_optimal_cutpoints(const std::vector<double> &candidates) {
    cutpoints.clear();
    if (feature.size() <= static_cast<size_t>(dp_size_limit)) {
      DynamicProgramming dp(feature, target);
      cutpoints = dp.optimize(max_bins);
    } else {
      std::vector<CutpointStats> stats = calculate_cutpoint_stats(candidates);
      std::sort(stats.begin(), stats.end(),
                [](const CutpointStats &a, const CutpointStats &b) {
                  return a.iv > b.iv;
                });

      const size_t n_take = std::min(stats.size(), static_cast<size_t>(max_bins - 1));
      for (size_t i = 0; i < n_take; ++i) {
        cutpoints.push_back(stats[i].cutpoint);
      }
      std::sort(cutpoints.begin(), cutpoints.end());
    }

    create_initial_bins();
  }

  // Bins [min, c1], (c1, c2], ..., (c_m, max]; each observation is located by
  // binary search (first cutpoint >= value), which is exactly the bin the
  // former linear scan picked.
  void create_initial_bins() {
    bins.clear();

    const double min_val = obs_min;
    const double max_val = obs_max;

    if (cutpoints.empty()) {
      bins.push_back(NumericalBin(min_val, max_val));
    } else {
      bins.push_back(NumericalBin(min_val, cutpoints[0]));
      for (size_t i = 0; i + 1 < cutpoints.size(); ++i) {
        bins.push_back(NumericalBin(cutpoints[i], cutpoints[i + 1]));
      }
      bins.push_back(NumericalBin(cutpoints.back(), max_val));
    }

    for (size_t i = 0; i < feature.size(); ++i) {
      const size_t b = static_cast<size_t>(
        std::lower_bound(cutpoints.begin(), cutpoints.end(), feature[i]) - cutpoints.begin());
      bins[b].add_value(target[i]);
    }
  }

  // Merge bins below bin_cutoff into the neighbour with the closest event
  // rate, down to min_bins. Empty bins -- a candidate equal to the maximum,
  // or fewer distinct values than min_bins -- are merged even below
  // min_bins: they carry no observation and used to be returned as bins
  // with count 0 and duplicate cutpoints.
  void enforce_bin_cutoff() {
    const int min_count = static_cast<int>(std::ceil(bin_cutoff * static_cast<double>(feature.size())));

    bool any_change = true;
    while (any_change) {
      any_change = false;

      for (size_t i = 0; i < bins.size() && bins.size() > 1; ++i) {
        if (bins[i].count >= min_count) continue;
        if (bins.size() <= static_cast<size_t>(min_bins) && bins[i].count > 0) continue;

        size_t best_neighbor = i + 1;
        double min_diff = std::numeric_limits<double>::max();
        if (i > 0) {
          min_diff = std::fabs(bins[i].event_rate() - bins[i - 1].event_rate());
          best_neighbor = i - 1;
        }
        if (i + 1 < bins.size()) {
          const double diff = std::fabs(bins[i].event_rate() - bins[i + 1].event_rate());
          if (diff < min_diff) {
            best_neighbor = i + 1;
          }
        }

        // Always merge into the lower index
        if (best_neighbor < i) {
          bins[best_neighbor].merge_with(bins[i]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));
        } else {
          bins[i].merge_with(bins[best_neighbor]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(best_neighbor));
        }
        any_change = true;
        break;
      }
    }

    update_cutpoints_from_bins();
  }

  void calculate_initial_woe() {
    for (auto &bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }

  // Pool adjacent violators until the WoE is monotonic or min_bins is reached
  void enforce_monotonicity() {
    if (!monotonic || bins.size() <= 1) {
      return;
    }

    const bool increasing = bins.back().woe >= bins.front().woe;

    bool any_change = true;
    while (any_change && bins.size() > static_cast<size_t>(min_bins)) {
      any_change = false;
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const bool violation = (increasing && bins[i].woe > bins[i + 1].woe + EPSILON) ||
          (!increasing && bins[i].woe < bins[i + 1].woe - EPSILON);
        if (violation) {
          bins[i].merge_with(bins[i + 1]);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i) + 1);
          bins[i].calculate_metrics(total_pos, total_neg);
          any_change = true;
          break;
        }
      }
    }

    update_cutpoints_from_bins();
  }

  void update_cutpoints_from_bins() {
    cutpoints.clear();
    for (size_t i = 1; i < bins.size(); ++i) {
      cutpoints.push_back(bins[i].lower_bound);
    }
  }

public:
  OBN_Sketch(const std::vector<double> &feature_,
             const std::vector<int> &target_,
             int min_bins_ = 3,
             int max_bins_ = 5,
             double bin_cutoff_ = 0.05,
             bool monotonic_ = true,
             int max_iterations_ = 1000,
             int sketch_k_ = 200)
    : feature(feature_),
      target(target_),
      min_bins(min_bins_),
      max_bins(max_bins_),
      bin_cutoff(bin_cutoff_),
      monotonic(monotonic_),
      max_iterations(max_iterations_),
      sketch_k(sketch_k_),
      dp_size_limit(50),
      total_pos(0),
      total_neg(0),
      obs_min(0.0),
      obs_max(0.0) {}

  Rcpp::List fit() {
    try {
      validate_inputs();
      build_sketch();
      select_optimal_cutpoints(extract_candidates());
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();

      // DP and greedy selection both return at most max_bins - 1 cutpoints
      // and every later step only merges, so bins.size() <= max_bins holds
      // here and no further reduction is needed: the binning has converged.
      const bool converged_flag = true;
      const int iterations_done = 0;

      const size_t n_bins = bins.size();
      const R_xlen_t nb = static_cast<R_xlen_t>(n_bins);

      NumericVector bin_lower(nb);
      NumericVector bin_upper(nb);
      NumericVector bin_woe(nb);
      NumericVector bin_iv(nb);
      IntegerVector bin_count(nb);
      IntegerVector bin_count_pos(nb);
      IntegerVector bin_count_neg(nb);
      NumericVector ids(nb);
      CharacterVector bin_labels(nb);
      double total_iv_value = 0.0;

      for (R_xlen_t i = 0; i < nb; ++i) {
        const NumericalBin &b = bins[static_cast<size_t>(i)];
        bin_lower[i] = b.lower_bound;
        bin_upper[i] = b.upper_bound;
        bin_woe[i] = b.woe;
        bin_iv[i] = b.iv;
        bin_count[i] = b.count;
        bin_count_pos[i] = b.count_pos;
        bin_count_neg[i] = b.count_neg;
        ids[i] = static_cast<double>(i + 1);
        total_iv_value += b.iv;

        // "(lower;upper]" like every sibling algorithm. The first bin is
        // labelled from -Inf and the last up to +Inf: they hold everything
        // below the first / above the last cutpoint (the observed minimum is
        // inside the first bin, which "(min;" wrongly excluded).
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        if (i == 0) {
          oss << "(-Inf;";
        } else {
          oss << "(" << b.lower_bound << ";";
        }
        if (i == nb - 1) {
          oss << "+Inf]";
        } else {
          oss << b.upper_bound << "]";
        }
        bin_labels[i] = oss.str();
      }

      return Rcpp::List::create(
        Named("id") = ids,
        Named("bin") = bin_labels,
        Named("bin_lower") = bin_lower,
        Named("bin_upper") = bin_upper,
        Named("woe") = bin_woe,
        Named("iv") = bin_iv,
        Named("count") = bin_count,
        Named("count_pos") = bin_count_pos,
        Named("count_neg") = bin_count_neg,
        Named("total_iv") = total_iv_value,
        Named("cutpoints") = cutpoints,
        Named("converged") = converged_flag,
        Named("iterations") = iterations_done);

    } catch (const std::exception &e) {
      Rcpp::stop("Binning error: " + std::string(e.what()));
    }
  }
};

// ============================================================================
// Exported function for R
// ============================================================================
// [[Rcpp::export]]
Rcpp::List optimal_binning_numerical_sketch(
    Rcpp::IntegerVector target,
    Rcpp::NumericVector feature,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    bool monotonic = true,
    double convergence_threshold = 1e-6,
    int max_iterations = 1000,
    int sketch_k = 200) {
  // max_n_prebins and convergence_threshold are accepted for API
  // compatibility: the candidate grid is a fixed set of ~37 sketch quantiles
  // and the selected cutpoints never need an iterative reduction.
  (void)max_n_prebins;
  (void)convergence_threshold;

  if (feature.size() == 0 || target.size() == 0) {
    Rcpp::stop("Feature and target cannot be empty.");
  }
  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same size.");
  }

  std::vector<double> feature_vec;
  std::vector<int> target_vec;
  feature_vec.reserve(static_cast<size_t>(feature.size()));
  target_vec.reserve(static_cast<size_t>(target.size()));

  bool has_zero = false;
  bool has_one = false;
  for (R_xlen_t i = 0; i < target.size(); ++i) {
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target cannot contain missing values (NA).");
    }
    if (target[i] != 0 && target[i] != 1) {
      Rcpp::stop("Target must contain only 0 and 1.");
    }
    if (target[i] == 0) has_zero = true; else has_one = true;
  }
  if (!has_zero || !has_one) {
    Rcpp::stop("Target must contain both 0 and 1.");
  }

  // Rows with a missing feature (NA / NaN) are dropped silently; +/-Inf are
  // kept as extreme values.
  double fin_min = std::numeric_limits<double>::infinity();
  double fin_max = -std::numeric_limits<double>::infinity();
  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    const double v = feature[i];
    if (std::isnan(v)) continue;
    if (std::isfinite(v)) {
      fin_min = std::min(fin_min, v);
      fin_max = std::max(fin_max, v);
    }
    feature_vec.push_back(v);
    target_vec.push_back(target[i]);
  }
  if (feature_vec.empty()) {
    Rcpp::stop("All feature values are missing (NA/NaN); nothing to bin.");
  }

  const double min_val = *std::min_element(feature_vec.begin(), feature_vec.end());
  const double max_val = *std::max_element(feature_vec.begin(), feature_vec.end());

  // Fewer than two distinct finite values (constant feature, or only +/-Inf
  // besides one value): a single bin. (A tolerance of 1e-10 used to collapse
  // genuinely distinct small-scale features as well.)
  if (!(fin_min < fin_max)) {
    int count_pos = 0;
    int count_neg = 0;
    for (int t : target_vec) {
      if (t == 1) count_pos++;
      else count_neg++;
    }
    const int total_count = static_cast<int>(feature_vec.size());

    // A single bin holds every observation: WoE = log(1 / 1) = 0, IV = 0.
    const double woe = 0.0;
    const double iv = 0.0;

    return Rcpp::List::create(
      Named("id") = NumericVector::create(1),
      Named("bin") = CharacterVector::create("(-Inf;+Inf]"),
      Named("bin_lower") = NumericVector::create(min_val),
      Named("bin_upper") = NumericVector::create(max_val),
      Named("woe") = NumericVector::create(woe),
      Named("iv") = NumericVector::create(iv),
      Named("count") = IntegerVector::create(total_count),
      Named("count_pos") = IntegerVector::create(count_pos),
      Named("count_neg") = IntegerVector::create(count_neg),
      Named("total_iv") = iv,
      Named("cutpoints") = NumericVector::create(),
      Named("converged") = true,
      Named("iterations") = 0);
  }

  OBN_Sketch sketch_binner(feature_vec, target_vec, min_bins, max_bins,
                           bin_cutoff, monotonic, max_iterations, sketch_k);
  return sketch_binner.fit();
}
