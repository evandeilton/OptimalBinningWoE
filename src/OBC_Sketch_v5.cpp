// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>


// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"
#include "common/ob_core.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Global constants
static constexpr double LAPLACE_ALPHA = 0.5;
// Strength (pseudo-observations) of the empirical-Bayes beta prior, centred on
// the overall event rate, used for the merge decisions (audit E-03 follow-up,
// Section 15.5); same value as OBC_MBA. WoE/IV are still reported with the
// estimator of CategoricalBin::calculate_metrics().
static constexpr double EVIDENCE_M = 20.0;
// [D8] Standardized to "NA", matching every other categorical algorithm
// (this file and OBC_JEDIMWoE_v5.cpp were the only two using "N/A"). The
// R wrapper (R/obc_sketch.R) already converts NA to this token before the
// vector reaches here, so this constant is normally unreachable, but is
// kept in sync for any caller that invokes this .Call() target directly.
static constexpr const char *MISSING_VALUE = "NA";

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
    // Through unsigned char: plain char is signed on some platforms only.
    hash ^= static_cast<size_t>(static_cast<unsigned char>(c));
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

    // Seeds straight from mt19937, whose output is fixed by the standard;
    // uniform_int_distribution differs between C++ libraries.
    std::mt19937 gen(42);
    seeds.resize(depth);
    for (size_t i = 0; i < depth; ++i) {
      const uint64_t hi = gen();
      const uint64_t lo = gen();
      seeds[i] = static_cast<size_t>((hi << 32) | lo);
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

    int threshold = static_cast<int>(static_cast<double>(total_count) * threshold_ratio);

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
  // Distinct categories in name order, with their exact class counts, and
  // the number of observations.
  std::vector<std::string> categories;
  std::vector<int> cat_pos;
  std::vector<int> cat_neg;
  std::unordered_map<std::string, size_t> cat_index;
  size_t n_obs;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;
  size_t sketch_width;
  size_t sketch_depth;

  int total_neg;
  int total_pos;

  std::vector<CategoricalBin> bins;
  std::unique_ptr<CountMinSketch> sketch;
  std::unique_ptr<CountMinSketch> sketch_pos;
  std::unique_ptr<CountMinSketch> sketch_neg;

  void validate_inputs() {
    if (n_obs == 0 || categories.empty()) {
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

    int all_pos = 0, all_neg = 0;
    for (size_t c = 0; c < categories.size(); ++c) {
      all_pos += cat_pos[c];
      all_neg += cat_neg[c];
    }
    if (all_pos == 0 || all_neg == 0) {
      throw std::invalid_argument("Target must contain both 0 and 1.");
    }
  }

  // Exact class counts of a category. The bin statistics used to be read
  // back from the Count-Min sketches, whose estimates are inflated by hash
  // collisions: with a few hundred categories the reported counts no longer
  // summed to n (and to the numbers of 0s and 1s), and the WoE/IV were
  // computed from those inflated counts. The sketches are still what decides
  // which categories are heavy hitters.
  std::pair<int, int> exact_counts(const std::string &cat) const {
    auto it = cat_index.find(cat);
    if (it == cat_index.end()) return {0, 0};
    return {cat_pos[it->second], cat_neg[it->second]};
  }

  void build_sketches() {
    sketch = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_pos = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);
    sketch_neg = std::make_unique<CountMinSketch>(sketch_width, sketch_depth);

    total_neg = 0;
    total_pos = 0;

    // Count-Min updates are additive, so adding each category's count in one
    // update leaves exactly the tables the per-observation updates built.
    for (size_t c = 0; c < categories.size(); ++c) {
      const std::string &cat = categories[c];
      sketch->update(cat, cat_pos[c] + cat_neg[c]);
      sketch_pos->update(cat, cat_pos[c]);
      sketch_neg->update(cat, cat_neg[c]);
      total_pos += cat_pos[c];
      total_neg += cat_neg[c];
    }

    int ncat = static_cast<int>(categories.size());
    if (max_bins > ncat) {
      max_bins = ncat;
    }
    min_bins = std::min(min_bins, max_bins);
  }

  void prebinning() {
    // Distinct categories in name order (see the entry point).
    const std::vector<std::string> &unique_categories = categories;

    double heavy_hitter_threshold = bin_cutoff / 2.0;
    std::vector<std::string> heavy_hitters =
        sketch->heavy_hitters(unique_categories, heavy_hitter_threshold);

    bins.clear();
    bins.reserve(heavy_hitters.size());

    std::unordered_set<std::string> heavy_set(heavy_hitters.begin(),
                                              heavy_hitters.end());
    for (const auto &cat : heavy_hitters) {
      CategoricalBin bin;
      const auto counts = exact_counts(cat);
      int pos_count = counts.first;
      int neg_count = counts.second;
      bin.categories.push_back(cat);
      bin.count_pos += pos_count;
      bin.count_neg += neg_count;
      bin.update_count();
      bins.push_back(bin);
    }

    // Non-heavy categories are pooled into one "rare" bin. When the heavy
    // hitters plus that pool cannot reach min_bins, the largest rare
    // categories keep their own bins (only as many as needed); the result
    // used to have fewer than min_bins bins although more were possible.
    std::vector<std::string> rare_cats;
    for (const auto &cat : unique_categories) {
      if (heavy_set.count(cat) == 0) rare_cats.push_back(cat);
    }
    const size_t n_heavy = bins.size();
    const size_t n_rare = rare_cats.size();
    const size_t target = static_cast<size_t>(std::max(min_bins, 0));
    size_t keep = 0;
    while (keep < n_rare &&
           n_heavy + keep + (n_rare > keep ? 1 : 0) < target) {
      ++keep;
    }
    std::unordered_set<std::string> promoted;
    if (keep > 0) {
      std::vector<size_t> order(n_rare);
      for (size_t i = 0; i < n_rare; ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        const auto cx = exact_counts(rare_cats[x]);
        const auto cy = exact_counts(rare_cats[y]);
        return cx.first + cx.second > cy.first + cy.second;
      });
      for (size_t i = 0; i < keep; ++i) promoted.insert(rare_cats[order[i]]);
    }

    CategoricalBin rare_bin;
    for (const auto &cat : rare_cats) {
      const auto counts = exact_counts(cat);
      if (promoted.count(cat)) {
        CategoricalBin bin;
        bin.categories.push_back(cat);
        bin.count_pos = counts.first;
        bin.count_neg = counts.second;
        bin.update_count();
        bins.push_back(bin);
        continue;
      }
      rare_bin.categories.push_back(cat);
      rare_bin.count_pos += counts.first;
      rare_bin.count_neg += counts.second;
      rare_bin.update_count();
    }

    if (rare_bin.count > 0) {
      bins.push_back(rare_bin);
    }

    // Descending count; equal counts keep the order above (stable sort).
    std::stable_sort(bins.begin(), bins.end(),
                     [](const CategoricalBin &a, const CategoricalBin &b) {
                       return a.count > b.count;
                     });

    if (static_cast<int>(bins.size()) > max_n_prebins &&
        static_cast<int>(bins.size()) > min_bins) {
      reduce_to_max_prebins();
    }
  }

  // Greedy reduction to max_n_prebins: repeatedly merge the pair (i < j)
  // with the smallest size-penalised divergence, the lexicographically first
  // pair on ties. The bins are not reordered while merging, so each bin keeps
  // its best partner to the right (score, then smallest j); after a merge
  // only the merged bin and the bins whose best partner was involved are
  // rescanned. This picks exactly the pairs of the former full O(B^2) scan
  // per merge (O(B^3) overall: minutes with a few thousand heavy hitters) in
  // about O(B^2) total.
  double prebin_score(size_t i, size_t j) const {
    int combined_count = bins[i].count + bins[j].count;
    double size_penalty = 1.0 + std::log(1.0 + combined_count);
    return bins[i].divergence_from(bins[j], total_pos, total_neg) * size_penalty;
  }

  void reduce_to_max_prebins() {
    const size_t B = bins.size();
    const size_t NONE = static_cast<size_t>(-1);
    const double NO_SCORE = std::numeric_limits<double>::max();
    std::vector<char> alive(B, 1);
    std::vector<double> best_score(B, NO_SCORE);
    std::vector<size_t> best_j(B, NONE);

    auto rescan = [&](size_t i) {
      best_score[i] = NO_SCORE;
      best_j[i] = NONE;
      for (size_t j = i + 1; j < B; ++j) {
        if (!alive[j]) continue;
        const double sc = prebin_score(i, j);
        if (sc < best_score[i]) {
          best_score[i] = sc;
          best_j[i] = j;
        }
      }
    };
    for (size_t i = 0; i < B; ++i) rescan(i);

    size_t remaining = B;
    while (static_cast<int>(remaining) > max_n_prebins &&
           static_cast<int>(remaining) > min_bins) {
      size_t a = NONE, b = NONE;
      double best = NO_SCORE;
      for (size_t i = 0; i < B; ++i) {
        if (alive[i] && best_j[i] != NONE && best_score[i] < best) {
          best = best_score[i];
          a = i;
          b = best_j[i];
        }
      }
      if (a == NONE) {  // no finite score: the old scan merged the first two
        size_t k = 0;
        while (!alive[k]) ++k;
        a = k++;
        while (!alive[k]) ++k;
        b = k;
      }

      bins[a].merge_with(bins[b]);
      bins[a].calculate_metrics(total_pos, total_neg);
      alive[b] = 0;
      --remaining;

      rescan(a);
      for (size_t i = 0; i < b; ++i) {
        if (!alive[i] || i == a) continue;
        if (best_j[i] == a || best_j[i] == b) {
          rescan(i);
        } else if (i < a) {
          const double sc = prebin_score(i, a);
          if (sc < best_score[i] || (sc == best_score[i] && a < best_j[i])) {
            best_score[i] = sc;
            best_j[i] = a;
          }
        }
      }
    }

    std::vector<CategoricalBin> kept;
    kept.reserve(remaining);
    for (size_t i = 0; i < B; ++i) {
      if (alive[i]) kept.push_back(std::move(bins[i]));
    }
    bins = std::move(kept);
  }

  // Empirical-Bayes (posterior mean) event rate of a bin under a beta prior
  // centred on the overall rate with EVIDENCE_M pseudo-observations.
  double shrunk_rate(const CategoricalBin &bin) const {
    const double r0 = static_cast<double>(total_pos) /
                      static_cast<double>(total_pos + total_neg);
    return (static_cast<double>(bin.count_pos) + EVIDENCE_M * r0) /
           (static_cast<double>(bin.count) + EVIDENCE_M);
  }

  // bin_cutoff is a share of *observations* (documented contract). The
  // baseline also required count_pos >= bin_cutoff * total_pos, which merged
  // large low-risk bins and every pure non-event bin (audit E-03 / X-15).
  // A bin below the cutoff joins the bin with the closest shrunk event rate
  // (posterior mean, Gelman et al., Bayesian Data Analysis, ch. 5) instead of
  // the closest raw shares: the raw rate of a small bin is mostly noise.
  void enforce_bin_cutoff() {
    int min_count = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(n_obs)));

    std::vector<size_t> low_freq_bins;

    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count) {
        low_freq_bins.push_back(i);
      }
    }

    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break;
      }

      if (idx >= bins.size() || bins[idx].count >= min_count) {
        continue;
      }

      size_t merge_idx = idx;
      double min_divergence = std::numeric_limits<double>::max();
      const double rate_idx = shrunk_rate(bins[idx]);

      for (size_t i = 0; i < bins.size(); ++i) {
        if (i == idx)
          continue;

        double div = std::fabs(rate_idx - shrunk_rate(bins[i]));
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

      const size_t kept = std::min(idx, merge_idx);
      const size_t erased = std::max(idx, merge_idx);
      if (!try_merge_bins(kept, erased)) {
        continue;
      }

      // try_merge_bins() keeps the lower position and erases the higher one:
      // an entry naming the erased bin now names the merged bin, and every
      // later position moves down by one. The baseline decremented the
      // entries above merge_idx instead: when the partner was the lower
      // position the entries between the two named the wrong bin, and when it
      // was the higher one the pending entry of the erased bin was lost, so a
      // bin below bin_cutoff could survive (audit E-02; same fix as OBC_MBA).
      for (auto &remaining_idx : low_freq_bins) {
        if (remaining_idx == erased) {
          remaining_idx = kept;
        } else if (remaining_idx > erased) {
          remaining_idx--;
        }
      }
    }
  }

  void calculate_initial_woe() {
    for (auto &bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }

  void enforce_monotonicity() {
    if (bins.empty()) {
      throw std::runtime_error("No bins available to enforce monotonicity.");
    }

    // Ascending WoE; ties by ascending count, then the current order.
    std::stable_sort(bins.begin(), bins.end(),
                     [](const CategoricalBin &a, const CategoricalBin &b) {
                       if (a.woe != b.woe) return a.woe < b.woe;
                       return a.count < b.count;
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

  // Reduce to max_bins by merging, among the bins adjacent in WoE order
  // (enforce_monotonicity() sorted them), the pair that loses the least IV
  // under the empirical-Bayes prior (ob::eb_woe_iv, EVIDENCE_M pseudo-
  // observations per bin, shares normalized over the current K bins). The
  // baseline alternated every 5 merges between the Jensen-Shannon divergence
  // and the sum |IV_i| + |IV_j| over *any* pair of bins; the latter merged the
  // two least informative bins wherever they sat (a pure category and a mid
  // bin), and after E-03 kept a pure bin of 10% of the rows apart while
  // pooling two large bins with rates 0.3 and 0.6 (hold-out log-loss +5%).
  void optimize_bins() {
    const double Tp = static_cast<double>(total_pos);
    const double Tn = static_cast<double>(total_neg);
    int iterations = 0;
    while (static_cast<int>(bins.size()) > max_bins &&
           static_cast<int>(bins.size()) > min_bins &&
           iterations < max_iterations) {
      const int K = static_cast<int>(bins.size());
      double min_loss = std::numeric_limits<double>::max();
      size_t min_idx = 0;
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double pa = bins[i].count_pos, na = bins[i].count_neg;
        const double pb = bins[i + 1].count_pos, nb = bins[i + 1].count_neg;
        const double loss =
            ob::eb_woe_iv(pa, na, Tp, Tn, K, EVIDENCE_M).iv +
            ob::eb_woe_iv(pb, nb, Tp, Tn, K, EVIDENCE_M).iv -
            ob::eb_woe_iv(pa + pb, na + nb, Tp, Tn, K, EVIDENCE_M).iv;
        if (loss < min_loss) {
          min_loss = loss;
          min_idx = i;
        }
      }
      if (!try_merge_bins(min_idx, min_idx + 1)) {
        break;
      }
      iterations++;
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
    bins[index1].calculate_metrics(total_pos, total_neg);

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
        static_cast<double>(total_count) / static_cast<double>(n_obs);
    if (count_ratio < 0.95 || count_ratio > 1.05) {
      Rcpp::warning(
          "Possible inconsistency after binning due to sketch approximation. "
          "Total count: " +
          std::to_string(total_count) +
          ", expected: " + std::to_string(n_obs) +
          ". Ratio: " + std::to_string(count_ratio));
    }

    double pos_ratio = total_pos > 0 ? static_cast<double>(total_count_pos) /
                                           static_cast<double>(total_pos)
                                     : 0.0;
    double neg_ratio = total_neg > 0 ? static_cast<double>(total_count_neg) /
                                            static_cast<double>(total_neg)
                                      : 0.0;
    if (pos_ratio < 0.95 || pos_ratio > 1.05 || neg_ratio < 0.95 ||
        neg_ratio > 1.05) {
      Rcpp::warning("Possible inconsistency in positive/negative counts after "
                    "binning due to sketch approximation. "
                    "Positives: " +
                    std::to_string(total_count_pos) + " vs " +
                    std::to_string(total_pos) +
                    ", Negatives: " + std::to_string(total_count_neg) + " vs " +
                    std::to_string(total_neg));
    }
  }

public:
  OBC_Sketch(std::vector<std::string> categories_, std::vector<int> pos_,
             std::vector<int> neg_, size_t n_obs_, int min_bins_ = 3,
             int max_bins_ = 5, double bin_cutoff_ = 0.05,
             int max_n_prebins_ = 20, std::string bin_separator_ = "%;%",
             double convergence_threshold_ = 1e-6, int max_iterations_ = 1000,
             size_t sketch_width_ = 2000, size_t sketch_depth_ = 5)
      : categories(std::move(categories_)), cat_pos(std::move(pos_)),
        cat_neg(std::move(neg_)), n_obs(n_obs_),
        min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_),
        max_n_prebins(max_n_prebins_), bin_separator(std::move(bin_separator_)),
        convergence_threshold(convergence_threshold_),
        max_iterations(max_iterations_), sketch_width(sketch_width_),
        sketch_depth(sketch_depth_), total_neg(0),
        total_pos(0) {
    cat_index.reserve(categories.size());
    for (size_t c = 0; c < categories.size(); ++c) {
      cat_index.emplace(categories[c], c);
    }
    bins.reserve(static_cast<size_t>(std::max(0, std::min(max_n_prebins_, 1000))));
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

        // Leaving the loop because the bin-count target was reached is a valid
        // stopping state, just like meeting the IV tolerance above. Only
        // exhausting max_iterations leaves converged_flag == false.
        if (iterations_done < max_iterations) {
          converged_flag = converged_flag ||
                           (static_cast<int>(bins.size()) <= max_bins);
        }
      }

      // max_bins is a hard limit. When max_iterations ran out first (it caps
      // the merges of optimize_bins()), the result used to be returned with
      // more than max_bins bins and a warning -- repeated on every outer
      // iteration. Finish the reduction with the most similar pairs (the
      // divergence criterion of optimize_bins()); 'converged' stays FALSE.
      while (static_cast<int>(bins.size()) > max_bins) {
        size_t best_i = 0, best_j = 1;
        double best = std::numeric_limits<double>::max();
        for (size_t i = 0; i < bins.size(); ++i) {
          for (size_t j = i + 1; j < bins.size(); ++j) {
            const double div = bins[i].divergence_from(bins[j], total_pos, total_neg);
            if (div < best) {
              best = div;
              best_i = i;
              best_j = j;
            }
          }
        }
        if (!try_merge_bins(best_i, best_j)) break;
      }

      // optimize_bins() may merge two non-adjacent bins, which left the bins
      // out of WoE order although monotonic WoE is documented. Categorical
      // bins have no intrinsic order, so listing them by WoE restores it;
      // only a non-monotonic sequence is reordered.
      if (bins.size() > 2) {
        bool inc = true, dec = true;
        for (size_t i = 1; i < bins.size(); ++i) {
          if (bins[i].woe < bins[i - 1].woe) inc = false;
          if (bins[i].woe > bins[i - 1].woe) dec = false;
        }
        if (!inc && !dec) {
          // Ties by ascending count, then the current order.
          std::stable_sort(bins.begin(), bins.end(),
                           [](const CategoricalBin &a, const CategoricalBin &b) {
                             if (a.woe != b.woe) return a.woe < b.woe;
                             return a.count < b.count;
                           });
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
        ids[static_cast<R_xlen_t>(i)] = static_cast<double>(i + 1);
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

  if (sketch_width < 100) {
    Rcpp::stop("Error in optimal binning with sketch: sketch_width must be >= "
               "100 for reasonable accuracy.");
  }
  if (sketch_depth < 3) {
    Rcpp::stop("Error in optimal binning with sketch: sketch_depth must be >= "
               "3 for reasonable accuracy.");
  }

  // Aggregate per distinct category. Each distinct CHARSXP is resolved once;
  // the string map merges equal byte strings stored under different encodings.
  std::vector<std::string> categories;
  std::vector<int> pos, neg;
  std::unordered_map<std::string, size_t> str_index;
  std::unordered_map<SEXP, size_t> ptr_index;
  const int *tg = INTEGER(target);
  const R_xlen_t n = feature.size();

  for (R_xlen_t i = 0; i < n; ++i) {
    const int t = tg[i];
    if (t == NA_INTEGER) {
      Rcpp::stop("Target cannot contain missing values.");
    }
    if (t != 0 && t != 1) {
      Rcpp::stop("Error in optimal binning with sketch: Target must contain "
                 "only 0 and 1.");
    }
    SEXP cs = STRING_ELT(feature, i);
    size_t idx;
    auto pit = ptr_index.find(cs);
    if (pit != ptr_index.end()) {
      idx = pit->second;
    } else {
      std::string cat = (cs == NA_STRING) ? std::string(MISSING_VALUE)
                                          : std::string(CHAR(cs));
      auto ins = str_index.emplace(cat, categories.size());
      if (ins.second) {
        categories.push_back(std::move(cat));
        pos.push_back(0);
        neg.push_back(0);
      }
      idx = ins.first->second;
      ptr_index.emplace(cs, idx);
    }
    if (t == 1)
      pos[idx]++;
    else
      neg[idx]++;
  }

  // Name order, so the fit does not depend on the row order.
  std::vector<size_t> by_name(categories.size());
  std::iota(by_name.begin(), by_name.end(), 0);
  std::sort(by_name.begin(), by_name.end(), [&categories](size_t a, size_t b) {
    return categories[a] < categories[b];
  });
  std::vector<std::string> cats_sorted;
  std::vector<int> pos_sorted, neg_sorted;
  cats_sorted.reserve(by_name.size());
  pos_sorted.reserve(by_name.size());
  neg_sorted.reserve(by_name.size());
  for (size_t k : by_name) {
    cats_sorted.push_back(std::move(categories[k]));
    pos_sorted.push_back(pos[k]);
    neg_sorted.push_back(neg[k]);
  }

  OBC_Sketch sketch_binner(std::move(cats_sorted), std::move(pos_sorted),
                           std::move(neg_sorted), static_cast<size_t>(n), min_bins,
                           max_bins, bin_cutoff, max_n_prebins, bin_separator,
                           convergence_threshold, max_iterations,
                           static_cast<size_t>(sketch_width),
                           static_cast<size_t>(sketch_depth));

  return sketch_binner.fit();
}
