// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <utility>

// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

static constexpr double LAPLACE_ALPHA = 0.5;  // Laplace smoothing parameter
// [D8] Standardized to "NA", matching every other categorical algorithm.
// The R wrapper (R/obc_jedi_mwoe.R) already converts NA to this token before
// the vector reaches here; kept for callers of the .Call() target itself.
static constexpr const char* MISSING_VALUE = "NA";

// Namespace for utility functions
namespace utils {
// Safe logarithm function to avoid -Inf
inline double safe_log(double x) {
  return x > EPSILON ? std::log(x) : std::log(EPSILON);
}

// Join vector of strings ensuring uniqueness
inline std::string join_categories(const std::vector<std::string>& categories,
                                   const std::string& separator) {
  if (categories.empty()) return "";
  if (categories.size() == 1) return categories[0];

  std::unordered_set<std::string> unique_cats;
  std::vector<std::string> unique_vec;
  unique_vec.reserve(categories.size());

  for (const auto& cat : categories) {
    if (unique_cats.insert(cat).second) {
      unique_vec.push_back(cat);
    }
  }

  size_t total_length = 0;
  for (const auto& cat : unique_vec) {
    total_length += cat.length();
  }
  total_length += separator.length() * (unique_vec.size() - 1);

  std::string result;
  result.reserve(total_length);

  result = unique_vec[0];
  for (size_t i = 1; i < unique_vec.size(); ++i) {
    result += separator;
    result += unique_vec[i];
  }

  return result;
}

// One-vs-rest M-WoE and IV of a class with Laplace smoothing. `other_total`
// and `total_other` are the summed counts of the remaining classes (in the
// bin and overall).
inline void mwoe_iv(int class_count, int total_class_count, int other_total,
                    int total_other, double alpha, double& woe, double& iv) {
  double class_rate = (class_count + alpha) / (total_class_count + alpha * 2);
  double other_rate = (other_total + alpha) / (total_other + alpha * 2);
  woe = safe_log(class_rate / other_rate);
  iv = (class_rate - other_rate) * woe;
}

// Jensen-Shannon divergence between the Laplace-smoothed class distributions
// of two bins. Same expressions, in the same order, as the allocation-based
// version it replaces, so the value is bit-identical.
inline double calculate_divergence(const std::vector<int>& bin1_counts,
                                   const std::vector<int>& bin2_counts) {
  const size_t k = bin1_counts.size();
  int bin1_total = std::accumulate(bin1_counts.begin(), bin1_counts.end(), 0);
  int bin2_total = std::accumulate(bin2_counts.begin(), bin2_counts.end(), 0);
  const double d1 = bin1_total + LAPLACE_ALPHA * static_cast<double>(k);
  const double d2 = bin2_total + LAPLACE_ALPHA * static_cast<double>(k);

  double div = 0.0;
  for (size_t i = 0; i < k; ++i) {
    const double p1 = (bin1_counts[i] + LAPLACE_ALPHA) / d1;
    const double p2 = (bin2_counts[i] + LAPLACE_ALPHA) / d2;
    const double m = (p1 + p2) / 2.0;
    if (p1 > EPSILON) {
      div += 0.5 * p1 * safe_log(p1 / m);
    }
    if (p2 > EPSILON) {
      div += 0.5 * p2 * safe_log(p2 / m);
    }
  }
  return div;
}
}  // namespace utils

// Enhanced structure for multinomial bin information
struct MultiCatBinInfo {
  std::unordered_set<std::string> category_set;  // For uniqueness check
  std::vector<std::string> categories;           // For ordered storage
  int total_count;
  std::vector<int> class_counts;
  std::vector<double> woes;
  std::vector<double> ivs;
  std::vector<double> class_rates;  // Cache for class rates

  MultiCatBinInfo() : total_count(0) {}

  explicit MultiCatBinInfo(size_t n_classes)
    : total_count(0),
      class_counts(n_classes, 0),
      woes(n_classes, 0.0),
      ivs(n_classes, 0.0),
      class_rates(n_classes, 0.0) {}

  // Add a category ensuring uniqueness
  inline void add_category(const std::string& cat) {
    if (category_set.insert(cat).second) {
      categories.push_back(cat);
    }
  }

  // Merge with another bin ensuring uniqueness of categories
  inline void merge_with(const MultiCatBinInfo& other) {
    for (const auto& cat : other.categories) {
      add_category(cat);
    }
    total_count += other.total_count;
    for (size_t i = 0; i < class_counts.size(); ++i) {
      class_counts[i] += other.class_counts[i];
    }
    update_class_rates();
  }

  inline void update_class_rates() {
    if (total_count > 0) {
      for (size_t i = 0; i < class_counts.size(); ++i) {
        class_rates[i] = static_cast<double>(class_counts[i]) / total_count;
      }
    }
  }

  // Compute M-WoE and IV (one class against the pooled others) with Laplace
  // smoothing. The pooled counts are sums of int counts, so taking them as
  // (total - own) is exact and replaces the per-class copy of the others.
  inline void calculate_metrics(const std::vector<int>& total_class_counts,
                                int grand_total) {
    for (size_t c = 0; c < class_counts.size(); ++c) {
      utils::mwoe_iv(class_counts[c], total_class_counts[c],
                     total_count - class_counts[c],
                     grand_total - total_class_counts[c], LAPLACE_ALPHA,
                     woes[c], ivs[c]);
    }
  }
};

// Main class for multinomial categorical binning
class OBC_JEDIMWoE {
private:
  std::vector<std::string> feature_;
  std::vector<int> target_;
  size_t n_classes_;
  int min_bins_;
  int max_bins_;
  double bin_cutoff_;
  int max_n_prebins_;
  std::string bin_separator_;
  double convergence_threshold_;
  int max_iterations_;

  std::vector<MultiCatBinInfo> bins_;
  std::vector<int> total_class_counts_;
  int grand_total_;
  bool converged_;
  int iterations_run_;
  bool use_divergence_;  // Flag to toggle between IV and divergence-based merging

  // Per-category statistics, built once (same reserve and insertion
  // sequence as before, so the iteration order is unchanged).
  std::unordered_map<std::string, MultiCatBinInfo> bin_map_;
  int ncat_;

  // Advanced input validation
  void validate_inputs() {
    if (feature_.empty() || feature_.size() != target_.size()) {
      throw std::invalid_argument("Feature and target vectors must have the same non-empty length");
    }

    int max_class = -1;
    std::unordered_set<int> class_set;

    for (int t : target_) {
      if (t < 0) {
        throw std::invalid_argument("Target values must be non-negative integers");
      }
      max_class = std::max(max_class, t);
      class_set.insert(t);
    }

    n_classes_ = static_cast<size_t>(max_class) + 1;

    if (class_set.size() < 2) {
      throw std::invalid_argument("Target must have at least 2 distinct classes");
    }

    // Ensure all classes from 0 to max_class are present
    for (int i = 0; i < static_cast<int>(n_classes_); ++i) {
      if (class_set.find(i) == class_set.end()) {
        throw std::invalid_argument("Target classes must be consecutive integers starting from 0");
      }
    }

    if (min_bins_ < 1) {
      throw std::invalid_argument("min_bins must be at least 1");
    }
    if (max_bins_ < min_bins_) {
      throw std::invalid_argument("max_bins must be greater than or equal to min_bins");
    }
    if (bin_cutoff_ <= 0 || bin_cutoff_ >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive)");
    }
    if (max_n_prebins_ < min_bins_) {
      throw std::invalid_argument("max_n_prebins must be at least min_bins");
    }
  }

  // Single pass over the data: one hash lookup per row, class rates computed
  // once per category instead of after every observation.
  void count_categories() {
    size_t est_cats = std::min(feature_.size() / 4, static_cast<size_t>(1024));
    bin_map_.reserve(est_cats);
    total_class_counts_.assign(n_classes_, 0);

    for (size_t i = 0; i < feature_.size(); ++i) {
      const std::string& cat = feature_[i];
      const int class_idx = target_[i];

      auto it = bin_map_.find(cat);
      if (it == bin_map_.end()) {
        it = bin_map_.emplace(cat, MultiCatBinInfo(n_classes_)).first;
        it->second.add_category(cat);
      }
      MultiCatBinInfo& bin = it->second;
      bin.total_count++;
      bin.class_counts[static_cast<size_t>(class_idx)]++;
      total_class_counts_[static_cast<size_t>(class_idx)]++;
    }
    for (auto& kv : bin_map_) {
      kv.second.update_class_rates();
    }
    grand_total_ = std::accumulate(total_class_counts_.begin(),
                                   total_class_counts_.end(), 0);
    ncat_ = static_cast<int>(bin_map_.size());
  }

  void initial_binning() {
    bins_.clear();
    bins_.reserve(bin_map_.size());
    for (auto& kv : bin_map_) {
      bins_.push_back(std::move(kv.second));
    }
    bin_map_.clear();
  }

  // Enhanced merging of low frequency categories
  void merge_low_freq() {
    int total_count = std::accumulate(bins_.begin(), bins_.end(), 0,
                                      [](int sum, const MultiCatBinInfo& bin) {
                                        return sum + bin.total_count;
                                      });
    double cutoff_count = total_count * bin_cutoff_;

    // Sort bins by count (ascending)
    std::sort(bins_.begin(), bins_.end(),
              [](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                return a.total_count < b.total_count;
              });

    std::vector<MultiCatBinInfo> new_bins;
    new_bins.reserve(bins_.size());

    MultiCatBinInfo rare_bin(n_classes_);
    bool has_rare = false;

    // Up to min_bins rare categories stay separate (as they always have); the
    // rest are pooled. They are now the most frequent of the rare ones (the
    // last of the rare prefix of the ascending order); the previous test kept
    // the min_bins rarest instead.
    size_t n_rare = 0;
    for (const auto& bin : bins_) {
      if (bin.total_count < cutoff_count) ++n_rare;
    }
    const size_t keep_rare =
      std::min(n_rare, static_cast<size_t>(std::max(min_bins_, 0)));

    for (size_t pos = 0; pos < bins_.size(); ++pos) {
      auto& bin = bins_[pos];
      if (bin.total_count >= cutoff_count || pos >= n_rare - keep_rare) {
        new_bins.push_back(std::move(bin));
      } else {
        rare_bin.merge_with(bin);
        has_rare = true;
      }
    }

    if (has_rare && rare_bin.total_count > 0) {
      new_bins.push_back(std::move(rare_bin));
    }

    bins_ = std::move(new_bins);
  }

  // Class-specific IV, summed in bin order
  double calculate_class_iv(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    double iv = 0.0;
    for (const auto& bin : current_bins) {
      iv += bin.ivs[class_idx];
    }
    return iv;
  }

  void compute_metrics() {
    for (auto& bin : bins_) {
      bin.calculate_metrics(total_class_counts_, grand_total_);
    }
  }

  bool is_monotonic_for_class(const std::vector<MultiCatBinInfo>& current_bins, size_t class_idx) const {
    if (current_bins.size() <= 2) return true;

    bool should_increase = true;
    bool should_decrease = true;

    for (size_t i = 1; i < current_bins.size(); ++i) {
      if (current_bins[i].woes[class_idx] < current_bins[i-1].woes[class_idx] - EPSILON) {
        should_increase = false;
      }
      if (current_bins[i].woes[class_idx] > current_bins[i-1].woes[class_idx] + EPSILON) {
        should_decrease = false;
      }
      if (!should_increase && !should_decrease) {
        return false;
      }
    }
    return true;
  }

  bool is_monotonic(const std::vector<MultiCatBinInfo>& current_bins) const {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      if (!is_monotonic_for_class(current_bins, class_idx)) {
        return false;
      }
    }
    return true;
  }

  // Sum of the per-class IV losses of merging bins i and i+1, accumulated
  // exactly as the version that built a merged copy of the whole bin vector:
  // for each class, the total over the bins in order with the merged bin in
  // place of the pair, subtracted from the current total.
  double merge_loss(size_t i, const std::vector<double>& original_ivs) const {
    MultiCatBinInfo merged(n_classes_);
    merged.total_count = bins_[i].total_count + bins_[i + 1].total_count;
    for (size_t c = 0; c < n_classes_; ++c) {
      merged.class_counts[c] = bins_[i].class_counts[c] + bins_[i + 1].class_counts[c];
    }
    merged.calculate_metrics(total_class_counts_, grand_total_);

    double total_iv_loss = 0.0;
    for (size_t c = 0; c < n_classes_; ++c) {
      double new_iv = 0.0;
      for (size_t j = 0; j < bins_.size(); ++j) {
        if (j == i) {
          new_iv += merged.ivs[c];
        } else if (j != i + 1) {
          new_iv += bins_[j].ivs[c];
        }
      }
      total_iv_loss += original_ivs[c] - new_iv;
    }
    return total_iv_loss;
  }

  std::vector<double> class_iv_totals() const {
    std::vector<double> ivs(n_classes_);
    for (size_t c = 0; c < n_classes_; ++c) {
      ivs[c] = calculate_class_iv(bins_, c);
    }
    return ivs;
  }

  // Main optimization algorithm
  void optimize() {
    std::vector<double> prev_ivs = class_iv_totals();

    converged_ = false;
    iterations_run_ = 0;

    while (iterations_run_ < max_iterations_) {
      if (is_monotonic(bins_) &&
          static_cast<int>(bins_.size()) <= max_bins_ &&
          static_cast<int>(bins_.size()) >= min_bins_) {
        converged_ = true;
        break;
      }

      if (static_cast<int>(bins_.size()) > min_bins_) {
        if (static_cast<int>(bins_.size()) > max_bins_) {
          if (use_divergence_) {
            merge_most_similar_bins();
          } else {
            merge_adjacent_bins();
          }
          use_divergence_ = !use_divergence_;
        } else {
          improve_monotonicity();
        }
      } else {
        break;
      }

      std::vector<double> current_ivs(n_classes_);
      bool all_converged = true;

      for (size_t i = 0; i < n_classes_; ++i) {
        current_ivs[i] = calculate_class_iv(bins_, i);
        if (std::abs(current_ivs[i] - prev_ivs[i]) >= convergence_threshold_) {
          all_converged = false;
        }
      }

      if (all_converged) {
        converged_ = true;
        break;
      }

      prev_ivs = std::move(current_ivs);
      iterations_run_++;
    }

    // Final adjustments to meet max_bins
    while (static_cast<int>(bins_.size()) > max_bins_ && bins_.size() >= 2) {
      merge_adjacent_bins();
    }

    ensure_monotonic_order();
    compute_metrics();
  }

  // Weighted pair score used by merge_most_similar_bins(): the divergence,
  // with a 5% discount for neighbouring bins.
  double pair_score(size_t i, size_t j, bool adjacent) const {
    double div = utils::calculate_divergence(bins_[i].class_counts, bins_[j].class_counts);
    if (adjacent) {
      div *= 0.95;  // Small bias towards adjacent bins
    }
    return div;
  }

  // Find and merge statistically most similar bins: the first pair (i < j),
  // in index order, with the smallest weighted divergence.
  void merge_most_similar_bins() {
    double min_divergence = std::numeric_limits<double>::max();
    size_t merge_idx1 = 0;
    size_t merge_idx2 = 0;

    for (size_t i = 0; i < bins_.size(); ++i) {
      for (size_t j = i + 1; j < bins_.size(); ++j) {
        double div = pair_score(i, j, j == i + 1);
        if (div < min_divergence) {
          min_divergence = div;
          merge_idx1 = i;
          merge_idx2 = j;
        }
      }
    }

    merge_bins(merge_idx1, merge_idx2);
  }

  // Pre-binning reduction: repeat merge_most_similar_bins() while there are
  // more than `target` bins. Scanning all pairs after every merge cost
  // O(B^2) divergences per merge -- O(B^3) overall, ~20 s for 2000
  // categories. Divergences depend only on the two bins, so each bin keeps
  // its best partner among the bins after it; after a merge only the rows
  // that referenced the merged or removed bin, the merged bin's own row and
  // the row whose neighbour changed are rescanned. Bins keep their relative
  // order (a merge keeps the lower slot), so comparing slots is comparing
  // positions and the first minimal pair in index order is the one chosen.
  void reduce_prebins_by_divergence(size_t target) {
    const size_t nb = bins_.size();
    if (nb <= target || nb < 2) return;
    const size_t NONE = std::numeric_limits<size_t>::max();
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<size_t> nxt(nb), prv(nb);
    for (size_t i = 0; i < nb; ++i) {
      nxt[i] = (i + 1 < nb) ? i + 1 : NONE;
      prv[i] = (i > 0) ? i - 1 : NONE;
    }
    std::vector<char> alive(nb, 1);
    std::vector<double> best_d(nb, INF);
    std::vector<size_t> best_j(nb, NONE);

    auto recompute_row = [&](size_t i) {
      best_d[i] = INF;
      best_j[i] = NONE;
      for (size_t j = nxt[i]; j != NONE; j = nxt[j]) {
        const double d = pair_score(i, j, j == nxt[i]);
        if (d < best_d[i]) {
          best_d[i] = d;
          best_j[i] = j;
        }
      }
    };

    for (size_t i = 0; i < nb; ++i) recompute_row(i);

    size_t head = 0;
    size_t count = nb;
    while (count > target && count >= 2) {
      // First row, in slot order, holding the smallest score.
      double bd = std::numeric_limits<double>::max();
      size_t a = NONE;
      for (size_t i = head; i != NONE; i = nxt[i]) {
        if (best_d[i] < bd) {
          bd = best_d[i];
          a = i;
        }
      }
      if (a == NONE) break;
      const size_t b = best_j[a];

      bins_[a].merge_with(bins_[b]);
      alive[b] = 0;
      const size_t p = prv[b];
      const size_t q = nxt[b];
      if (p != NONE) nxt[p] = q;
      if (q != NONE) prv[q] = p;
      if (b == head) head = q;
      --count;

      recompute_row(a);
      if (p != NONE && p != a) recompute_row(p);
      for (size_t i = head; i != NONE; i = nxt[i]) {
        if (i == a || i == p) continue;
        if (best_j[i] == a || best_j[i] == b) {
          recompute_row(i);
        } else if (i < a) {
          const double d = pair_score(i, a, a == nxt[i]);
          if (d < best_d[i] || (d == best_d[i] && a < best_j[i])) {
            best_d[i] = d;
            best_j[i] = a;
          }
        }
      }
      Rcpp::checkUserInterrupt();
    }

    std::vector<MultiCatBinInfo> kept;
    kept.reserve(count);
    for (size_t i = 0; i < nb; ++i) {
      if (alive[i]) kept.push_back(std::move(bins_[i]));
    }
    bins_ = std::move(kept);
    compute_metrics();
  }

  // Optimized merging of adjacent bins based on IV loss
  void merge_adjacent_bins() {
    // `<= 2` made this a no-op on exactly two bins, so with max_bins = 1 the
    // final `while (bins_.size() > max_bins_)` loop never terminated.
    if (bins_.size() < 2) return;

    double min_total_iv_loss = std::numeric_limits<double>::max();
    size_t best_merge_idx = 0;

    const std::vector<double> original_ivs = class_iv_totals();

    for (size_t i = 0; i + 1 < bins_.size(); ++i) {
      const double total_iv_loss = merge_loss(i, original_ivs);
      if (total_iv_loss < min_total_iv_loss) {
        min_total_iv_loss = total_iv_loss;
        best_merge_idx = i;
      }
    }

    merge_bins(best_merge_idx, best_merge_idx + 1);
  }

  // Merge bin idx2 into idx1. Each bin's metrics depend only on its own
  // counts and the fixed totals, so only the merged bin is recomputed.
  void merge_bins(size_t idx1, size_t idx2) {
    if (idx1 >= bins_.size() || idx2 >= bins_.size() || idx1 == idx2) return;
    if (idx2 < idx1) std::swap(idx1, idx2);

    bins_[idx1].merge_with(bins_[idx2]);
    bins_.erase(bins_.begin() + static_cast<std::ptrdiff_t>(idx2));
    bins_[idx1].calculate_metrics(total_class_counts_, grand_total_);
  }

  // Improved algorithm for monotonicity correction
  void improve_monotonicity() {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      double max_violation = 0.0;
      size_t violation_idx = 0;
      bool found_violation = false;

      for (size_t i = 1; i < bins_.size(); ++i) {
        double curr_violation = 0.0;

        bool is_peak = (i + 1 < bins_.size() &&
                        bins_[i].woes[class_idx] > bins_[i-1].woes[class_idx] + EPSILON &&
                        bins_[i].woes[class_idx] > bins_[i+1].woes[class_idx] + EPSILON);

        bool is_valley = (i + 1 < bins_.size() &&
                          bins_[i].woes[class_idx] < bins_[i-1].woes[class_idx] - EPSILON &&
                          bins_[i].woes[class_idx] < bins_[i+1].woes[class_idx] - EPSILON);

        if (is_peak) {
          curr_violation = std::max(bins_[i].woes[class_idx] - bins_[i-1].woes[class_idx],
                                    bins_[i].woes[class_idx] - bins_[i+1].woes[class_idx]);
        } else if (is_valley) {
          curr_violation = std::max(bins_[i-1].woes[class_idx] - bins_[i].woes[class_idx],
                                    bins_[i+1].woes[class_idx] - bins_[i].woes[class_idx]);
        }

        if (curr_violation > max_violation) {
          max_violation = curr_violation;
          violation_idx = i;
          found_violation = true;
        }
      }

      if (found_violation) {
        const std::vector<double> original_ivs = class_iv_totals();

        // violation_idx is always >= 1 and has a successor (peaks and
        // valleys are interior bins).
        const double loss_prev = merge_loss(violation_idx - 1, original_ivs);
        const double loss_next = merge_loss(violation_idx, original_ivs);

        if (loss_prev <= loss_next) {
          merge_bins(violation_idx - 1, violation_idx);
        } else {
          merge_bins(violation_idx, violation_idx + 1);
        }

        break;
      }
    }
  }

  // Ensure monotonic ordering of bins
  void ensure_monotonic_order() {
    for (size_t class_idx = 0; class_idx < n_classes_; ++class_idx) {
      if (!is_monotonic_for_class(bins_, class_idx)) {
        std::stable_sort(bins_.begin(), bins_.end(),
                         [class_idx](const MultiCatBinInfo& a, const MultiCatBinInfo& b) {
                           return a.woes[class_idx] < b.woes[class_idx];
                         });
        compute_metrics();
      }
    }
  }

public:
  OBC_JEDIMWoE(
    std::vector<std::string> feature,
    std::vector<int> target,
    int min_bins = 3,
    int max_bins = 5,
    double bin_cutoff = 0.05,
    int max_n_prebins = 20,
    std::string bin_separator = "%;%",
    double convergence_threshold = 1e-6,
    int max_iterations = 1000
  ) : feature_(std::move(feature)),
  target_(std::move(target)),
  n_classes_(0),  // Will be set in validate_inputs
  min_bins_(min_bins),
  max_bins_(max_bins),
  bin_cutoff_(bin_cutoff),
  max_n_prebins_(max_n_prebins),
  bin_separator_(std::move(bin_separator)),
  convergence_threshold_(convergence_threshold),
  max_iterations_(max_iterations),
  grand_total_(0),
  converged_(false),
  iterations_run_(0),
  use_divergence_(true),  // Start with divergence-based merging
  ncat_(0)
  {
    validate_inputs();

    count_categories();
    const int ncat = ncat_;

    // Cap max_bins at number of unique categories
    max_bins_ = std::min(max_bins_, ncat);

    // Ensure min_bins is valid
    min_bins_ = std::min(min_bins_, max_bins_);

    // Ensure max_n_prebins is sufficient
    if (max_n_prebins_ < min_bins_) {
      max_n_prebins_ = min_bins_;
    }
  }

  void fit() {
    if (ncat_ <= 2) {
      // Trivial case: <= 2 categories
      initial_binning();
      compute_metrics();
      converged_ = true;
      iterations_run_ = 0;
      return;
    }

    initial_binning();
    merge_low_freq();
    compute_metrics();

    // Reduce number of pre-bins if needed
    reduce_prebins_by_divergence(static_cast<size_t>(std::max(max_n_prebins_, 1)));

    optimize();
  }

  Rcpp::List get_results() const {
    const size_t n_bins = bins_.size();
    const int nb = static_cast<int>(n_bins);
    const int nc = static_cast<int>(n_classes_);

    CharacterVector bin_names(nb);
    NumericMatrix woes(nb, nc);
    NumericMatrix ivs(nb, nc);
    IntegerVector counts(nb);
    IntegerMatrix class_counts(nb, nc);
    NumericMatrix class_rates(nb, nc);
    NumericVector ids(nb);
    NumericVector total_ivs(nc);

    for (int i = 0; i < nb; ++i) {
      const MultiCatBinInfo& bin = bins_[static_cast<size_t>(i)];
      bin_names[i] = utils::join_categories(bin.categories, bin_separator_);
      counts[i] = bin.total_count;
      ids[i] = i + 1;

      for (int j = 0; j < nc; ++j) {
        const size_t c = static_cast<size_t>(j);
        woes(i, j) = bin.woes[c];
        ivs(i, j) = bin.ivs[c];
        class_counts(i, j) = bin.class_counts[c];
        class_rates(i, j) = bin.class_rates[c];
        total_ivs[j] += std::fabs(bin.ivs[c]);
      }
    }

    return Rcpp::List::create(
      Named("id") = ids,
      Named("bin") = bin_names,
      Named("woe") = woes,
      Named("iv") = ivs,
      Named("count") = counts,
      Named("class_counts") = class_counts,
      Named("class_rates") = class_rates,
      Named("converged") = converged_,
      Named("iterations") = iterations_run_,
      Named("n_classes") = nc,
      Named("total_iv") = total_ivs
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_jedi_mwoe(
   Rcpp::IntegerVector target,
   Rcpp::StringVector feature,
   int min_bins = 3,
   int max_bins = 5,
   double bin_cutoff = 0.05,
   int max_n_prebins = 20,
   std::string bin_separator = "%;%",
   double convergence_threshold = 1e-6,
   int max_iterations = 1000
) {
 try {
   std::vector<std::string> feature_vec;
   feature_vec.reserve(static_cast<size_t>(feature.size()));

   for (R_xlen_t i = 0; i < feature.size(); ++i) {
     SEXP s = STRING_ELT(feature, i);
     if (s == NA_STRING) {
       feature_vec.emplace_back(MISSING_VALUE);
     } else {
       feature_vec.emplace_back(CHAR(s));
     }
   }

   std::vector<int> target_vec;
   target_vec.reserve(static_cast<size_t>(target.size()));

   for (R_xlen_t i = 0; i < target.size(); ++i) {
     if (IntegerVector::is_na(target[i])) {
       Rcpp::stop("Target cannot contain missing values");
     }
     target_vec.push_back(target[i]);
   }

   OBC_JEDIMWoE jedi(
       std::move(feature_vec), std::move(target_vec),
       min_bins, max_bins,
       bin_cutoff, max_n_prebins,
       bin_separator, convergence_threshold,
       max_iterations
   );
   jedi.fit();
   return jedi.get_results();
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal_binning_categorical_jedi_mwoe: " + std::string(e.what()));
 }
}
