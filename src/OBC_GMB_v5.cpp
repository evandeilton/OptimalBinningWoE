// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>


// Include shared headers
#include "common/optimal_binning_common.h"
#include "common/bin_structures.h"

using namespace Rcpp;
using namespace OptimalBinning;

namespace {

// Category names that contain bin_separator. A bin label is its categories
// joined with the separator, and obwoe_apply() recovers the categories by
// splitting the label on it, so such a name is cut into pieces that are not
// categories (and a piece equal to another category claims it too). The
// binning itself is unaffected; the caller is warned that its labels are
// ambiguous. Checked over the distinct categories only, so the cost is O(k).
template <typename T>
inline const std::string& separator_key(const std::pair<const std::string, T>& kv) {
  return kv.first;
}

template <typename Container>
std::size_t count_separator_hits(const Container& categories, const std::string& sep,
                                 std::string& example) {
  std::size_t hits = 0;
  if (sep.empty()) return 0;
  for (const auto& item : categories) {
    const std::string& cat = separator_key(item);
    if (cat.find(sep) != std::string::npos) {
      if (hits == 0) example = cat;
      ++hits;
    }
  }
  return hits;
}

inline void warn_separator_hits(const std::string& sep, std::size_t hits,
                                const std::string& example) {
  if (hits == 0) return;
  Rcpp::warning("bin_separator \"%s\" occurs inside %d categor%s of the feature "
                "(e.g. \"%s\"): the bin labels are ambiguous and cannot be split "
                "back into categories. Choose a bin_separator that does not occur "
                "in the category names.",
                sep, hits, (hits == 1 ? "y" : "ies"), example);
}

} // namespace

namespace {
constexpr double NEG_INFINITY = -std::numeric_limits<double>::infinity();
}

class OBC_GMB {
private:
  const std::vector<std::string>& feature;
  const std::vector<int>& target;
  int min_bins;
  int max_bins;
  double bin_cutoff;
  int max_n_prebins;
  std::string bin_separator;
  double convergence_threshold;
  int max_iterations;

  std::vector<CategoricalBin> bins;
  int total_pos = 0;
  int total_neg = 0;
  bool converged = false;
  int iterations_run = 0;
  std::size_t sep_hits = 0;   // categories whose name contains bin_separator
  std::string sep_example;

  // Enhanced input validation with more comprehensive checks
  void validateInput() const {
    if (feature.size() != target.size()) {
      throw std::invalid_argument("Feature and target must have the same length.");
    }
    if (feature.empty()) {
      throw std::invalid_argument("Feature cannot be empty.");
    }
    if (min_bins < 2) {
      throw std::invalid_argument("min_bins must be >= 2.");
    }
    if (max_bins < min_bins) {
      throw std::invalid_argument("max_bins must be >= min_bins.");
    }
    if (bin_cutoff <= 0 || bin_cutoff >= 1) {
      throw std::invalid_argument("bin_cutoff must be between 0 and 1 (exclusive).");
    }
    if (max_n_prebins < min_bins) {
      throw std::invalid_argument("max_n_prebins must be >= min_bins.");
    }

    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(), [](const std::string& s) {
      return s.empty();
    })) {
      throw std::invalid_argument("Feature cannot contain empty strings. Consider preprocessing your data.");
    }

    // Binary target. Every value is checked: stopping at the first 0 and 1
    // let a later 2 (or -1) through, to be counted as a negative.
    bool has_zero = false;
    bool has_one = false;
    for (int t : target) {
      if (t == 0) has_zero = true;
      else if (t == 1) has_one = true;
      else throw std::invalid_argument("Target must be binary (0 or 1).");
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }
  }

  // IV contribution of one bin as the total-IV sum counts it: the bin's own
  // (Bayesian-smoothed) IV when it is non-zero and finite, otherwise the same
  // quantity recomputed from its counts, skipped when not finite.
  double ivContribution(const CategoricalBin& bin) const {
    if (bin.iv != 0.0 && std::isfinite(bin.iv)) {
      return bin.iv;
    }
    double prior_pos = BAYESIAN_PRIOR_STRENGTH * static_cast<double>(total_pos) /
      (total_pos + total_neg);
    double prior_neg = BAYESIAN_PRIOR_STRENGTH - prior_pos;

    double pos_rate = static_cast<double>(bin.count_pos + prior_pos) /
      static_cast<double>(total_pos + BAYESIAN_PRIOR_STRENGTH);
    double neg_rate = static_cast<double>(bin.count_neg + prior_neg) /
      static_cast<double>(total_neg + BAYESIAN_PRIOR_STRENGTH);

    double woe = std::log(pos_rate / neg_rate);
    double local_iv = (pos_rate - neg_rate) * woe;
    return std::isfinite(local_iv) ? local_iv : 0.0;
  }

  // Total IV of the current bins, summed left to right.
  double calculateIV() const {
    double iv = 0.0;
    for (const auto& bin : bins) {
      iv += ivContribution(bin);
    }
    return iv;
  }

  // Total IV the bins would have after merging bins[i] and bins[i + 1].
  //
  // Computed from the counts, in the same left-to-right order as the total of
  // the merged configuration, so the value is bit-identical to building that
  // configuration. It used to be built literally -- a full copy of the bin
  // vector, category-name strings included, for every candidate pair -- and,
  // above 10 bins, "cached" by pair position. That cache was wrong: a pair's
  // score is the total IV of the whole configuration, so every merge
  // invalidates all of them, and after the erase the cached positions pointed
  // at different pairs. Stale scores then decided which pair was merged.
  double mergeScore(const std::vector<double>& prefix, size_t i) const {
    CategoricalBin merged;
    merged.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
    merged.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
    merged.update_count();
    merged.calculate_metrics(total_pos, total_neg);

    double iv = prefix[i];
    iv += ivContribution(merged);
    for (size_t j = i + 2; j < bins.size(); ++j) {
      iv += ivContribution(bins[j]);
    }
    return iv;
  }

  // prefix[i] = running total of the contributions of bins[0 .. i-1]
  std::vector<double> contributionPrefix() const {
    std::vector<double> prefix(bins.size() + 1, 0.0);
    double s = 0.0;
    for (size_t j = 0; j < bins.size(); ++j) {
      prefix[j] = s;
      s += ivContribution(bins[j]);
    }
    prefix[bins.size()] = s;
    return prefix;
  }

  // Merge rare bins (frequency < bin_cutoff) with their neighbours in
  // event-rate order. With close_at_cutoff == false a run of consecutive rare
  // bins is pooled into one bin; with true, a pool is closed as soon as it
  // reaches bin_cutoff.
  std::vector<CategoricalBin> mergeRare(std::vector<CategoricalBin>& src,
                                        int total_count,
                                        bool close_at_cutoff) const {
    std::vector<CategoricalBin> merged_bins;
    merged_bins.reserve(src.size());

    CategoricalBin current_rare_bin;
    bool has_rare_bin = false;

    for (auto& bin : src) {
      double freq = static_cast<double>(bin.count) / static_cast<double>(total_count);

      if (freq < bin_cutoff) {
        current_rare_bin.merge_with(bin);
        has_rare_bin = true;
        if (close_at_cutoff &&
            static_cast<double>(current_rare_bin.count) / static_cast<double>(total_count) >= bin_cutoff) {
          merged_bins.push_back(std::move(current_rare_bin));
          current_rare_bin = CategoricalBin();
          has_rare_bin = false;
        }
      } else {
        if (has_rare_bin) {
          merged_bins.push_back(std::move(current_rare_bin));
          current_rare_bin = CategoricalBin();
          has_rare_bin = false;
        }
        merged_bins.push_back(bin);
      }
    }

    if (has_rare_bin) {
      merged_bins.push_back(std::move(current_rare_bin));
    }
    return merged_bins;
  }

  // Enhanced bin initialization with optimized counting
  void initializeBins() {
    // Efficient single-pass counting
    std::unordered_map<std::string, std::pair<int, int>> category_stats;
    category_stats.reserve(std::min(static_cast<size_t>(feature.size() / 4), static_cast<size_t>(1024)));

    total_pos = 0;
    total_neg = 0;

    for (size_t i = 0; i < feature.size(); ++i) {
      auto& stats = category_stats[feature[i]];

      if (target[i] == 1) {
        stats.first++;  // pos_count
        total_pos++;
      } else {
        stats.second++;  // neg_count
        total_neg++;
      }
    }

    sep_hits = count_separator_hits(category_stats, bin_separator, sep_example);

    // The bin limits cannot exceed the number of categories. (This used to
    // need a second hash pass over every row, in the constructor.)
    const int ncat = static_cast<int>(category_stats.size());
    max_bins = std::min(max_bins, ncat);
    min_bins = std::min(min_bins, max_bins);

    // Check for extremely imbalanced datasets
    if (total_pos < 5 || total_neg < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results may be unstable.");
    }

    bins.clear();
    bins.reserve(category_stats.size());

    for (const auto& [cat, stats] : category_stats) {
      CategoricalBin bin;
      bin.categories.push_back(cat);
      bin.count_pos = stats.first;
      bin.count_neg = stats.second;
      bin.update_count();
      bins.push_back(std::move(bin));
    }

    // Sort by positive rate for consistent ordering
    std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
      return a.event_rate() < b.event_rate();
    });

    // Rare category handling
    int total_count = std::accumulate(bins.begin(), bins.end(), 0,
                                      [](int sum, const CategoricalBin& bin) { return sum + bin.count; });

    std::vector<CategoricalBin> merged_bins = mergeRare(bins, total_count, false);
    if (static_cast<int>(merged_bins.size()) < min_bins) {
      // Pooling each run of rare categories into one bin left fewer than
      // min_bins bins. On a high-cardinality feature whose levels are all
      // below bin_cutoff (500 levels at 0.2% each) every level was pooled
      // and the whole sample came back as a single bin with IV = 0. Close
      // each pool once it reaches bin_cutoff instead, which keeps adjacent
      // (similar event rate) categories together and every pool but the last
      // above the cutoff.
      merged_bins = mergeRare(bins, total_count, true);
      if (static_cast<int>(merged_bins.size()) < min_bins) {
        // Even the pools are too few (a very large bin_cutoff): keep one bin
        // per category and let the greedy merge do the grouping.
        merged_bins = bins;
      }
    }

    bins = std::move(merged_bins);

    // Limit number of pre-bins if necessary.
    //
    // The excess bins are folded into the smallest of the retained ones (they
    // used to be dropped, silently losing their observations). Which
    // categories are kept as separate identities: the max_n_prebins largest.
    if (static_cast<int>(bins.size()) > max_n_prebins) {
      // validateInput() enforces min_bins >= 2 and max_n_prebins >= min_bins,
      // so keep is at least 2 and the absorber index below is always in range.
      const size_t keep = static_cast<size_t>(max_n_prebins);
      const size_t n_bins = bins.size();

      std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
        return a.count > b.count;  // Descending order
      });

      CategoricalBin& absorber = bins[keep - 1];
      for (size_t i = keep; i < n_bins; ++i) {
        absorber.merge_with(bins[i]);
      }

      bins.resize(keep);

      std::sort(bins.begin(), bins.end(), [](const CategoricalBin& a, const CategoricalBin& b) {
        return a.event_rate() < b.event_rate();
      });
    }

    // Calculate metrics for all bins
    for (auto& bin : bins) {
      bin.calculate_metrics(total_pos, total_neg);
    }
  }

  // Greedy merge: repeatedly merge the adjacent pair whose merge leaves the
  // highest total IV.
  void greedyMerge() {
    // Early exit if we already have few enough bins
    if (static_cast<int>(bins.size()) <= max_bins) {
      converged = true;
      return;
    }

    double prev_iv = calculateIV();
    double current_iv = prev_iv;

    while (static_cast<int>(bins.size()) > min_bins && iterations_run < max_iterations) {
      double best_merge_score = NEG_INFINITY;
      double second_best_score = NEG_INFINITY;
      size_t best_merge_index = 0;
      size_t second_best_index = 0;

      const std::vector<double> prefix = contributionPrefix();

      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double merge_score = mergeScore(prefix, i);

        // Early exit if we find an excellent merge (5% improvement)
        if (merge_score > current_iv * 1.05 && std::isfinite(merge_score)) {
          best_merge_score = merge_score;
          best_merge_index = i;
          break;
        }

        // Track best and second best options
        if (merge_score > best_merge_score && std::isfinite(merge_score)) {
          second_best_score = best_merge_score;
          second_best_index = best_merge_index;
          best_merge_score = merge_score;
          best_merge_index = i;
        } else if (merge_score > second_best_score && std::isfinite(merge_score)) {
          second_best_score = merge_score;
          second_best_index = i;
        }
      }

      // Tie handling: If best and second best are very close, prefer more balanced bins
      if (std::abs(best_merge_score - second_best_score) < convergence_threshold * 10) {
        int size_diff_best = std::abs(bins[best_merge_index].count - bins[best_merge_index + 1].count);
        int size_diff_second = std::abs(bins[second_best_index].count - bins[second_best_index + 1].count);

        if (size_diff_second < size_diff_best * 0.8) {  // Second option is significantly more balanced
          best_merge_index = second_best_index;
          best_merge_score = second_best_score;
        }
      }

      // Execute the best merge
      CategoricalBin& bin1 = bins[best_merge_index];
      CategoricalBin& bin2 = bins[best_merge_index + 1];

      bin1.merge_with(bin2);
      bin1.calculate_metrics(total_pos, total_neg);

      bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(best_merge_index) + 1);

      // Recalculate IV after merging
      current_iv = calculateIV();

      // Check convergence
      if (std::fabs(current_iv - prev_iv) < convergence_threshold) {
        converged = true;
        break;
      }

      prev_iv = current_iv;
      iterations_run++;

      // Stop if we've reached max_bins
      if (static_cast<int>(bins.size()) <= max_bins) {
        break;
      }
    }

    // Reaching the bin-count target (or min_bins) is a valid stopping state,
    // just like meeting the IV tolerance above. Only exhausting max_iterations
    // leaves converged == false.
    converged = converged || (static_cast<int>(bins.size()) <= max_bins);
  }

  // Enforce max_bins as a hard post-condition.
  //
  // greedyMerge() can stop on its IV-change tolerance before the bin count
  // reaches max_bins. max_bins is a documented user-facing parameter, so we
  // keep merging by the algorithm's own criterion -- the adjacent pair whose
  // merge leaves the highest total IV -- until the cap is met. min_bins still
  // wins: we never merge below it.
  void enforceMaxBins() {
    while (static_cast<int>(bins.size()) > max_bins &&
           static_cast<int>(bins.size()) > min_bins &&
           bins.size() > 1) {
      double best_score = NEG_INFINITY;
      size_t best_index = 0;

      const std::vector<double> prefix = contributionPrefix();
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        double score = mergeScore(prefix, i);
        if (std::isfinite(score) && score > best_score) {
          best_score = score;
          best_index = i;
        }
      }

      bins[best_index].merge_with(bins[best_index + 1]);
      bins[best_index].calculate_metrics(total_pos, total_neg);
      bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
      iterations_run++;
    }
  }

  // Mean absolute WoE gap between neighbouring bins
  double averageWoeGap() const {
    double total_gap = 0.0;
    for (size_t i = 1; i < bins.size(); i++) {
      total_gap += std::abs(bins[i].woe - bins[i-1].woe);
    }
    return total_gap / static_cast<double>(bins.size() - 1);
  }

  // Monotonicity enforcement with an adaptive tolerance. Each pass merges at
  // most one violating pair, so the loop ends after at most
  // bins.size() - min_bins merges.
  void ensureMonotonicity() {
    if (bins.size() <= 1) return;

    double monotonicity_threshold = std::min(EPSILON, averageWoeGap() * 0.01);
    bool monotonic = false;

    while (!monotonic && static_cast<int>(bins.size()) > min_bins) {
      monotonic = true;

      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i-1].woe - monotonicity_threshold) {
          bins[i-1].merge_with(bins[i]);
          bins[i-1].calculate_metrics(total_pos, total_neg);
          bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(i));

          if (bins.size() > 1) {
            monotonicity_threshold = std::min(EPSILON, averageWoeGap() * 0.01);
          }

          monotonic = false;
          break;
        }
      }
    }
  }

  // Efficient category name joining for bin representation
  std::string joinCategoryNames(const std::vector<std::string>& categories) const {
    if (categories.empty()) return "";
    if (categories.size() == 1) return categories[0];

    size_t total_size = 0;
    for (const auto& cat : categories) {
      total_size += cat.size();
    }
    total_size += bin_separator.size() * (categories.size() - 1);

    std::string result;
    result.reserve(total_size);

    result = categories[0];
    for (size_t i = 1; i < categories.size(); ++i) {
      result += bin_separator;
      result += categories[i];
    }

    return result;
  }

public:
  OBC_GMB(const std::vector<std::string>& feature_,
          const std::vector<int>& target_,
          int min_bins_ = 3,
          int max_bins_ = 5,
          double bin_cutoff_ = 0.05,
          int max_n_prebins_ = 20,
          std::string bin_separator_ = "%;%",
          double convergence_threshold_ = 1e-6,
          int max_iterations_ = 1000)
    : feature(feature_), target(target_), min_bins(min_bins_), max_bins(max_bins_),
      bin_cutoff(bin_cutoff_), max_n_prebins(max_n_prebins_), bin_separator(std::move(bin_separator_)),
      convergence_threshold(convergence_threshold_), max_iterations(max_iterations_) {

    validateInput();
  }

  /// Number of categories whose name contains bin_separator (and one of them)
  std::size_t separatorHits(std::string& example) const {
    example = sep_example;
    return sep_hits;
  }

  Rcpp::List fit() {
    // Initialization (also clamps max_bins/min_bins to the category count)
    initializeBins();

    // Greedy merging
    greedyMerge();

    // Enforce the caller's bin budget: the greedy rule's own tolerance does
    // not guarantee bins.size() <= max_bins.
    enforceMaxBins();

    // Monotonicity enforcement
    ensureMonotonicity();

    const size_t n_bins = bins.size();

    Rcpp::NumericVector ids(n_bins);
    Rcpp::CharacterVector bin_names(n_bins);
    Rcpp::NumericVector woe_values(n_bins);
    Rcpp::NumericVector iv_values(n_bins);
    Rcpp::IntegerVector count_values(n_bins);
    Rcpp::IntegerVector count_pos_values(n_bins);
    Rcpp::IntegerVector count_neg_values(n_bins);

    for (size_t i = 0; i < n_bins; ++i) {
      ids[i] = static_cast<double>(i + 1);
      bin_names[i] = joinCategoryNames(bins[i].categories);
      woe_values[i] = bins[i].woe;
      iv_values[i] = bins[i].iv;
      count_values[i] = bins[i].count;
      count_pos_values[i] = bins[i].count_pos;
      count_neg_values[i] = bins[i].count_neg;
    }

    double total_iv = std::accumulate(iv_values.begin(), iv_values.end(), 0.0);

    return Rcpp::List::create(
      Rcpp::Named("id") = ids,
      Rcpp::Named("bin") = bin_names,
      Rcpp::Named("woe") = woe_values,
      Rcpp::Named("iv") = iv_values,
      Rcpp::Named("count") = count_values,
      Rcpp::Named("count_pos") = count_pos_values,
      Rcpp::Named("count_neg") = count_neg_values,
      Rcpp::Named("total_iv") = total_iv,
      Rcpp::Named("converged") = converged,
      Rcpp::Named("iterations") = iterations_run
    );
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_gmb(Rcpp::IntegerVector target,
                                          Rcpp::StringVector feature,
                                          int min_bins = 3,
                                          int max_bins = 5,
                                          double bin_cutoff = 0.05,
                                          int max_n_prebins = 20,
                                          std::string bin_separator = "%;%",
                                          double convergence_threshold = 1e-6,
                                          int max_iterations = 1000) {
 if (feature.size() == 0 || target.size() == 0) {
   Rcpp::stop("Input vectors cannot be empty.");
 }

 if (feature.size() != target.size()) {
   Rcpp::stop("Feature and target must have the same length (got %d and %d).",
              feature.size(), target.size());
 }

 std::vector<std::string> feature_vec;
 std::vector<int> target_vec;

 feature_vec.reserve(static_cast<size_t>(feature.size()));
 target_vec.reserve(static_cast<size_t>(target.size()));

 R_xlen_t na_feature_count = 0;

 for (R_xlen_t i = 0; i < feature.size(); ++i) {
   if (feature[i] == NA_STRING) {
     feature_vec.push_back("NA");
     na_feature_count++;
   } else {
     feature_vec.push_back(Rcpp::as<std::string>(feature[i]));
   }

   if (IntegerVector::is_na(target[i])) {
     Rcpp::stop("Target cannot contain missing values. Found NA at position %d.", i+1);
   } else {
     target_vec.push_back(target[i]);
   }
 }

 // Warn about NA values in feature (the R wrapper maps them to "NA" first,
 // so this only fires for direct calls)
 if (na_feature_count > 0) {
   Rcpp::warning("%d missing values found in feature and converted to \"NA\" category.",
                 na_feature_count);
 }

 try {
   OBC_GMB binner(feature_vec, target_vec, min_bins, max_bins,
                  bin_cutoff, max_n_prebins, bin_separator,
                  convergence_threshold, max_iterations);
   Rcpp::List res = binner.fit();
   std::string example;
   warn_separator_hits(bin_separator, binner.separatorHits(example), example);
   return res;
 } catch (const std::exception& e) {
   Rcpp::stop("Error in optimal binning: %s", e.what());
 }
}
