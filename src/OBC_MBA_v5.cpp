// [[Rcpp::depends(Rcpp)]]

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Include shared headers
#include "common/bin_structures.h"
#include "common/optimal_binning_common.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Enhanced main class with improved optimization strategies
class OBC_MBA {
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

  int total_good;
  int total_bad;

  std::vector<CategoricalBin> bins;

  // Per-category counts, built once while validating (it also yields the
  // number of categories, which used a separate hash pass over the data).
  // Same reserve and insertion sequence as before, so the iteration order --
  // which decides the order of equal-count bins -- is unchanged.
  std::unordered_map<std::string, CategoricalBin> category_bins_;

  // Enhanced input validation with comprehensive checks
  void validate_inputs() {
    if (feature.size() != target.size()) {
      throw std::invalid_argument(
          "Feature and target must have the same length.");
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

    // Check for empty strings in feature
    if (std::any_of(feature.begin(), feature.end(),
                    [](const std::string &s) { return s.empty(); })) {
      throw std::invalid_argument("Feature cannot contain empty strings. "
                                  "Consider preprocessing your data.");
    }

    // Efficient check for binary target values
    bool has_zero = false;
    bool has_one = false;

    for (int val : target) {
      if (val == 0)
        has_zero = true;
      else if (val == 1)
        has_one = true;
      else
        throw std::invalid_argument("Target must contain only 0 and 1.");
      // No early exit: every value must be checked, otherwise a 2 or -1
      // after the first 0 and 1 was silently counted as a positive.
    }

    if (!has_zero || !has_one) {
      throw std::invalid_argument("Target must contain both 0 and 1 values.");
    }

    // Adjust max_bins based on unique categories
    count_categories();
    int ncat = static_cast<int>(category_bins_.size());
    if (max_bins > ncat) {
      max_bins = ncat;
    }

    // Adjust min_bins if necessary
    min_bins = std::min(min_bins, max_bins);
  }

  // Single-pass counting, one hash lookup per row.
  void count_categories() {
    category_bins_.clear();
    category_bins_.reserve(
        std::min(feature.size() / 4, static_cast<size_t>(1024)));

    total_good = 0;
    total_bad = 0;

    for (size_t i = 0; i < feature.size(); ++i) {
      const auto &cat = feature[i];
      const int is_positive = target[i];

      auto it = category_bins_.find(cat);
      if (it == category_bins_.end()) {
        it = category_bins_.emplace(cat, CategoricalBin()).first;
        it->second.categories.push_back(cat);
      }
      CategoricalBin &bin = it->second;
      bin.count++;
      if (is_positive) {
        bin.count_pos++;
        total_bad++;
      } else {
        bin.count_neg++;
        total_good++;
      }
    }
  }

  // Enhanced prebinning with better statistical handling
  void prebinning() {
    std::unordered_map<std::string, CategoricalBin> category_bins =
        std::move(category_bins_);
    category_bins_ = std::unordered_map<std::string, CategoricalBin>();

    // Check for extremely imbalanced datasets
    if (total_good < 5 || total_bad < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results "
                    "may be unstable.");
    }

    // Transfer to vector and sort by count (descending)
    bins.clear();
    bins.reserve(category_bins.size());

    for (auto &pair : category_bins) {
      bins.push_back(std::move(pair.second));
    }

    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.count > b.count;
              });

    // Reduce to max_n_prebins if necessary with improved strategy
    if (static_cast<int>(bins.size()) > max_n_prebins &&
        static_cast<int>(bins.size()) > min_bins) {
      // Merge the smallest bins first. `bins` is sorted by count descending,
      // so ascending count is descending position. Walking the positions from
      // the back keeps every index valid: a merge either erases the current
      // (last unprocessed) position or a position behind it. The previous
      // version sorted a list of positions taken before any merge; once a
      // merge erased an earlier-listed position, later entries named the
      // wrong bin (or ran past the end), so bins were skipped and pre-binning
      // could stop above max_n_prebins.
      for (size_t idx = bins.size() - 1;
           idx >= static_cast<size_t>(max_n_prebins); --idx) {
        if (static_cast<int>(bins.size()) <= max_n_prebins ||
            static_cast<int>(bins.size()) <= min_bins) {
          break;
        }
        if (idx >= bins.size()) {
          continue;
        }

        // Find best merge candidate based on similarity
        double best_similarity = -1.0;
        size_t best_candidate = 0;

        for (size_t j = 0; j < bins.size(); ++j) {
          if (j == idx)
            continue;

          // Calculate event rate similarity as a merge criterion
          double rate_diff =
              std::fabs(bins[idx].event_rate() - bins[j].event_rate());
          double similarity = 1.0 / (rate_diff + EPSILON);

          if (similarity > best_similarity) {
            best_similarity = similarity;
            best_candidate = j;
          }
        }

        // Try to merge with best candidate
        if (best_candidate < bins.size()) {
          size_t merge_idx1 = std::min(idx, best_candidate);
          size_t merge_idx2 = std::max(idx, best_candidate);

          // Adjust index if it's out of bounds after previous merges
          if (merge_idx1 >= bins.size())
            merge_idx1 = bins.size() - 1;
          if (merge_idx2 >= bins.size())
            merge_idx2 = bins.size() - 1;

          if (merge_idx1 != merge_idx2) {
            try_merge_bins(merge_idx1, merge_idx2);
          }
        }
      }
    }
  }

  // Enhanced bin_cutoff enforcement with improved handling
  void enforce_bin_cutoff() {
    // Calculate minimum counts based on bin_cutoff
    int min_count = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    int min_count_pos = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(total_bad)));

    // Identify all low-frequency bins at once
    std::vector<size_t> low_freq_bins;

    for (size_t i = 0; i < bins.size(); ++i) {
      if (bins[i].count < min_count || bins[i].count_pos < min_count_pos) {
        low_freq_bins.push_back(i);
      }
    }

    // Sort bins by frequency (ascending) for better merging strategy
    std::sort(
        low_freq_bins.begin(), low_freq_bins.end(),
        [this](size_t a, size_t b) { return bins[a].count < bins[b].count; });

    // Process low-frequency bins efficiently
    for (size_t idx : low_freq_bins) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Never go below min_bins
      }

      // Check if bin still exists and is still below cutoff
      if (idx >= bins.size() || (bins[idx].count >= min_count &&
                                 bins[idx].count_pos >= min_count_pos)) {
        continue;
      }

      // Find best merge candidate based on event rate similarity
      double best_similarity = -1.0;
      size_t best_candidate = idx;

      for (size_t j = 0; j < bins.size(); ++j) {
        if (j == idx)
          continue;

        // Calculate event rate similarity
        double rate_diff =
            std::fabs(bins[idx].event_rate() - bins[j].event_rate());
        double similarity = 1.0 / (rate_diff + EPSILON);

        if (similarity > best_similarity) {
          best_similarity = similarity;
          best_candidate = j;
        }
      }

      // Try to merge with best candidate
      size_t kept = 0, erased = 0;
      bool merged = false;
      if (best_candidate != idx && best_candidate < bins.size()) {
        kept = std::min(idx, best_candidate);
        erased = std::max(idx, best_candidate);
        merged = try_merge_bins(kept, erased);
        if (!merged) {
          // If unable to merge with best candidate, try adjacent bins
          if (idx > 0) {
            kept = idx - 1;
            erased = idx;
            merged = try_merge_bins(kept, erased);
          } else if (idx + 1 < bins.size()) {
            kept = idx;
            erased = idx + 1;
            merged = try_merge_bins(kept, erased);
          }
        }
      }

      // try_merge_bins() keeps the lower position and erases the higher one:
      // an entry for the erased bin now names the merged bin, and every
      // position after it moves down by one. The previous adjustment
      // decremented everything after the *lower* position, so entries between
      // the two named the wrong bin and a rare bin could be left unmerged.
      if (merged) {
        for (auto &remaining_idx : low_freq_bins) {
          if (remaining_idx == erased) {
            remaining_idx = kept;
          } else if (remaining_idx > erased) {
            remaining_idx--;
          }
        }
      }
    }
  }

  // Helper to calculate metrics for a single bin (local logic for now)
  void calculate_bin_metrics(CategoricalBin &bin) {
    // Calculate Bayesian prior based on overall prevalence
    double prior_weight = BAYESIAN_PRIOR_STRENGTH;
    double overall_event_rate =
        static_cast<double>(total_bad) / (total_bad + total_good);

    double prior_pos = prior_weight * overall_event_rate;
    double prior_neg = prior_weight * (1.0 - overall_event_rate);

    // Apply Bayesian smoothing to proportions
    double prop_event = static_cast<double>(bin.count_pos + prior_pos) /
                        static_cast<double>(total_bad + prior_weight);
    double prop_non_event = static_cast<double>(bin.count_neg + prior_neg) /
                            static_cast<double>(total_good + prior_weight);

    // Ensure numerical stability
    prop_event = std::max(prop_event, EPSILON);
    prop_non_event = std::max(prop_non_event, EPSILON);

    // Calculate WoE and IV
    bin.woe = std::log(prop_event /
                       prop_non_event); // using std::log, EPSILON handled
    bin.iv = (prop_event - prop_non_event) * bin.woe;

    // Handle non-finite values
    if (!std::isfinite(bin.woe))
      bin.woe = 0.0;
    if (!std::isfinite(bin.iv))
      bin.iv = 0.0;
  }

  // Enhanced initial WoE calculation
  void calculate_initial_woe() {
    for (auto &bin : bins) {
      calculate_bin_metrics(bin);
    }
  }

  // Enhanced monotonicity enforcement with adaptive thresholds
  void enforce_monotonicity() {
    if (bins.empty()) {
      throw std::runtime_error("No bins available to enforce monotonicity.");
    }

    // Sort bins by WoE
    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.woe < b.woe;
              });

    // Calculate average WoE gap for adaptive threshold
    double total_woe_gap = 0.0;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        total_woe_gap += std::fabs(bins[i].woe - bins[i - 1].woe);
      }
    }

    double avg_gap =
        bins.size() > 1
            ? total_woe_gap / static_cast<double>(bins.size() - 1)
            : 0.0;
    double monotonicity_threshold = std::min(EPSILON, avg_gap * 0.01);

    // Determine monotonicity direction
    bool increasing = true;
    if (bins.size() > 1) {
      for (size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].woe < bins[i - 1].woe - monotonicity_threshold) {
          increasing = false;
          break;
        }
      }
    }

    // Optimized loop to enforce monotonicity
    bool any_merge;
    int attempts = 0;
    const int max_attempts = static_cast<int>(bins.size() * 3); // Safety limit

    do {
      any_merge = false;

      // Find all violations and their severity
      std::vector<std::pair<size_t, double>> violations;

      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        bool violation =
            (increasing &&
             bins[i].woe > bins[i + 1].woe + monotonicity_threshold) ||
            (!increasing &&
             bins[i].woe < bins[i + 1].woe - monotonicity_threshold);

        if (violation) {
          double severity = std::fabs(bins[i].woe - bins[i + 1].woe);
          violations.push_back({i, severity});
        }
      }

      // If no violations, we're done
      if (violations.empty())
        break;

      // Sort violations by severity (descending)
      std::sort(
          violations.begin(), violations.end(),
          [](const auto &a, const auto &b) { return a.second > b.second; });

      // Fix the most severe violation
      if (!violations.empty() && static_cast<int>(bins.size()) > min_bins) {
        size_t idx = violations[0].first;
        if (try_merge_bins(idx, idx + 1)) {
          any_merge = true;
        }
      }

      attempts++;
      if (attempts >= max_attempts) {
        Rcpp::warning("Could not ensure monotonicity in %d attempts. Using "
                      "best solution found.",
                      max_attempts);
        break;
      }

    } while (any_merge && static_cast<int>(bins.size()) > min_bins);

    // Final sort by WoE to ensure correct order
    std::sort(bins.begin(), bins.end(),
              [](const CategoricalBin &a, const CategoricalBin &b) {
                return a.woe < b.woe;
              });
  }

  // Enhanced bin optimization with improved IV-based strategy
  void optimize_bins() {
    if (static_cast<int>(bins.size()) <= max_bins) {
      return; // Already within maximum limit
    }

    // Track best solution seen so far
    double best_total_iv = 0.0;
    std::vector<CategoricalBin> best_bins = bins;

    for (const auto &bin : bins) {
      best_total_iv += std::fabs(bin.iv);
    }

    int iterations = 0;

    while (static_cast<int>(bins.size()) > max_bins &&
           iterations < max_iterations) {
      if (static_cast<int>(bins.size()) <= min_bins) {
        break; // Don't reduce below min_bins
      }

      // Find pair of bins with minimum combined IV or minimum IV loss
      double min_iv_loss = std::numeric_limits<double>::max();
      size_t min_iv_index = 0;

      // The loss of a pair depends only on the two bins, so it is evaluated
      // directly; the merged bin is simulated from the counts alone instead of
      // copying the category-name vector for every candidate.
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        CategoricalBin merged_bin;
        merged_bin.count = bins[i].count + bins[i + 1].count;
        merged_bin.count_pos = bins[i].count_pos + bins[i + 1].count_pos;
        merged_bin.count_neg = bins[i].count_neg + bins[i + 1].count_neg;
        calculate_bin_metrics(merged_bin);

        double original_iv = std::fabs(bins[i].iv) + std::fabs(bins[i + 1].iv);
        double new_iv = std::fabs(merged_bin.iv);
        double iv_loss = original_iv - new_iv;

        if (iv_loss < min_iv_loss) {
          min_iv_loss = iv_loss;
          min_iv_index = i;
        }
      }

      // Try to merge bins with minimum IV loss
      if (!try_merge_bins(min_iv_index, min_iv_index + 1)) {
        break; // Unable to merge, stop
      }

      // Calculate new total IV
      double total_iv = 0.0;
      for (const auto &bin : bins) {
        total_iv += std::fabs(bin.iv);
      }

      // Track best solution
      if (static_cast<int>(bins.size()) <= max_bins &&
          total_iv > best_total_iv) {
        best_total_iv = total_iv;
        best_bins = bins;
      }

      // No IV-convergence exit here: the loop runs only while there are
      // more than max_bins bins, and stopping it early because one merge
      // happened to cost less than convergence_threshold (two bins with the
      // same event rate) left the result above max_bins, raised the warning
      // below and made fit() call this function again -- one warning per
      // such merge.
      iterations++;
    }

    // Restore best solution if we found a better one
    if (best_total_iv > 0.0 && static_cast<int>(best_bins.size()) <= max_bins &&
        static_cast<int>(best_bins.size()) >= min_bins) {
      bins = std::move(best_bins);
    }

    if (static_cast<int>(bins.size()) > max_bins) {
      Rcpp::warning("Could not reduce number of bins to max_bins without "
                    "violating min_bins or convergence constraints.");
    }
  }

  // Enhanced bin merging with improved checks
  bool try_merge_bins(size_t index1, size_t index2) {
    // Safety checks
    if (static_cast<int>(bins.size()) <= min_bins) {
      return false; // Already at minimum, don't merge
    }

    if (index1 >= bins.size() || index2 >= bins.size() || index1 == index2) {
      return false;
    }

    if (index2 < index1)
      std::swap(index1, index2);

    // Efficient merging
    bins[index1].merge_with(bins[index2]);
    calculate_bin_metrics(bins[index1]);

    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(index2));

    return true;
  }

  // Enhanced consistency check
  void check_consistency() const {
    int total_count = 0;
    int total_count_pos = 0;
    int total_count_neg = 0;

    for (const auto &bin : bins) {
      total_count += bin.count;
      total_count_pos += bin.count_pos;
      total_count_neg += bin.count_neg;
    }

    if (static_cast<size_t>(total_count) != feature.size()) {
      throw std::runtime_error("Inconsistent total count after binning.");
    }

    if (total_count_pos != total_bad || total_count_neg != total_good) {
      throw std::runtime_error(
          "Inconsistent positive/negative counts after binning.");
    }
  }

public:
  // Enhanced constructor with improved parameter handling
  OBC_MBA(std::vector<std::string> feature_,
          const Rcpp::IntegerVector &target_, int min_bins_ = 3,
          int max_bins_ = 5, double bin_cutoff_ = 0.05, int max_n_prebins_ = 20,
          std::string bin_separator_ = "%;%",
          double convergence_threshold_ = 1e-6, int max_iterations_ = 1000)
      : feature(std::move(feature_)),
        target(Rcpp::as<std::vector<int>>(target_)),
        min_bins(min_bins_), max_bins(max_bins_), bin_cutoff(bin_cutoff_),
        max_n_prebins(max_n_prebins_), bin_separator(bin_separator_),
        convergence_threshold(convergence_threshold_),
        max_iterations(max_iterations_), total_good(0), total_bad(0) {

    // Pre-allocation for better performance
    bins.reserve(std::min(feature.size() / 4, static_cast<size_t>(1024)));
  }

  // Enhanced fit method with better algorithm flow
  Rcpp::List fit() {
    try {
      // Binning process with performance checks
      validate_inputs();
      prebinning();
      enforce_bin_cutoff();
      calculate_initial_woe();
      enforce_monotonicity();

      // CategoricalBin optimization if necessary
      bool converged_flag = false;
      int iterations_done = 0;

      if (static_cast<int>(bins.size()) <= max_bins) {
        converged_flag = true;
      } else {
        double prev_total_iv = 0.0;
        for (const auto &bin : bins) {
          prev_total_iv += std::fabs(bin.iv);
        }

        // Track best solution
        double best_total_iv = prev_total_iv;
        std::vector<CategoricalBin> best_bins = bins;

        for (int i = 0; i < max_iterations; ++i) {
          // Starting point for this iteration
          size_t start_bins = bins.size();

          optimize_bins();

          // Calculate current IV
          double total_iv = 0.0;
          for (const auto &bin : bins) {
            total_iv += std::fabs(bin.iv);
          }

          // Track best solution
          if (static_cast<int>(bins.size()) <= max_bins &&
              static_cast<int>(bins.size()) >= min_bins &&
              total_iv > best_total_iv) {
            best_total_iv = total_iv;
            best_bins = bins;
          }

          // Check for convergence or no change
          if (bins.size() == start_bins ||
              static_cast<int>(bins.size()) <= max_bins) {
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

        // Restore best solution if found
        if (best_total_iv > 0.0 &&
            static_cast<int>(best_bins.size()) <= max_bins &&
            static_cast<int>(best_bins.size()) >= min_bins) {
          bins = std::move(best_bins);
        }

        // Leaving the loop because the bin-count target was reached is a valid
        // stopping state, just like meeting the IV tolerance above. Only
        // exhausting max_iterations leaves converged_flag == false.
        if (iterations_done < max_iterations) {
          converged_flag = converged_flag ||
                           (static_cast<int>(bins.size()) <= max_bins);
        }
      }

      // Report the bins in WoE order. Merging two WoE-adjacent bins does not
      // always keep that order under Bayesian smoothing (two pure bins of
      // different sizes), which left the documented monotone WoE unmet. The
      // sort is stable, so an already ordered result is unchanged.
      std::stable_sort(bins.begin(), bins.end(),
                       [](const CategoricalBin &a, const CategoricalBin &b) {
                         return a.woe < b.woe;
                       });

      // Final consistency check
      check_consistency();

      // Optimized results preparation with uniqueness guarantee
      const size_t n_bins = bins.size();

      CharacterVector bin_names(n_bins);
      NumericVector bin_woe(n_bins);
      NumericVector bin_iv(n_bins);
      IntegerVector bin_count(n_bins);
      IntegerVector bin_count_pos(n_bins);
      IntegerVector bin_count_neg(n_bins);
      NumericVector ids(n_bins);

      double total_iv = 0.0;

      for (size_t i = 0; i < n_bins; ++i) {
        // Use shared bin structure name method
        bin_names[i] = bins[i].name(bin_separator);
        bin_woe[i] = bins[i].woe;
        bin_iv[i] = bins[i].iv;
        bin_count[i] = bins[i].count;
        bin_count_pos[i] = bins[i].count_pos;
        bin_count_neg[i] = bins[i].count_neg;
        ids[i] = static_cast<double>(i + 1);

        total_iv += std::fabs(bins[i].iv);
      }

      return Rcpp::List::create(
          Named("id") = ids, Named("bin") = bin_names, Named("woe") = bin_woe,
          Named("iv") = bin_iv, Named("count") = bin_count,
          Named("count_pos") = bin_count_pos,
          Named("count_neg") = bin_count_neg, Named("total_iv") = total_iv,
          Named("converged") = converged_flag,
          Named("iterations") = iterations_done);
    } catch (const std::exception &e) {
      Rcpp::stop("Error in optimal binning: %s", e.what());
    }
  }
};

// [[Rcpp::export]]
Rcpp::List optimal_binning_categorical_mba(
    Rcpp::IntegerVector target, Rcpp::CharacterVector feature, int min_bins = 3,
    int max_bins = 5, double bin_cutoff = 0.05, int max_n_prebins = 20,
    std::string bin_separator = "%;%", double convergence_threshold = 1e-6,
    int max_iterations = 1000) {
  // Preliminary checks
  if (feature.size() == 0 || target.size() == 0) {
    Rcpp::stop("Feature and target cannot be empty.");
  }

  if (feature.size() != target.size()) {
    Rcpp::stop("Feature and target must have the same length.");
  }

  // Optimized R to C++ conversion
  std::vector<std::string> feature_vec;
  feature_vec.reserve(feature.size());

  int na_feature_count = 0;

  for (R_xlen_t i = 0; i < feature.size(); ++i) {
    // Handle NA in feature
    SEXP s = STRING_ELT(feature, i);
    if (s == NA_STRING) {
      feature_vec.emplace_back("NA");
      na_feature_count++;
    } else {
      feature_vec.emplace_back(CHAR(s));
    }

    // Check for NA in target
    if (IntegerVector::is_na(target[i])) {
      Rcpp::stop("Target cannot contain missing values at position %d.", i + 1);
    }
  }

  // Warn about NA values in feature
  if (na_feature_count > 0) {
    Rcpp::warning(
        "%d missing values found in feature and converted to \"NA\" category.",
        na_feature_count);
  }

  // Execute optimized algorithm
  OBC_MBA mba(std::move(feature_vec), target, min_bins, max_bins, bin_cutoff,
              max_n_prebins, bin_separator, convergence_threshold,
              max_iterations);

  return mba.fit();
}