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
#include "common/ob_core.h"

using namespace Rcpp;
using namespace OptimalBinning;

// Ascending WoE; equal WoE by ascending count, so ties do not depend on the
// position a bin happened to reach.
static bool woe_then_count(const CategoricalBin &a, const CategoricalBin &b) {
  if (a.woe != b.woe)
    return a.woe < b.woe;
  return a.count < b.count;
}

namespace {
// Strength (pseudo-observations) of the empirical-Bayes beta prior, centred on
// the overall event rate, that orders the categories and drives the merge
// decisions (audit E-03 follow-up, Section 15.5): a group of m = 20 rows
// moves its rate half-way from the prior to the data. The hold-out log-loss
// of the harness corpus was stable for m in [5, 20].
constexpr double kEvidenceM = 20.0;
} // namespace

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
  // number of categories). Bins are taken from it in category-name order.
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

  // Empirical-Bayes (posterior mean) event rate of a bin under a beta prior
  // centred on the overall rate with kEvidenceM pseudo-observations.
  double shrunk_rate(const CategoricalBin &bin) const {
    const double r0 = static_cast<double>(total_bad) /
                      static_cast<double>(total_bad + total_good);
    return (static_cast<double>(bin.count_pos) + kEvidenceM * r0) /
           (static_cast<double>(bin.count) + kEvidenceM);
  }

  // Position of the smallest bin (the first one on ties).
  size_t smallest_bin() const {
    size_t k = 0;
    for (size_t i = 1; i < bins.size(); ++i) {
      if (bins[i].count < bins[k].count) k = i;
    }
    return k;
  }

  // Merge bins[i] into the adjacent bin (in shrunk-rate order) whose shrunk
  // rate is closer; the left one on ties. Merging neighbours keeps every bin
  // a contiguous run of categories in that order.
  void merge_into_neighbour(size_t i) {
    const size_t K = bins.size();
    if (K < 2) return;
    size_t left;
    if (i == 0) {
      left = 0;
    } else if (i + 1 == K) {
      left = K - 2;
    } else {
      const double s = shrunk_rate(bins[i]);
      left = (std::fabs(s - shrunk_rate(bins[i - 1])) <=
              std::fabs(s - shrunk_rate(bins[i + 1])))
                 ? i - 1
                 : i;
    }
    bins[left].merge_with(bins[left + 1]);
    bins.erase(bins.begin() + static_cast<std::ptrdiff_t>(left + 1));
  }

  // Pre-binning: one bin per category, in shrunk-rate order; then the
  // smallest bin is merged into its closer neighbour until max_n_prebins is
  // met. The baseline merged each small category into the bin with the
  // closest *raw* event rate. With many sparse categories (200 categories of
  // about 10 rows) the raw rates of tiny categories are mostly 0 or 1, so the
  // categories that had no event by chance were pooled into a bin whose
  // training rate (0.3%) was far below its hold-out rate (10%): the winner's
  // curse. The posterior mean (pos + m r0) / (n + m) ranks a small category
  // near the overall rate unless its data say otherwise (Gelman et al.,
  // Bayesian Data Analysis, ch. 5); for large categories it is the raw rate.
  void prebinning() {
    std::unordered_map<std::string, CategoricalBin> category_bins =
        std::move(category_bins_);
    category_bins_ = std::unordered_map<std::string, CategoricalBin>();

    // Check for extremely imbalanced datasets
    if (total_good < 5 || total_bad < 5) {
      Rcpp::warning("Dataset has fewer than 5 samples in one class. Results "
                    "may be unstable.");
    }

    bins.clear();
    bins.reserve(category_bins.size());
    for (auto &pair : category_bins) {
      bins.push_back(std::move(pair.second));
    }

    // Shrunk rate, then count, then name: the order depends only on the data,
    // not on the row order or the hash-map layout.
    std::sort(bins.begin(), bins.end(),
              [this](const CategoricalBin &a, const CategoricalBin &b) {
                const double sa = shrunk_rate(a), sb = shrunk_rate(b);
                if (sa != sb) return sa < sb;
                if (a.count != b.count) return a.count < b.count;
                return a.categories.front() < b.categories.front();
              });

    while (static_cast<int>(bins.size()) > max_n_prebins &&
           static_cast<int>(bins.size()) > min_bins) {
      merge_into_neighbour(smallest_bin());
    }
  }

  // bin_cutoff is a share of *observations* (documented contract; Siddiqi
  // 2006). The baseline also required count_pos >= bin_cutoff * total_bad,
  // which merged large low-risk bins (few events, strongly negative WoE) and
  // every pure non-event bin (audit E-03 / X-15). That undocumented rule also
  // acted as an accidental regulariser on sparse high-cardinality features;
  // its role is now played, explicitly, by the shrunk-rate order above and
  // the empirical-Bayes merge criterion of optimize_bins(). The smallest bin
  // below the cutoff is merged into its closer neighbour until none is left
  // (or min_bins is reached).
  void enforce_bin_cutoff() {
    const int min_count = static_cast<int>(
        std::ceil(bin_cutoff * static_cast<double>(feature.size())));
    while (static_cast<int>(bins.size()) > min_bins && bins.size() > 1) {
      const size_t k = smallest_bin();
      if (bins[k].count >= min_count) break;
      merge_into_neighbour(k);
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
    std::stable_sort(bins.begin(), bins.end(), woe_then_count);

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
      std::stable_sort(
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
    std::stable_sort(bins.begin(), bins.end(), woe_then_count);
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
      // Merge criterion: the IV lost by the merge under the empirical-Bayes
      // prior (ob::eb_woe_iv, kEvidenceM pseudo-observations per bin, shares
      // normalized over the current K bins). The baseline compared the
      // reported IV of the pair with that of the merged bin; a small bin with
      // an extreme observed rate (a pure category) then looked too valuable
      // to merge, and two large, genuinely different bins were merged
      // instead. Shrinkage makes a bin's IV proportional to the evidence it
      // carries. Only the decision changes: WoE/IV are still reported with
      // the estimator of calculate_bin_metrics().
      const int K = static_cast<int>(bins.size());
      const double Tp = static_cast<double>(total_bad);
      const double Tn = static_cast<double>(total_good);
      for (size_t i = 0; i + 1 < bins.size(); ++i) {
        const double pa = bins[i].count_pos, na = bins[i].count_neg;
        const double pb = bins[i + 1].count_pos, nb = bins[i + 1].count_neg;
        double iv_loss =
            ob::eb_woe_iv(pa, na, Tp, Tn, K, kEvidenceM).iv +
            ob::eb_woe_iv(pb, nb, Tp, Tn, K, kEvidenceM).iv -
            ob::eb_woe_iv(pa + pb, na + nb, Tp, Tn, K, kEvidenceM).iv;

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
      std::stable_sort(bins.begin(), bins.end(), woe_then_count);

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