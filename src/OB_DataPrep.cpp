// OB_DataPrep.cpp: missing-value and outlier preprocessing (ob_preprocess()).

#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <numeric>
#include <unordered_set>
#include <cstring>

using namespace Rcpp;

namespace {

// Summary statistics of the non-missing values
List compute_summary(const NumericVector& data) {
  std::vector<double> vec;
  vec.reserve(static_cast<size_t>(data.size()));
  for (double val : data) {
    if (!NumericVector::is_na(val)) vec.push_back(val);
  }
  const size_t n = vec.size();
  if (n == 0) {
    return List::create(Named("min") = NA_REAL,
                        Named("Q1") = NA_REAL,
                        Named("median") = NA_REAL,
                        Named("mean") = NA_REAL,
                        Named("Q3") = NA_REAL,
                        Named("max") = NA_REAL);
  }

  std::sort(vec.begin(), vec.end());

  const double dn = static_cast<double>(n);
  auto get_percentile = [&](double p) -> double {
    if (n == 1) return vec[0];
    const double pos = p * (dn + 1.0) / 100.0;
    if (pos < 1.0) return vec[0];
    if (pos >= dn) return vec[n - 1];
    const size_t idx = static_cast<size_t>(std::floor(pos)) - 1;
    const double frac = pos - std::floor(pos);
    return vec[idx] + frac * (vec[idx + 1] - vec[idx]);
  };

  const double mean = std::accumulate(vec.begin(), vec.end(), 0.0) / dn;

  return List::create(Named("min") = vec.front(),
                      Named("Q1") = get_percentile(25.0),
                      Named("median") = get_percentile(50.0),
                      Named("mean") = mean,
                      Named("Q3") = get_percentile(75.0),
                      Named("max") = vec.back());
}

// Two-sided Grubbs critical value for a sample of size n
double grubbs_critical(size_t n, double alpha) {
  const double dn = static_cast<double>(n);
  const double t = R::qt(1 - alpha / (2 * dn), dn - 2, 1, 0);
  const double numerator = (dn - 1) * std::sqrt(t * t);
  const double denominator = std::sqrt(dn) * std::sqrt(dn - 2 + t * t);
  return numerator / denominator;
}

// Iterative two-sided Grubbs test: while the most extreme observation is
// significant, remove it. Returns the positions of the removed observations.
//
// The observation farthest from the mean is always the smallest or the
// largest remaining value, so the remaining sample is a contiguous range of
// the sorted data and each round costs O(1): the sums over that range come
// from cumulative sums built outward from the median (removed extremes never
// enter the sums of the retained values, so there is no cancellation), and
// the variance uses the data shifted by the median. The former
// implementation recomputed mean and SD over the whole sample and erased
// from the middle of a vector every round, O(n) per removal -- quadratic on
// heavy-tailed data. The decisions are the same; the mean and SD may differ
// from the two-pass values in the last bits only. Among tied extremes the
// earliest observation is removed first, as before. A sample containing
// +-Inf has an undefined mean and SD: nothing is removed (as before).
std::vector<R_xlen_t> grubbs_outliers(std::vector<std::pair<double, R_xlen_t>> data, double alpha) {
  std::vector<R_xlen_t> removed;
  const size_t m = data.size();
  if (m <= 2) return removed;
  for (const auto& d : data) {
    if (!std::isfinite(d.first)) return removed;
  }
  std::sort(data.begin(), data.end());  // by value, then by position

  const size_t mid = m / 2;
  const double c = data[mid].first;
  // F1(j), F2(j): signed cumulative sums of (v - c) and (v - c)^2 anchored at
  // mid, so that sum over [lo, hi] = F(hi + 1) - F(lo).
  std::vector<double> F1(m + 1, 0.0), F2(m + 1, 0.0);
  for (size_t j = mid; j < m; ++j) {
    const double d = data[j].first - c;
    F1[j + 1] = F1[j] + d;
    F2[j + 1] = F2[j] + d * d;
  }
  for (size_t j = mid; j-- > 0;) {
    const double d = data[j].first - c;
    F1[j] = F1[j + 1] - d;
    F2[j] = F2[j + 1] - d * d;
  }

  size_t lo = 0, hi = m - 1;             // remaining values: data[lo..hi]
  size_t run_start = 0, run_next = 0;    // top run of equal values being consumed
  bool run_active = false;

  while (hi - lo + 1 > 2) {
    const size_t cnt = hi - lo + 1;
    if (data[lo].first == data[hi].first) break;  // all equal: SD = 0

    const double dn = static_cast<double>(cnt);
    const double s1 = F1[hi + 1] - F1[lo];
    const double s2 = F2[hi + 1] - F2[lo];
    const double mean = c + s1 / dn;
    double var = (s2 - s1 * s1 / dn) / (dn - 1.0);
    if (!(var > 0.0)) break;
    const double sd = std::sqrt(var);

    // Positions of the removal candidates at each end: the bottom run is
    // consumed left to right (ascending positions); the top run is located
    // once and consumed in ascending position order as well.
    if (!run_active || hi < run_start || data[hi].first != data[run_start].first) {
      run_start = hi;
      while (run_start > lo && data[run_start - 1].first == data[hi].first) --run_start;
      run_next = run_start;
      run_active = true;
    }
    const double dev_lo = std::abs(data[lo].first - mean);
    const double dev_hi = std::abs(data[hi].first - mean);
    bool take_top;
    if (dev_hi > dev_lo) take_top = true;
    else if (dev_lo > dev_hi) take_top = false;
    else take_top = data[run_next].second < data[lo].second;
    const double max_dev = take_top ? dev_hi : dev_lo;

    if (!(max_dev / sd > grubbs_critical(cnt, alpha))) break;
    if (take_top) {
      removed.push_back(data[run_next].second);
      ++run_next;
      --hi;
    } else {
      removed.push_back(data[lo].second);
      ++lo;
    }
  }
  return removed;
}

bool vector_contains(const CharacterVector& vec, const char* str) {
  for (R_xlen_t i = 0; i < vec.size(); i++) {
    SEXP s = STRING_ELT(vec, i);
    if (s != NA_STRING && std::strcmp(CHAR(s), str) == 0) return true;
  }
  return false;
}

// "{ name: value, ... }" rendering of a flat list of scalars
std::string list_to_string(const List& lst) {
  std::string result = "{ ";
  SEXP names = Rf_getAttrib(lst, R_NamesSymbol);
  for (R_xlen_t i = 0; i < lst.size(); i++) {
    const std::string name = (names == R_NilValue) ? std::string() : std::string(CHAR(STRING_ELT(names, i)));
    std::string value;
    SEXP el = lst[i];
    if (TYPEOF(el) == REALSXP) {
      value = std::to_string(as<double>(el));
    } else if (TYPEOF(el) == STRSXP) {
      SEXP s = STRING_ELT(el, 0);
      value = (s == NA_STRING) ? "NA" : std::string(CHAR(s));
    } else {
      value = "NA";
    }
    result += name + ": " + value + ", ";
  }
  if (result.size() > 2) {
    result = result.substr(0, result.size() - 2);
  }
  result += " }";
  return result;
}

} // namespace

// Missing values are replaced by the sentinel (num_miss_value /
// char_miss_value). Outlier detection and treatment only ever look at the
// values that were observed: missing entries keep the sentinel, do not enter
// the quartiles / mean / standard deviation, and are never counted as
// outliers. Neither input vector is modified: 'feature' holds the original
// values and 'feature_preprocessed' the treated ones.
// [[Rcpp::export]]
List OBDataPreprocessor(
    NumericVector target,
    SEXP feature,
    double num_miss_value = -999.0,
    std::string char_miss_value = "N/A",
    std::string outlier_method = "iqr",
    bool outlier_process = false,
    CharacterVector preprocess = CharacterVector::create("both"),
    double iqr_k = 1.5,
    double zscore_threshold = 3.0,
    double grubbs_alpha = 0.05)
{
  std::string variable_type = "unknown";
  int missing_count = 0;
  int outlier_count = 0;
  List original_stats;
  List preprocessed_stats;

  // Target must be binary
  std::unordered_set<double> target_unique;
  for (double val : target) {
    if (!NumericVector::is_na(val)) target_unique.insert(val);
  }
  if (target_unique.size() != 2) {
    stop("Target variable must be binary.");
  }

  bool is_numeric = false;
  NumericVector feature_original;   // original values (never modified)
  NumericVector feature_numeric;    // treated copy
  CharacterVector character_original;
  CharacterVector feature_character;

  if (TYPEOF(feature) == REALSXP || Rf_isInteger(feature) || Rf_isReal(feature)) {
    is_numeric = true;
    feature_original = as<NumericVector>(feature);
    // clone(): as<NumericVector>() does not copy a double vector, so writing
    // into it used to overwrite the caller's own R object.
    feature_numeric = clone(feature_original);
    variable_type = "numeric";
  } else if (TYPEOF(feature) == STRSXP || Rf_isFactor(feature) || Rf_isString(feature)) {
    character_original = as<CharacterVector>(feature);
    feature_character = clone(character_original);
    variable_type = "categorical";
  } else {
    stop("Feature must be either numeric or categorical (string).");
  }

  // Positions holding an observed (non-missing) value
  std::vector<R_xlen_t> observed;

  if (is_numeric) {
    original_stats = compute_summary(feature_numeric);
    observed.reserve(static_cast<size_t>(feature_numeric.size()));
    for (R_xlen_t i = 0; i < feature_numeric.size(); ++i) {
      if (NumericVector::is_na(feature_numeric[i])) {
        feature_numeric[i] = num_miss_value;
        missing_count++;
      } else {
        observed.push_back(i);
      }
    }
  } else {
    original_stats = List::create(Named("min") = NA_STRING,
                                  Named("Q1") = NA_STRING,
                                  Named("median") = NA_STRING,
                                  Named("mean") = NA_STRING,
                                  Named("Q3") = NA_STRING,
                                  Named("max") = NA_STRING);
    for (R_xlen_t i = 0; i < feature_character.size(); ++i) {
      if (feature_character[i] == NA_STRING) {
        feature_character[i] = char_miss_value;
        missing_count++;
      }
    }
  }

  if (is_numeric && outlier_process) {
    const size_t m = observed.size();
    if (outlier_method == "iqr") {
      if (m >= 1) {
        std::vector<double> sorted;
        sorted.reserve(m);
        for (R_xlen_t i : observed) sorted.push_back(feature_numeric[i]);
        std::sort(sorted.begin(), sorted.end());
        // Same order-statistic rule as before, clamped so that a sample of
        // one or two values no longer reads sorted[-1].
        auto order_stat = [&](double p) -> double {
          double k = std::floor(p * (static_cast<double>(m) + 1.0)) - 1.0;
          if (k < 0.0) k = 0.0;
          if (k > static_cast<double>(m - 1)) k = static_cast<double>(m - 1);
          return sorted[static_cast<size_t>(k)];
        };
        const double Q1 = order_stat(0.25);
        const double Q3 = order_stat(0.75);
        const double IQR = Q3 - Q1;
        const double lower_bound = Q1 - iqr_k * IQR;
        const double upper_bound = Q3 + iqr_k * IQR;

        for (R_xlen_t i : observed) {
          const double val = feature_numeric[i];
          if (val < lower_bound) {
            feature_numeric[i] = lower_bound;
            outlier_count++;
          } else if (val > upper_bound) {
            feature_numeric[i] = upper_bound;
            outlier_count++;
          }
        }
      }

    } else if (outlier_method == "zscore") {
      if (m >= 2) {
        double sum = 0.0;
        for (R_xlen_t i : observed) sum += feature_numeric[i];
        const double mean = sum / static_cast<double>(m);
        double sq_sum = 0.0;
        for (R_xlen_t i : observed) {
          const double d = feature_numeric[i] - mean;
          sq_sum += d * d;
        }
        const double sd = std::sqrt(sq_sum / static_cast<double>(m - 1));

        const double lower_bound = mean - zscore_threshold * sd;
        const double upper_bound = mean + zscore_threshold * sd;
        for (R_xlen_t i : observed) {
          const double val = feature_numeric[i];
          if (val < lower_bound) {
            feature_numeric[i] = lower_bound;
            outlier_count++;
          } else if (val > upper_bound) {
            feature_numeric[i] = upper_bound;
            outlier_count++;
          }
        }
      }

    } else if (outlier_method == "grubbs") {
      std::vector<std::pair<double, R_xlen_t>> data;
      data.reserve(m);
      for (R_xlen_t i : observed) data.emplace_back(feature_numeric[i], i);
      for (R_xlen_t pos : grubbs_outliers(std::move(data), grubbs_alpha)) {
        feature_numeric[pos] = num_miss_value;
        outlier_count++;
      }
    } else {
      stop("Invalid outlier_method. Choose from 'iqr', 'zscore', or 'grubbs'.");
    }
  }

  if (is_numeric) {
    preprocessed_stats = compute_summary(feature_numeric);
  } else {
    std::unordered_set<std::string> unique_cats;
    for (R_xlen_t i = 0; i < feature_character.size(); ++i) {
      SEXP s = STRING_ELT(feature_character, i);
      if (s != NA_STRING) unique_cats.insert(std::string(CHAR(s)));
    }
    preprocessed_stats = List::create(Named("unique_count") = unique_cats.size());
  }

  const std::string original_stats_str = list_to_string(original_stats);
  const std::string preprocessed_stats_str = list_to_string(preprocessed_stats);

  const bool want_feature = vector_contains(preprocess, "feature") || vector_contains(preprocess, "both");
  const bool want_report = vector_contains(preprocess, "report") || vector_contains(preprocess, "both");

  List output;
  if (want_feature) {
    if (is_numeric) {
      output["preprocess"] = DataFrame::create(
        Named("feature") = feature_original,
        Named("feature_preprocessed") = feature_numeric
      );
    } else {
      output["preprocess"] = DataFrame::create(
        Named("feature") = character_original,
        Named("feature_preprocessed") = feature_character
      );
    }
  }
  if (want_report) {
    output["report"] = DataFrame::create(
      Named("variable_type") = variable_type,
      Named("missing_count") = missing_count,
      Named("outlier_count") = (is_numeric ? outlier_count : NA_INTEGER),
      Named("original_stats") = original_stats_str,
      Named("preprocessed_stats") = preprocessed_stats_str
    );
  }

  return output;
}
