// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <sstream>
#include <cmath>

using namespace Rcpp;

// Scores a numerical feature with fitted cut points.
//
// For k sorted cut points c_1 <= ... <= c_k there are k + 1 bins. With
// include_upper_bound = TRUE they are right-closed, (-Inf, c_1], (c_1, c_2],
// ..., (c_k, +Inf), exactly the intervals obwoe_apply() (cut(right = TRUE))
// and obwoe_sql() use; with FALSE they are left-closed, [c_{j-1}, c_j). The
// bin of every value is found by one binary search, O(log k), and the output
// labels are shared CHARSXPs built once per bin, so scoring is O(n log k).
//
// 'woe' and 'id' normally carry k + 1 entries. They may carry k + 2 when the
// fit has a dedicated missing-value bin (e.g. ob_numerical_udt() on data with
// NA appends a bin labelled "NA"): that bin is the one whose 'bin' label is
// "NA" or "Missing" (the last one when there is no such label), and NA/NaN
// values and values listed in 'missing_values' are scored with it. Without
// such a bin they are scored as "Special" with WoE NA, as before.
// [[Rcpp::export]]
DataFrame OBApplyWoENum(const List& obresults,
                        const NumericVector& feature,
                        bool include_upper_bound = true,
                        Nullable<NumericVector> missing_values = R_NilValue) {

  if (!obresults.containsElementNamed("cutpoints")) {
    stop("The 'obresults' list must contain a 'cutpoints' element.");
  }
  if (!obresults.containsElementNamed("woe")) {
    stop("The 'obresults' list must contain a 'woe' element.");
  }
  if (!obresults.containsElementNamed("id")) {
    stop("The 'obresults' list must contain an 'id' element.");
  }

  NumericVector cutpoints_nv = obresults["cutpoints"];
  NumericVector woe_nv = obresults["woe"];
  NumericVector id_nv = obresults["id"];

  if (any(is_na(cutpoints_nv))) {
    stop("Cutpoints cannot contain NA values.");
  }
  if (any(is_na(woe_nv))) {
    stop("WoE values cannot contain NA values.");
  }
  if (any(is_na(id_nv))) {
    stop("ID values cannot contain NA values.");
  }

  std::vector<double> cutpoints = as<std::vector<double>>(cutpoints_nv);
  if (!std::is_sorted(cutpoints.begin(), cutpoints.end())) {
    stop("Cutpoints must be sorted in ascending order.");
  }

  const size_t num_bins = cutpoints.size() + 1;
  const size_t n_woe = static_cast<size_t>(woe_nv.size());
  if (static_cast<size_t>(id_nv.size()) != n_woe ||
      (n_woe != num_bins && n_woe != num_bins + 1)) {
    stop("The number of WoE values and IDs must be equal to the number of bins (cutpoints + 1), "
         "or cutpoints + 2 when the fit carries a missing-value bin.");
  }

  // Locate the dedicated missing-value bin, if any, and the regular bins.
  long missing_bin = -1;
  CharacterVector fitted_labels;
  bool has_labels = false;
  if (n_woe == num_bins + 1) {
    missing_bin = static_cast<long>(num_bins);  // default: the last entry
    if (obresults.containsElementNamed("bin")) {
      fitted_labels = as<CharacterVector>(obresults["bin"]);
      if (static_cast<size_t>(fitted_labels.size()) == n_woe) {
        has_labels = true;
        for (size_t j = 0; j < n_woe; ++j) {
          SEXP s = STRING_ELT(fitted_labels, static_cast<R_xlen_t>(j));
          if (s == NA_STRING) continue;
          const std::string lab(CHAR(s));
          if (lab == "NA" || lab == "Missing") {
            missing_bin = static_cast<long>(j);
            break;
          }
        }
      }
    }
  }
  std::vector<double> woe_values, bin_ids;
  woe_values.reserve(num_bins);
  bin_ids.reserve(num_bins);
  for (size_t j = 0; j < n_woe; ++j) {
    if (static_cast<long>(j) == missing_bin) continue;
    woe_values.push_back(woe_nv[static_cast<R_xlen_t>(j)]);
    bin_ids.push_back(id_nv[static_cast<R_xlen_t>(j)]);
  }

  // ID for missing values without a dedicated bin (number of bins + 1)
  const double missing_id = static_cast<double>(num_bins + 1);

  // Missing value codes, sorted for binary search (NA entries ignored)
  std::vector<double> missing_codes;
  if (missing_values.isNotNull()) {
    NumericVector mv = as<NumericVector>(missing_values);
    for (R_xlen_t i = 0; i < mv.size(); ++i) {
      if (!NumericVector::is_na(mv[i])) missing_codes.push_back(mv[i]);
    }
  } else {
    missing_codes.push_back(-999.0);
  }
  std::sort(missing_codes.begin(), missing_codes.end());
  missing_codes.erase(std::unique(missing_codes.begin(), missing_codes.end()),
                      missing_codes.end());

  const double NEG_INF = -std::numeric_limits<double>::infinity();
  const double POS_INF = std::numeric_limits<double>::infinity();

  auto format_number = [](double val) -> std::string {
    if (std::isinf(val)) {
      return val < 0 ? "-Inf" : "+Inf";
    }
    std::ostringstream oss;
    oss.precision(6);
    oss << std::noshowpoint << val;
    return oss.str();
  };

  // Interval labels, built once
  CharacterVector labels_out(static_cast<R_xlen_t>(num_bins));
  for (size_t i = 0; i < num_bins; ++i) {
    const double lower = (i == 0) ? NEG_INF : cutpoints[i - 1];
    const double upper = (i == num_bins - 1) ? POS_INF : cutpoints[i];
    std::ostringstream oss;
    if (include_upper_bound) {
      oss << "(" << format_number(lower) << ";" << format_number(upper) << "]";
    } else {
      oss << "[" << format_number(lower) << ";" << format_number(upper) << ")";
    }
    labels_out[static_cast<R_xlen_t>(i)] = oss.str();
  }

  // Scoring of missing values
  CharacterVector special_cv(1);
  double miss_woe = NA_REAL;
  double miss_id = missing_id;
  if (missing_bin >= 0) {
    miss_woe = woe_nv[missing_bin];
    miss_id = id_nv[missing_bin];
    SEXP s = has_labels ? STRING_ELT(fitted_labels, missing_bin) : NA_STRING;
    if (s == NA_STRING) {
      special_cv[0] = "NA";
    } else {
      SET_STRING_ELT(special_cv, 0, s);
    }
  } else {
    special_cv[0] = "Special";
  }
  SEXP special_sx = STRING_ELT(special_cv, 0);

  const R_xlen_t n = feature.size();
  NumericVector featurewoe(n);
  CharacterVector featurebins(n);
  NumericVector featureid(n);
  IntegerVector ismissing(n);
  NumericVector feature_values = clone(feature);

  const double* x_p = REAL(feature);
  double* out_woe = REAL(featurewoe);
  double* out_id = REAL(featureid);
  int* out_miss = INTEGER(ismissing);
  const double* cp_begin = cutpoints.data();
  const double* cp_end = cp_begin + cutpoints.size();
  const bool few_codes = missing_codes.size() <= 8;

  for (R_xlen_t i = 0; i < n; ++i) {
    const double x = x_p[i];
    bool is_missing = std::isnan(x);
    if (!is_missing && !missing_codes.empty()) {
      if (few_codes) {
        for (double c : missing_codes) {
          if (x == c) { is_missing = true; break; }
        }
      } else {
        is_missing = std::binary_search(missing_codes.begin(), missing_codes.end(), x);
      }
    }

    if (is_missing) {
      out_woe[i] = miss_woe;
      SET_STRING_ELT(featurebins, i, special_sx);
      out_id[i] = miss_id;
      out_miss[i] = 1;
      continue;
    }

    // (a, b]: x belongs to bin j = number of cutpoints strictly below x,
    //         i.e. the position of the first cutpoint >= x (lower_bound).
    // [a, b): x belongs to bin j = number of cutpoints <= x (upper_bound).
    const double* it = include_upper_bound ? std::lower_bound(cp_begin, cp_end, x)
                                           : std::upper_bound(cp_begin, cp_end, x);
    const size_t bin_index = static_cast<size_t>(it - cp_begin);  // <= num_bins - 1

    out_woe[i] = woe_values[bin_index];
    SET_STRING_ELT(featurebins, i, STRING_ELT(labels_out, static_cast<R_xlen_t>(bin_index)));
    out_id[i] = bin_ids[bin_index];
    out_miss[i] = 0;
  }

  return DataFrame::create(
    Named("feature") = feature_values,
    Named("bin") = featurebins,
    Named("woe") = featurewoe,
    Named("idbin") = featureid,
    Named("ismissing") = ismissing
  );
}
