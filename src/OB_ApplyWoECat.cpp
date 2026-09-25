// [[Rcpp::plugins(cpp11)]]
#include <Rcpp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cctype>
#include <limits>

using namespace Rcpp;

namespace {

// Strip leading/trailing whitespace. The characters are classified as
// unsigned char: passing a negative char (any UTF-8 continuation byte) to
// std::isspace() is undefined behaviour. The old implementation also
// decremented s.begin() for an empty string, which is undefined as well --
// and "" is one of the default missing tokens, so that path was hit routinely.
std::string trim_ws(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Split on a (non-empty) delimiter string, keeping empty pieces, so a label
// such as "A%;%" yields {"A", ""}.
std::vector<std::string> split_on(const std::string& s, const std::string& delimiter) {
  std::vector<std::string> tokens;
  size_t pos = 0, prev = 0;
  while ((pos = s.find(delimiter, prev)) != std::string::npos) {
    tokens.push_back(s.substr(prev, pos - prev));
    prev = pos + delimiter.size();
  }
  tokens.push_back(s.substr(prev));
  return tokens;
}

// Outcome of scoring one distinct input string.
struct Assignment {
  int bin;        // index into the fitted bins, or -1 for "Special"
  int missing;    // value of the 'ismissing' output column
};

} // namespace

// Scores a categorical feature with a fitted binning.
//
// Hot path: every distinct CHARSXP of 'feature' is resolved once and cached by
// pointer (R interns strings, so equal strings in one encoding share one
// CHARSXP), so the per-row cost is a single pointer-hash lookup regardless of
// the number of categories: O(n + total label length). Output labels are
// copied as CHARSXPs instead of being re-created from std::string per row.
//
// Matching rules, in order, for a non-missing value v:
//   1. v equals a category of a bin exactly (byte for byte, as obwoe_sql() and
//      obwoe_apply() match) -> that bin. When a category appears in more than
//      one bin label the FIRST bin wins, which is exactly what the CASE
//      expression emitted by obwoe_sql() does.
//   2. trimws(v) is a missing token -> the missing-value bin (below), or
//      "Special" when the fit has none.
//   3. trimws(v) equals a trimmed category -> that bin (backward compatible
//      with the historical, whitespace-insensitive matching).
//   4. v contains 'bin_separator' and is a run of consecutive pieces of a bin
//      label -> the first such bin (a category whose own name contains the
//      separator, which splitting the label cannot recover).
//   5. otherwise "Special" (unseen category).
// NA inputs go to the missing-value bin. The missing-value bin is the first
// bin with a category among 'missing_values' (default "NA", "Missing", ""):
// the categorical binners store NA as the category "NA", and this is the
// rule obwoe_sql(null_to_na_bin = TRUE) and obwoe_apply() use as well.
// [[Rcpp::export]]
DataFrame OBApplyWoECat(const List& obresults,
                        const CharacterVector& feature,
                        const std::string& bin_separator = "%;%",
                        Nullable<CharacterVector> missing_values = R_NilValue) {

  // Validate input parameters
  if (!obresults.containsElementNamed("bin")) {
    stop("The 'obresults' list must contain a 'bin' element.");
  }
  if (!obresults.containsElementNamed("woe")) {
    stop("The 'obresults' list must contain a 'woe' element.");
  }
  if (!obresults.containsElementNamed("id")) {
    stop("The 'obresults' list must contain an 'id' element.");
  }
  if (bin_separator.empty()) {
    stop("The 'bin_separator' must be a non-empty string.");
  }

  CharacterVector bins_cv = obresults["bin"];
  NumericVector woe_nv = obresults["woe"];
  NumericVector id_nv = obresults["id"];

  if (any(is_na(woe_nv))) {
    stop("WoE values cannot contain NA values.");
  }
  if (any(is_na(id_nv))) {
    stop("ID values cannot contain NA values.");
  }

  const R_xlen_t num_bins = bins_cv.size();
  if (woe_nv.size() != num_bins || id_nv.size() != num_bins) {
    stop("The number of bins must match the number of WoE values and IDs.");
  }
  if (num_bins > static_cast<R_xlen_t>(std::numeric_limits<int>::max() - 1)) {
    stop("Too many bins.");
  }

  // ID for values that match no bin (number of bins + 1)
  const double missing_id = static_cast<double>(num_bins + 1);

  // Missing tokens
  std::unordered_set<std::string> missing_set;
  if (missing_values.isNotNull()) {
    CharacterVector mv = as<CharacterVector>(missing_values);
    for (R_xlen_t i = 0; i < mv.size(); ++i) {
      if (!CharacterVector::is_na(mv[i])) {
        missing_set.insert(as<std::string>(mv[i]));
      }
    }
  } else {
    missing_set.insert("NA");
    missing_set.insert("Missing");
    missing_set.insert("");
  }

  // Output labels (an NA label is emitted as the string "NA", as before) and
  // category -> bin maps. Protected by being elements of 'labels_out'.
  CharacterVector labels_out(num_bins);
  std::unordered_map<std::string, int> exact_map;
  std::unordered_map<std::string, int> trimmed_map;
  exact_map.reserve(static_cast<size_t>(num_bins) * 2);
  trimmed_map.reserve(static_cast<size_t>(num_bins) * 2);
  int na_bin = -1;
  std::vector<std::string> label_strings;
  label_strings.reserve(static_cast<size_t>(num_bins));

  for (R_xlen_t b = 0; b < num_bins; ++b) {
    SEXP lab = STRING_ELT(bins_cv, b);
    std::string label;
    if (lab == NA_STRING) {
      label = "NA";
      SET_STRING_ELT(labels_out, b, Rf_mkChar("NA"));
    } else {
      label = CHAR(lab);
      SET_STRING_ELT(labels_out, b, lab);
    }
    const int bi = static_cast<int>(b);
    for (const std::string& part : split_on(label, bin_separator)) {
      std::string tpart = trim_ws(part);
      exact_map.emplace(part, bi);      // first bin wins
      if (na_bin < 0 && (missing_set.count(part) || missing_set.count(tpart))) {
        na_bin = bi;
      }
      trimmed_map.emplace(std::move(tpart), bi);
    }
    label_strings.push_back(std::move(label));
  }

  auto resolve = [&](SEXP cs) -> Assignment {
    const std::string v(CHAR(cs));
    auto it = exact_map.find(v);
    const std::string t = trim_ws(v);
    const bool is_missing_token = missing_set.count(t) > 0;
    if (it != exact_map.end()) {
      return Assignment{it->second, is_missing_token ? 1 : 0};
    }
    if (is_missing_token) {
      return Assignment{na_bin, 1};
    }
    auto it2 = trimmed_map.find(t);
    if (it2 != trimmed_map.end()) {
      return Assignment{it2->second, 0};
    }
    if (v.find(bin_separator) != std::string::npos) {
      // A category whose own name contains the separator: it is a run of
      // consecutive pieces of the label of the bin that holds it.
      const std::string needle = bin_separator + v + bin_separator;
      for (size_t b = 0; b < label_strings.size(); ++b) {
        const std::string hay = bin_separator + label_strings[b] + bin_separator;
        if (hay.find(needle) != std::string::npos) {
          return Assignment{static_cast<int>(b), 0};
        }
      }
    }
    return Assignment{-1, 1};
  };

  // Prepare output vectors
  const R_xlen_t n = feature.size();
  CharacterVector featurebins(n);
  NumericVector featurewoe(n);
  NumericVector featureid(n);
  IntegerVector ismissing(n);
  CharacterVector feature_values = clone(feature);
  CharacterVector special_cv = CharacterVector::create("Special");
  SEXP special_sx = STRING_ELT(special_cv, 0);

  const double* woe_p = REAL(woe_nv);
  const double* id_p = REAL(id_nv);
  double* out_woe = REAL(featurewoe);
  double* out_id = REAL(featureid);
  int* out_miss = INTEGER(ismissing);

  std::unordered_map<SEXP, Assignment> cache;
  const Assignment na_assign{na_bin, 1};

  for (R_xlen_t i = 0; i < n; ++i) {
    SEXP cs = STRING_ELT(feature, i);
    Assignment a;
    if (cs == NA_STRING) {
      a = na_assign;
    } else {
      auto it = cache.find(cs);
      if (it == cache.end()) {
        a = resolve(cs);
        cache.emplace(cs, a);
      } else {
        a = it->second;
      }
    }
    if (a.bin >= 0) {
      SET_STRING_ELT(featurebins, i, STRING_ELT(labels_out, a.bin));
      out_woe[i] = woe_p[a.bin];
      out_id[i] = id_p[a.bin];
    } else {
      SET_STRING_ELT(featurebins, i, special_sx);
      out_woe[i] = NA_REAL;
      out_id[i] = missing_id;
    }
    out_miss[i] = a.missing;
  }

  return DataFrame::create(
    Named("feature") = feature_values,
    Named("bin") = featurebins,
    Named("woe") = featurewoe,
    Named("idbin") = featureid,
    Named("ismissing") = ismissing
  );
}
