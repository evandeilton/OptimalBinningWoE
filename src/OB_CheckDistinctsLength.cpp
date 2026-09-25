#include <Rcpp.h>

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

// Distinct (value, target) bookkeeping for one key type: the number of
// distinct values, and the largest number of distinct target classes seen
// for any single value.
template <typename Key>
struct DistinctCounter {
  std::unordered_map<Key, std::vector<int>> classes;

  void add(const Key& key, int target_val) {
    std::vector<int>& cls = classes[key];
    if (std::find(cls.begin(), cls.end(), target_val) == cls.end()) {
      cls.push_back(target_val);
    }
  }

  Rcpp::IntegerVector result() const {
    int max_classes = 0;
    for (const auto& kv : classes) {
      max_classes = std::max(max_classes, static_cast<int>(kv.second.size()));
    }
    return Rcpp::IntegerVector::create(static_cast<int>(classes.size()), max_classes);
  }
};

// Canonical bit pattern of a finite double, so that 0 and -0 are one value
// (as in R's unique()) and equal values hash equally.
inline std::uint64_t double_key(double v) {
  if (v == 0.0) v = 0.0;
  std::uint64_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits;
}

// Number of distinct non-missing values of x (NA, NaN and +-Inf excluded, as
// documented) and the largest number of distinct target classes observed for
// a single value. Rows whose target is NA are ignored.
//
// Numeric values are compared exactly. They used to be keyed by
// std::to_string(), which prints six decimals, so e.g. 1e-7, 2e-7 and 3e-7
// all collapsed into "0.000000" and -0 counted apart from 0.
Rcpp::IntegerVector create_frequency_table(SEXP x, const Rcpp::IntegerVector& target) {
  if (Rf_isNull(x) || target.size() == 0) {
    Rcpp::warning("Input is null or empty");
    return Rcpp::IntegerVector::create(0, 0);
  }

  const R_xlen_t x_len = Rf_xlength(x);
  if (x_len != target.size()) {
    Rcpp::stop("Length of feature and target do not match");
  }
  const int* tg = INTEGER(target);

  switch (TYPEOF(x)) {
  case INTSXP:
  case LGLSXP: {
    const int* v = (TYPEOF(x) == INTSXP) ? INTEGER(x) : LOGICAL(x);
    const int na = (TYPEOF(x) == INTSXP) ? NA_INTEGER : NA_LOGICAL;
    DistinctCounter<int> dc;
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (v[i] != na && tg[i] != NA_INTEGER) dc.add(v[i], tg[i]);
    }
    return dc.result();
  }
  case REALSXP: {
    const double* v = REAL(x);
    DistinctCounter<std::uint64_t> dc;
    for (R_xlen_t i = 0; i < x_len; ++i) {
      if (std::isfinite(v[i]) && tg[i] != NA_INTEGER) dc.add(double_key(v[i]), tg[i]);
    }
    return dc.result();
  }
  case STRSXP: {
    // Resolve each distinct CHARSXP once; strings equal byte for byte but
    // stored under different encodings still count as one value.
    std::unordered_map<SEXP, size_t> ptr_index;
    std::unordered_map<std::string, size_t> str_index;
    std::vector<std::vector<int>> classes;
    for (R_xlen_t i = 0; i < x_len; ++i) {
      SEXP cs = STRING_ELT(x, i);
      if (cs == NA_STRING || tg[i] == NA_INTEGER) continue;
      size_t idx;
      auto it = ptr_index.find(cs);
      if (it != ptr_index.end()) {
        idx = it->second;
      } else {
        auto ins = str_index.emplace(std::string(CHAR(cs)), classes.size());
        if (ins.second) classes.emplace_back();
        idx = ins.first->second;
        ptr_index.emplace(cs, idx);
      }
      std::vector<int>& cls = classes[idx];
      if (std::find(cls.begin(), cls.end(), tg[i]) == cls.end()) cls.push_back(tg[i]);
    }
    int max_classes = 0;
    for (const auto& cls : classes) max_classes = std::max(max_classes, static_cast<int>(cls.size()));
    return Rcpp::IntegerVector::create(static_cast<int>(classes.size()), max_classes);
  }
  default:
    Rcpp::stop("Unsupported type");
  }
}

} // namespace

// [[Rcpp::export]]
Rcpp::IntegerVector OBCheckDistinctsLength(SEXP x, Rcpp::IntegerVector target) {
  try {
    return create_frequency_table(x, target);
  } catch (std::exception& e) {
    Rcpp::stop("Error in OBCheckDistinctsLength: %s", e.what());
  } catch (...) {
    Rcpp::stop("Unknown error in OBCheckDistinctsLength");
  }
}
