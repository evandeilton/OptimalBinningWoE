#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <sstream>
#include <iomanip>

using namespace Rcpp;

namespace {

// WoE and IV per bin. A zero share is floored at 1e-4 to avoid log(0).
void calculate_woe_iv(const std::vector<int>& count_pos, const std::vector<int>& count_neg,
                      std::vector<double>& woe, std::vector<double>& iv,
                      int total_pos, int total_neg) {
  for (size_t i = 0; i < count_pos.size(); ++i) {
    double pos_rate = static_cast<double>(count_pos[i]) / total_pos;
    double neg_rate = static_cast<double>(count_neg[i]) / total_neg;

    if (pos_rate == 0) pos_rate = 0.0001;
    if (neg_rate == 0) neg_rate = 0.0001;

    woe[i] = std::log(pos_rate / neg_rate);
    iv[i] = (pos_rate - neg_rate) * woe[i];
  }
}

// Binary target check shared by both entry points: same length as the
// feature, only 0/1 (no NA), and both classes present -- with one class
// absent every WoE is log(0/0) = NaN.
void validate_target(const IntegerVector& target, R_xlen_t n) {
  if (target.size() != n) {
    stop("'feature' and 'target' must have the same length.");
  }
  bool has0 = false, has1 = false;
  for (R_xlen_t i = 0; i < n; ++i) {
    const int t = target[i];
    if (t == 0) has0 = true;
    else if (t == 1) has1 = true;
    else stop("'target' must contain only 0 and 1 (no missing values).");
  }
  if (!has0 || !has1) {
    stop("'target' must contain both 0 and 1.");
  }
}

// Bin label, right-closed (lower, upper], matching ob_apply_woe_num()'s
// default include_upper_bound = TRUE and obwoe()/obwoe_apply().
std::string format_bin_range(double lower, double upper) {
  // Infinite bounds print as -Inf/+Inf wherever they occur (an infinite
  // cutpoint used to print as "inf"/"-inf").
  auto fmt = [](double v) -> std::string {
    if (std::isinf(v)) return v < 0 ? "-Inf" : "+Inf";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << v;
    return oss.str();
  };
  return "(" + fmt(lower) + ";" + fmt(upper) + "]";
}

} // namespace

// [[Rcpp::export]]
List binning_numerical_cutpoints(NumericVector feature, IntegerVector target,
                                 NumericVector cutpoints) {
  const R_xlen_t n = feature.size();
  validate_target(target, n);

  // Work on a sorted COPY: std::sort() on the NumericVector itself used to
  // reorder the caller's own R vector in place (as.numeric() on a double
  // vector does not copy).
  std::vector<double> cuts(cutpoints.begin(), cutpoints.end());
  for (double c : cuts) {
    if (std::isnan(c)) stop("'cutpoints' cannot contain NA or NaN values.");
  }
  std::sort(cuts.begin(), cuts.end());
  if (cuts.size() >= static_cast<size_t>(std::numeric_limits<int>::max())) {
    stop("Too many cutpoints.");
  }
  const int num_bins = static_cast<int>(cuts.size()) + 1;

  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  int total_pos = 0, total_neg = 0;

  // Right-closed assignment: the bin of x is the number of cutpoints strictly
  // below x, i.e. the position of the first cutpoint >= x. Missing feature
  // values (NA/NaN) are not binned -- they used to fall silently into the
  // first bin -- and get a WoE of NA, as in ob_apply_woe_num().
  std::vector<int> bin_of(static_cast<size_t>(n), -1);
  const double* x = REAL(feature);
  for (R_xlen_t i = 0; i < n; ++i) {
    if (std::isnan(x[i])) continue;
    const int b = static_cast<int>(std::lower_bound(cuts.begin(), cuts.end(), x[i]) - cuts.begin());
    bin_of[static_cast<size_t>(i)] = b;
    count[b]++;
    if (target[i] == 1) {
      count_pos[b]++;
      total_pos++;
    } else {
      count_neg[b]++;
      total_neg++;
    }
  }
  if (total_pos == 0 || total_neg == 0) {
    stop("Both target classes must be present among the non-missing feature values.");
  }

  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);

  CharacterVector bin_ranges(num_bins);
  NumericVector ids(num_bins);
  const double NEG_INF = -std::numeric_limits<double>::infinity();
  const double POS_INF = std::numeric_limits<double>::infinity();
  for (int i = 0; i < num_bins; ++i) {
    const double lo = (i == 0) ? NEG_INF : cuts[static_cast<size_t>(i - 1)];
    const double hi = (i == num_bins - 1) ? POS_INF : cuts[static_cast<size_t>(i)];
    bin_ranges[i] = format_bin_range(lo, hi);
    ids[i] = i + 1;
  }

  DataFrame woebin = DataFrame::create(
    Named("id") = ids,
    Named("bin") = bin_ranges,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );

  NumericVector woefeature(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const int b = bin_of[static_cast<size_t>(i)];
    woefeature[i] = (b < 0) ? NA_REAL : woe[static_cast<size_t>(b)];
  }

  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin,
    Named("cutpoints") = NumericVector(cuts.begin(), cuts.end()),
    Named("id") = ids
  );
}

// [[Rcpp::export]]
List binning_categorical_cutpoints(CharacterVector feature, IntegerVector target,
                                   CharacterVector cutpoints) {
  const R_xlen_t n = feature.size();
  validate_target(target, n);

  const R_xlen_t nb = cutpoints.size();
  if (nb == 0) {
    stop("'cutpoints' must define at least one bin.");
  }
  if (nb >= static_cast<R_xlen_t>(std::numeric_limits<int>::max())) {
    stop("Too many bins.");
  }
  const int num_bins = static_cast<int>(nb);

  // Input cutpoints group categories with "+" (e.g. "A+B"); the emitted 'bin'
  // labels join them with "%;%", the separator ob_apply_woe_cat() defaults to.
  std::unordered_map<std::string, int> category_to_bin;
  CharacterVector bin_labels(num_bins);
  for (int i = 0; i < num_bins; ++i) {
    if (CharacterVector::is_na(cutpoints[i])) {
      stop("'cutpoints' cannot contain NA values.");
    }
    std::string bin_categories = as<std::string>(cutpoints[i]);
    std::vector<std::string> parts;
    size_t start = 0, pos;
    while ((pos = bin_categories.find('+', start)) != std::string::npos) {
      parts.push_back(bin_categories.substr(start, pos - start));
      start = pos + 1;
    }
    parts.push_back(bin_categories.substr(start));

    std::string joined;
    for (size_t j = 0; j < parts.size(); ++j) {
      // A category listed in two bins would get one bin's WoE here and,
      // through ob_apply_woe_cat()/obwoe_sql() (first match wins), possibly
      // the other's: reject the ambiguous specification.
      auto ins = category_to_bin.emplace(parts[j], i);
      if (!ins.second && ins.first->second != i) {
        stop("Category '" + parts[j] + "' appears in more than one bin of 'cutpoints'.");
      }
      if (j > 0) joined += "%;%";
      joined += parts[j];
    }
    bin_labels[i] = joined;
  }

  std::vector<int> count(num_bins, 0);
  std::vector<int> count_pos(num_bins, 0);
  std::vector<int> count_neg(num_bins, 0);
  std::vector<double> woe(num_bins);
  std::vector<double> iv(num_bins);
  int total_pos = 0, total_neg = 0;

  // Every category must belong to a bin. They used to be inserted on the fly
  // by map::operator[] with bin index 0, silently inflating the first bin.
  // NA is matched as the category "NA", the token every ob_categorical_*()
  // wrapper uses for missing values. Each distinct CHARSXP is resolved once.
  std::unordered_map<SEXP, int> cache;
  std::vector<int> bin_of(static_cast<size_t>(n));
  for (R_xlen_t i = 0; i < n; ++i) {
    SEXP cs = STRING_ELT(feature, i);
    int b;
    auto it = cache.find(cs);
    if (it != cache.end()) {
      b = it->second;
    } else {
      const std::string cat = (cs == NA_STRING) ? std::string("NA") : std::string(CHAR(cs));
      auto jt = category_to_bin.find(cat);
      if (jt == category_to_bin.end()) {
        stop("Category '" + cat + "' of 'feature' is not listed in any bin of 'cutpoints'.");
      }
      b = jt->second;
      cache.emplace(cs, b);
    }
    bin_of[static_cast<size_t>(i)] = b;
    count[b]++;
    if (target[i] == 1) {
      count_pos[b]++;
      total_pos++;
    } else {
      count_neg[b]++;
      total_neg++;
    }
  }

  calculate_woe_iv(count_pos, count_neg, woe, iv, total_pos, total_neg);

  NumericVector ids(num_bins);
  for (int i = 0; i < num_bins; ++i) ids[i] = i + 1;

  DataFrame woebin = DataFrame::create(
    Named("id") = ids,
    Named("bin") = bin_labels,
    Named("count") = count,
    Named("count_pos") = count_pos,
    Named("count_neg") = count_neg,
    Named("woe") = woe,
    Named("iv") = iv
  );

  NumericVector woefeature(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    woefeature[i] = woe[static_cast<size_t>(bin_of[static_cast<size_t>(i)])];
  }

  return List::create(
    Named("woefeature") = woefeature,
    Named("woebin") = woebin
  );
}
