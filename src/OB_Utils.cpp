#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <limits>
#include <string>
#include <unordered_map>
#include <cstdio>

using namespace Rcpp;

// -----------------------------------------------------------------------------
// Helper: KL divergence term for Bernoulli components (p*log(p/q))
// Rules:
// - If p == 0, return 0 (standard convention).
// - If q == 0 and p > 0, return +Inf.
// - p and q must be in [0,1].
// -----------------------------------------------------------------------------
inline double kl_divergence(double p, double q) {
  if (p == 0.0) return 0.0;
  if (q == 0.0) return R_PosInf;
  return p * std::log(p / q);
}



// [[Rcpp::export]]
DataFrame OBGainsTable(List binning_result) {
 // Extract
 NumericVector bin_ids_in   = binning_result["id"];
 CharacterVector bins_in    = binning_result["bin"];
 NumericVector counts_in    = binning_result["count"];
 NumericVector count_pos_in = binning_result["count_pos"];
 NumericVector count_neg_in = binning_result["count_neg"];
 
 const R_xlen_t n0x = bin_ids_in.size();
 if (bins_in.size() != n0x || counts_in.size() != n0x ||
     count_pos_in.size() != n0x || count_neg_in.size() != n0x) {
   stop("All vectors (id, bin, count, count_pos, count_neg) must have the same length.");
 }
 if (n0x > static_cast<R_xlen_t>(std::numeric_limits<int>::max())) {
   stop("Too many bins.");
 }
 const int n0 = static_cast<int>(n0x);
 
 // Sort by id (bin order governs cumulatives and KS).
 //
 // The permuted columns are built explicitly from an index vector instead of
 // the self-assigning Rcpp subset `v = v[idx]`. A stable sort keeps rows that
 // share an id in input order (std::sort left that order unspecified), and a
 // missing id (NA/NaN) sorts last: a raw `<` on NaN is not a strict weak
 // ordering, which is undefined behaviour for std::sort and can walk off the
 // end of the index range.
 std::vector<int> ord(static_cast<size_t>(n0));
 for (int i = 0; i < n0; ++i) ord[static_cast<size_t>(i)] = i;
 {
   const double* id_ptr = REAL(bin_ids_in);
   std::stable_sort(ord.begin(), ord.end(), [id_ptr](int i, int j) {
     const double a = id_ptr[i], b = id_ptr[j];
     const bool na = std::isnan(a), nb = std::isnan(b);
     if (na || nb) return !na && nb;   // non-missing before missing
     return a < b;
   });
 }

 NumericVector bin_ids(n0), counts(n0), count_pos(n0), count_neg(n0);
 CharacterVector bins(n0);
 for (int i = 0; i < n0; ++i) {
   const int k = ord[static_cast<size_t>(i)];
   bin_ids[i]   = bin_ids_in[k];
   bins[i]      = bins_in[k];
   counts[i]    = counts_in[k];
   count_pos[i] = count_pos_in[k];
   count_neg[i] = count_neg_in[k];
 }
 
 // Totals as double (avoid truncation on large samples)
 const double total_count = sum(counts);
 const double total_pos   = sum(count_pos);
 const double total_neg   = sum(count_neg);
 
 const double overall_pos_rate = (total_count > 0.0) ? (total_pos / total_count) : NA_REAL;
 
 const int n = n0;
 // Loop invariant, hoisted out of the per-bin loop.
 const double total_odds = (total_neg == 0.0) ? NA_REAL : (total_pos / total_neg);
 
 NumericVector count_perc(n), cum_count_perc(n);
 NumericVector pos_rate(n), neg_rate(n);
 NumericVector pos_perc(n), neg_perc(n);
 NumericVector cum_pos(n), cum_neg(n);
 NumericVector cum_pos_perc(n), cum_neg_perc(n);
 NumericVector cum_pos_perc_total(n), cum_neg_perc_total(n);
 NumericVector ks(n), woe(n), iv(n);
 NumericVector odds_pos(n), odds_ratio(n), lift(n), gini_contribution(n);
 NumericVector precision(n), recall(n), f1_score(n);
 NumericVector log_likelihood(n);
 NumericVector kl_divergence_metric(n), js_divergence_metric(n);
 
 double total_iv = 0.0;
 double cpos = 0.0, cneg = 0.0;
 
 for (int i = 0; i < n; ++i) {
   // Count percentage and cumulative
   count_perc[i]     = (total_count > 0.0) ? (counts[i] / total_count) : NA_REAL;
   cum_count_perc[i] = (i == 0) ? count_perc[i] : (cum_count_perc[i-1] + count_perc[i]);
   
   // In-bin rates (guard count==0)
   if (counts[i] > 0.0) {
     pos_rate[i] = count_pos[i] / counts[i];
     neg_rate[i] = count_neg[i] / counts[i];
   } else {
     pos_rate[i] = NA_REAL;
     neg_rate[i] = NA_REAL;
   }
   
   // Class-conditional shares
   pos_perc[i] = (total_pos > 0.0) ? (count_pos[i] / total_pos) : NA_REAL; // P(X|Y=1)
   neg_perc[i] = (total_neg > 0.0) ? (count_neg[i] / total_neg) : NA_REAL; // P(X|Y=0)
   
   // Cumulatives (counts and percentages)
   cpos += count_pos[i];
   cneg += count_neg[i];
   cum_pos[i] = cpos;
   cum_neg[i] = cneg;
   
   cum_pos_perc[i] = (total_pos > 0.0) ? (cum_pos[i] / total_pos) : NA_REAL;
   cum_neg_perc[i] = (total_neg > 0.0) ? (cum_neg[i] / total_neg) : NA_REAL;
   
   cum_pos_perc_total[i] = (total_count > 0.0) ? (cum_pos[i] / total_count) : NA_REAL;
   cum_neg_perc_total[i] = (total_count > 0.0) ? (cum_neg[i] / total_count) : NA_REAL;
   
   // KS
   ks[i] = (R_finite(cum_pos_perc[i]) && R_finite(cum_neg_perc[i]))
     ? std::abs(cum_pos_perc[i] - cum_neg_perc[i]) : NA_REAL;
   
   // WoE/IV (set 0 when class-conditional share is 0 to avoid +/-Inf)
   if (R_finite(pos_perc[i]) && R_finite(neg_perc[i]) && pos_perc[i] > 0.0 && neg_perc[i] > 0.0) {
     woe[i] = std::log(pos_perc[i] / neg_perc[i]);
     iv[i]  = (pos_perc[i] - neg_perc[i]) * woe[i];
   } else {
     woe[i] = 0.0;
     iv[i]  = 0.0;
   }
   total_iv += iv[i];
   
   // Odds, OR, Lift
   odds_pos[i] = (count_neg[i] == 0.0) ? R_PosInf : (count_pos[i] / count_neg[i]);
   odds_ratio[i] = (R_finite(total_odds) && total_odds > 0.0) ? (odds_pos[i] / total_odds) : NA_REAL;
   
   lift[i] = (R_finite(overall_pos_rate) && overall_pos_rate > 0.0 && R_finite(pos_rate[i]))
     ? (pos_rate[i] / overall_pos_rate) : NA_REAL;
   
   // ***** SCIENTIFIC CORRECTION (sign) *****
   // Gini contribution per bin: P(X|Y=0) * F1(i) - P(X|Y=1) * F0(i)
   if (R_finite(pos_perc[i]) && R_finite(neg_perc[i]) &&
   R_finite(cum_pos_perc[i]) && R_finite(cum_neg_perc[i])) {
     gini_contribution[i] = neg_perc[i] * cum_pos_perc[i] - pos_perc[i] * cum_neg_perc[i];
   } else {
     gini_contribution[i] = NA_REAL;
   }
   
   // Precision (per bin) and Recall (cumulative TPs)
   precision[i] = (R_finite(pos_rate[i])) ? pos_rate[i] : NA_REAL;
   recall[i]    = (R_finite(cum_pos_perc[i])) ? cum_pos_perc[i] : NA_REAL;
   
   f1_score[i]  = (R_finite(precision[i]) && R_finite(recall[i]) &&
     (precision[i] + recall[i] > 0.0))
     ? (2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
     : 0.0;
   
   // NumericalBin-level Bernoulli log-likelihood
   if (R_finite(pos_rate[i]) && R_finite(neg_rate[i]) &&
       pos_rate[i] > 0.0 && neg_rate[i] > 0.0) {
     log_likelihood[i] =
       count_pos[i] * std::log(pos_rate[i]) + count_neg[i] * std::log(neg_rate[i]);
   } else {
     log_likelihood[i] = NA_REAL; // consistent behavior across functions
   }
   
   // KL(Bern(p_i) || Bern(p))
   double kl_pos = (R_finite(pos_rate[i]) && R_finite(overall_pos_rate))
     ? kl_divergence(pos_rate[i], overall_pos_rate) : NA_REAL;
   double kl_neg = (R_finite(neg_rate[i]) && R_finite(overall_pos_rate))
     ? kl_divergence(neg_rate[i], 1.0 - overall_pos_rate) : NA_REAL;
   kl_divergence_metric[i] =
   (R_finite(kl_pos) && R_finite(kl_neg)) ? (kl_pos + kl_neg) : NA_REAL;
   
   // Jensen–Shannon: JS = 1/2 KL(P||M) + 1/2 KL(Q||M), P=(p_i,1-p_i), Q=(p,1-p)
   if (R_finite(pos_rate[i]) && R_finite(neg_rate[i]) && R_finite(overall_pos_rate)) {
     double m_pos = (pos_rate[i] + overall_pos_rate) / 2.0;
     double m_neg = (neg_rate[i] + (1.0 - overall_pos_rate)) / 2.0;
     double kl_p_m = kl_divergence(pos_rate[i], m_pos) + kl_divergence(neg_rate[i], m_neg);
     double kl_q_m = kl_divergence(overall_pos_rate, m_pos) +
       kl_divergence(1.0 - overall_pos_rate, m_neg);
     js_divergence_metric[i] = 0.5 * (kl_p_m + kl_q_m);
   } else {
     js_divergence_metric[i] = NA_REAL;
   }
 }
 
 return DataFrame::create(
   Named("id") = bin_ids,
   Named("bin") = bins,
   Named("count") = counts,
   Named("pos") = count_pos,
   Named("neg") = count_neg,
   Named("woe") = woe,
   Named("iv") = iv,
   Named("total_iv") = rep(total_iv, n),
   Named("cum_pos") = cum_pos,
   Named("cum_neg") = cum_neg,
   Named("pos_rate") = pos_rate,
   Named("neg_rate") = neg_rate,
   Named("pos_perc") = pos_perc,
   Named("neg_perc") = neg_perc,
   Named("count_perc") = count_perc,
   Named("cum_count_perc") = cum_count_perc,
   Named("cum_pos_perc") = cum_pos_perc,
   Named("cum_neg_perc") = cum_neg_perc,
   Named("cum_pos_perc_total") = cum_pos_perc_total,
   Named("cum_neg_perc_total") = cum_neg_perc_total,
   Named("odds_pos") = odds_pos,
   Named("odds_ratio") = odds_ratio,
   Named("lift") = lift,
   Named("ks") = ks,
   Named("gini_contribution") = gini_contribution,
   Named("precision") = precision,
   Named("recall") = recall,
   Named("f1_score") = f1_score,
   Named("log_likelihood") = log_likelihood,
   Named("kl_divergence") = kl_divergence_metric,
   Named("js_divergence") = js_divergence_metric
 );
}


// [[Rcpp::export]]
DataFrame OBGainsTableFeature(DataFrame binned_df,
                             NumericVector target,
                             std::string group_var = "bin") {
 if (target.size() != static_cast<R_xlen_t>(binned_df.nrows())) {
   stop("binned_df and target must have the same length.");
 }
 
 // Check binary target (0/1) and absence of NA
 // (Same checks and messages as before, without a std::set insert per row.)
 bool has0 = false, has1 = false, other = false;
 for (R_xlen_t i = 0; i < target.size(); ++i) {
   const double t = target[i];
   if (NumericVector::is_na(t))
     stop("target contains NA; please remove or impute first.");
   if (t == 0.0) has0 = true;
   else if (t == 1.0) has1 = true;
   else other = true;
 }
 if (other || !has0 || !has1) {
   stop("target must contain only 0 and 1.");
 }
 
 if (!binned_df.containsElementNamed("feature") ||
     !binned_df.containsElementNamed("bin") ||
     !binned_df.containsElementNamed("woe") ||
     !binned_df.containsElementNamed("idbin")) {
     stop("binned_df must contain columns: feature, bin, woe, idbin.");
 }
 
 if (group_var != "bin" && group_var != "woe" && group_var != "idbin") {
   stop("group_var must be one of: 'bin', 'woe', or 'idbin'.");
 }
 
 CharacterVector feature_bins = binned_df["bin"];
 NumericVector   feature_woe  = binned_df["woe"];
 NumericVector   feature_id   = binned_df["idbin"];
 
 // Aggregate by key and retain, per group, the idbin of its LAST row (that is
 // what orders the groups).
 //
 // Grouping used to build a std::string key for every row and look it up in a
 // std::map, i.e. one allocation plus O(log k) string compares per row. It now
 // hashes one scalar per row and only materialises strings once per group.
 //
 // For group_var = "woe"/"idbin" the key used to be std::to_string(value),
 // which keeps just 6 decimals: two different WoE values agreeing to 1e-6
 // (routine for bins with close event rates) were silently pooled into one
 // group, corrupting every count and metric of the table. Groups are now
 // formed on the exact value; the label keeps the std::to_string() text unless
 // two distinct values would share it, in which case both are printed with 17
 // significant digits so the rows stay distinguishable.
 const int nrow = static_cast<int>(binned_df.nrows());
 const double* tgt = REAL(target);
 const double* idv = REAL(feature_id);

 struct Group { std::string label; double value; int last_row; int pos; int neg; };
 std::vector<Group> groups;

 if (group_var == "bin") {
   // Equal CHARSXPs are equal strings; distinct CHARSXPs can still hold the
   // same bytes (different declared encodings), so pointer groups are merged
   // on their text afterwards, exactly like the old string-keyed map.
   std::unordered_map<SEXP, int> by_ptr;
   std::vector<SEXP> ptr_of;
   std::vector<int> last_row, npos, nneg;
   for (int i = 0; i < nrow; ++i) {
     SEXP key = STRING_ELT(feature_bins, i);
     auto it = by_ptr.find(key);
     size_t g;
     if (it == by_ptr.end()) {
       g = ptr_of.size();
       by_ptr.emplace(key, static_cast<int>(g));
       ptr_of.push_back(key);
       last_row.push_back(i); npos.push_back(0); nneg.push_back(0);
     } else {
       g = static_cast<size_t>(it->second);
     }
     last_row[g] = i;
     if (tgt[i] == 1.0) ++npos[g];
     else               ++nneg[g];
   }
   std::map<std::string, size_t> by_text;   // text -> index into groups
   for (size_t g = 0; g < ptr_of.size(); ++g) {
     std::string txt = (ptr_of[g] == NA_STRING) ? std::string("NA")
                                                : std::string(CHAR(ptr_of[g]));
     auto it = by_text.find(txt);
     if (it == by_text.end()) {
       by_text.emplace(txt, groups.size());
       groups.push_back({txt, 0.0, last_row[g], npos[g], nneg[g]});
     } else {
       Group& G = groups[it->second];
       G.last_row = std::max(G.last_row, last_row[g]);
       G.pos += npos[g];
       G.neg += nneg[g];
     }
   }
 } else {
   const double* val = (group_var == "woe") ? REAL(feature_woe) : idv;
   // Exact-value key. All NaNs of one sign form one group (to_string gave
   // "nan"/"-nan"); +0 and -0 stay apart as they did.
   struct KeyHash {
     size_t operator()(const std::pair<int, double>& k) const {
       return std::hash<double>()(k.second) ^ (static_cast<size_t>(k.first) << 1);
     }
   };
   std::unordered_map<std::pair<int, double>, size_t, KeyHash> by_val;
   for (int i = 0; i < nrow; ++i) {
     const double v = val[i];
     std::pair<int, double> key;
     if (std::isnan(v))       key = {std::signbit(v) ? 1 : 2, 0.0};
     else if (v == 0.0)       key = {std::signbit(v) ? 3 : 4, 0.0};
     else                     key = {0, v};
     auto it = by_val.find(key);
     size_t g;
     if (it == by_val.end()) {
       g = groups.size();
       by_val.emplace(key, g);
       groups.push_back({std::to_string(v), v, i, 0, 0});
     } else {
       g = it->second;
     }
     Group& G = groups[g];
     G.last_row = i;
     if (tgt[i] == 1.0) ++G.pos;
     else               ++G.neg;
   }
   // Disambiguate labels that std::to_string() collapsed.
   std::map<std::string, std::vector<size_t>> by_label;
   for (size_t g = 0; g < groups.size(); ++g) by_label[groups[g].label].push_back(g);
   for (const auto& e : by_label) {
     if (e.second.size() < 2) continue;
     for (size_t g : e.second) {
       char buf[64];
       std::snprintf(buf, sizeof(buf), "%.17g", groups[g].value);
       groups[g].label = buf;
     }
   }
 }

 // Order: label (the old std::map iteration order), then a STABLE sort on the
 // group's idbin. Missing idbin sorts last (a raw `<` on NaN is not a strict
 // weak ordering, which is undefined behaviour for std::sort).
 std::vector<size_t> ordered(groups.size());
 for (size_t g = 0; g < groups.size(); ++g) ordered[g] = g;
 std::sort(ordered.begin(), ordered.end(), [&groups](size_t a, size_t b) {
   if (groups[a].label != groups[b].label) return groups[a].label < groups[b].label;
   return groups[a].value < groups[b].value;
 });
 std::stable_sort(ordered.begin(), ordered.end(), [&groups, idv](size_t a, size_t b) {
   const double ia = idv[groups[a].last_row], ib = idv[groups[b].last_row];
   const bool na = std::isnan(ia), nb = std::isnan(ib);
   if (na || nb) return !na && nb;
   return ia < ib;
 });

 const int m = static_cast<int>(ordered.size());
 
 CharacterVector bin_labels(m);
 NumericVector   group_ids(m), counts(m), count_pos(m), count_neg(m);
 NumericVector   count_perc(m), cum_count_perc(m);
 NumericVector   pos_rate(m), neg_rate(m);
 NumericVector   pos_perc(m), neg_perc(m);
 NumericVector   cum_pos(m), cum_neg(m);
 NumericVector   cum_pos_perc(m), cum_neg_perc(m);
 NumericVector   cum_pos_perc_total(m), cum_neg_perc_total(m);
 NumericVector   ks(m), woe(m), iv(m);
 NumericVector   odds_pos(m), odds_ratio(m), lift(m);
 NumericVector   gini_contribution(m);
 NumericVector   precision(m), recall(m), f1_score(m);
 NumericVector   log_likelihood(m);
 NumericVector   kl_divergence_metric(m), js_divergence_metric(m);
 
 double total_count = 0.0, total_pos = 0.0, total_neg = 0.0;
 
 for (int i = 0; i < m; ++i) {
   const Group& G = groups[ordered[static_cast<size_t>(i)]];
   bin_labels[i] = G.label;
   group_ids[i]  = idv[G.last_row];

   count_pos[i] = G.pos;
   count_neg[i] = G.neg;
   counts[i]    = count_pos[i] + count_neg[i];
   
   total_pos += count_pos[i];
   total_neg += count_neg[i];
   total_count += counts[i];
 }
 
 const double overall_pos_rate = (total_count > 0.0) ? (total_pos / total_count) : NA_REAL;
 
 const double total_odds = (total_neg == 0.0) ? NA_REAL : (total_pos / total_neg);
 double cpos = 0.0, cneg = 0.0;
 double total_iv = 0.0;
 
 for (int i = 0; i < m; ++i) {
   count_perc[i]     = (total_count > 0.0) ? (counts[i] / total_count) : NA_REAL;
   cum_count_perc[i] = (i == 0) ? count_perc[i] : (cum_count_perc[i-1] + count_perc[i]);
   
   if (counts[i] > 0.0) {
     pos_rate[i] = count_pos[i] / counts[i];
     neg_rate[i] = count_neg[i] / counts[i];
   } else {
     pos_rate[i] = NA_REAL;
     neg_rate[i] = NA_REAL;
   }
   
   pos_perc[i] = (total_pos > 0.0) ? (count_pos[i] / total_pos) : NA_REAL;
   neg_perc[i] = (total_neg > 0.0) ? (count_neg[i] / total_neg) : NA_REAL;
   
   cpos += count_pos[i];
   cneg += count_neg[i];
   cum_pos[i] = cpos;
   cum_neg[i] = cneg;
   
   cum_pos_perc[i] = (total_pos > 0.0) ? (cum_pos[i] / total_pos) : NA_REAL;
   cum_neg_perc[i] = (total_neg > 0.0) ? (cum_neg[i] / total_neg) : NA_REAL;
   
   cum_pos_perc_total[i] = (total_count > 0.0) ? (cum_pos[i] / total_count) : NA_REAL;
   cum_neg_perc_total[i] = (total_count > 0.0) ? (cum_neg[i] / total_count) : NA_REAL;
   
   ks[i] = (R_finite(cum_pos_perc[i]) && R_finite(cum_neg_perc[i]))
     ? std::abs(cum_pos_perc[i] - cum_neg_perc[i]) : NA_REAL;
   
   // WoE/IV
   if (R_finite(pos_perc[i]) && R_finite(neg_perc[i]) && pos_perc[i] > 0.0 && neg_perc[i] > 0.0) {
     woe[i] = std::log(pos_perc[i] / neg_perc[i]);
     iv[i]  = (pos_perc[i] - neg_perc[i]) * woe[i];
   } else {
     woe[i] = 0.0;
     iv[i]  = 0.0;
   }
   total_iv += iv[i];
   
   odds_pos[i] = (count_neg[i] == 0.0) ? R_PosInf : (count_pos[i] / count_neg[i]);
   odds_ratio[i] = (R_finite(total_odds) && total_odds > 0.0) ? (odds_pos[i] / total_odds) : NA_REAL;
   
   lift[i] = (R_finite(overall_pos_rate) && overall_pos_rate > 0.0 && R_finite(pos_rate[i]))
     ? (pos_rate[i] / overall_pos_rate) : NA_REAL;
   
   // ***** SCIENTIFIC CORRECTION (sign) *****
   if (R_finite(pos_perc[i]) && R_finite(neg_perc[i]) &&
   R_finite(cum_pos_perc[i]) && R_finite(cum_neg_perc[i])) {
     gini_contribution[i] = neg_perc[i] * cum_pos_perc[i] - pos_perc[i] * cum_neg_perc[i];
   } else {
     gini_contribution[i] = NA_REAL;
   }
   
   precision[i] = (R_finite(pos_rate[i])) ? pos_rate[i] : NA_REAL;
   recall[i]    = (R_finite(cum_pos_perc[i])) ? cum_pos_perc[i] : NA_REAL;
   
   f1_score[i]  = (R_finite(precision[i]) && R_finite(recall[i]) &&
     (precision[i] + recall[i] > 0.0))
     ? (2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
     : 0.0;
   
   if (R_finite(pos_rate[i]) && R_finite(neg_rate[i]) &&
       pos_rate[i] > 0.0 && neg_rate[i] > 0.0) {
     log_likelihood[i] =
       count_pos[i] * std::log(pos_rate[i]) + count_neg[i] * std::log(neg_rate[i]);
   } else {
     log_likelihood[i] = NA_REAL;
   }
   
   double kl_pos = (R_finite(pos_rate[i]) && R_finite(overall_pos_rate))
     ? kl_divergence(pos_rate[i], overall_pos_rate) : NA_REAL;
   double kl_neg = (R_finite(neg_rate[i]) && R_finite(overall_pos_rate))
     ? kl_divergence(neg_rate[i], 1.0 - overall_pos_rate) : NA_REAL;
   kl_divergence_metric[i] =
   (R_finite(kl_pos) && R_finite(kl_neg)) ? (kl_pos + kl_neg) : NA_REAL;
   
   if (R_finite(pos_rate[i]) && R_finite(neg_rate[i]) && R_finite(overall_pos_rate)) {
     double m_pos = (pos_rate[i] + overall_pos_rate)/2.0;
     double m_neg = (neg_rate[i] + (1.0 - overall_pos_rate))/2.0;
     double kl_p_m = kl_divergence(pos_rate[i], m_pos) + kl_divergence(neg_rate[i], m_neg);
     double kl_q_m = kl_divergence(overall_pos_rate, m_pos) +
       kl_divergence(1.0 - overall_pos_rate, m_neg);
     js_divergence_metric[i] = 0.5 * (kl_p_m + kl_q_m);
   } else {
     js_divergence_metric[i] = NA_REAL;
   }
 }
 
 return DataFrame::create(
   Named("bin") = bin_labels,
   Named("id")  = group_ids,
   Named("count") = counts,
   Named("pos") = count_pos,
   Named("neg") = count_neg,
   Named("woe") = woe,
   Named("iv") = iv,
   Named("total_iv") = rep(total_iv, m),
   Named("cum_pos") = cum_pos,
   Named("cum_neg") = cum_neg,
   Named("pos_rate") = pos_rate,
   Named("neg_rate") = neg_rate,
   Named("pos_perc") = pos_perc,
   Named("neg_perc") = neg_perc,
   Named("count_perc") = count_perc,
   Named("cum_count_perc") = cum_count_perc,
   Named("cum_pos_perc") = cum_pos_perc,
   Named("cum_neg_perc") = cum_neg_perc,
   Named("cum_pos_perc_total") = cum_pos_perc_total,
   Named("cum_neg_perc_total") = cum_neg_perc_total,
   Named("odds_pos") = odds_pos,
   Named("odds_ratio") = odds_ratio,
   Named("lift") = lift,
   Named("ks") = ks,
   Named("gini_contribution") = gini_contribution,
   Named("precision") = precision,
   Named("recall") = recall,
   Named("f1_score") = f1_score,
   Named("log_likelihood") = log_likelihood,
   Named("kl_divergence") = kl_divergence_metric,
   Named("js_divergence") = js_divergence_metric
 );
}