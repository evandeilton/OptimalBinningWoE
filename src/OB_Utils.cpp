#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>
#include <set>

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
 NumericVector bin_ids   = binning_result["id"];
 CharacterVector bins    = binning_result["bin"];
 NumericVector counts    = binning_result["count"];
 NumericVector count_pos = binning_result["count_pos"];
 NumericVector count_neg = binning_result["count_neg"];
 
 int n0 = bin_ids.size();
 if (bins.size() != n0 || counts.size() != n0 ||
     count_pos.size() != n0 || count_neg.size() != n0) {
   stop("All vectors (id, bin, count, count_pos, count_neg) must have the same length.");
 }
 
 // Sort by id (bin order governs cumulatives and KS)
 IntegerVector idx = seq_len(n0) - 1;
 std::sort(idx.begin(), idx.end(),
           [&bin_ids](int i, int j){ return bin_ids[i] < bin_ids[j]; });
 
 bin_ids   = bin_ids[idx];
 bins      = bins[idx];
 counts    = counts[idx];
 count_pos = count_pos[idx];
 count_neg = count_neg[idx];
 
 // Totals as double (avoid truncation on large samples)
 const double total_count = sum(counts);
 const double total_pos   = sum(count_pos);
 const double total_neg   = sum(count_neg);
 
 const double overall_pos_rate = (total_count > 0.0) ? (total_pos / total_count) : NA_REAL;
 
 const int n = bins.size();
 
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
   double total_odds = (total_neg == 0.0) ? NA_REAL : (total_pos / total_neg);
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
 if ((int)target.size() != binned_df.nrows()) {
   stop("binned_df and target must have the same length.");
 }
 
 // Check binary target (0/1) and absence of NA
 std::set<double> uniq;
 for (int i = 0; i < target.size(); ++i) {
   if (NumericVector::is_na(target[i]))
     stop("target contains NA; please remove or impute first.");
   uniq.insert(target[i]);
 }
 if (uniq.size() != 2 || !uniq.count(0.0) || !uniq.count(1.0)) {
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
 
 // Aggregate by key (string) and retain idbin for stable ordering
 std::map<std::string, std::pair<int,int>> grp_counts; // pos,neg
 std::map<std::string, double> grp_id;
 
 for (int i = 0; i < binned_df.nrows(); ++i) {
   std::string key;
   if (group_var == "bin") {
     key = Rcpp::as<std::string>(feature_bins[i]);
   } else if (group_var == "woe") {
     key = std::to_string((double)feature_woe[i]);
   } else { // idbin
     key = std::to_string((double)feature_id[i]);
   }
   grp_id[key] = (double)feature_id[i];
   
   if (target[i] == 1.0) grp_counts[key].first += 1;
   else                  grp_counts[key].second += 1;
 }
 
 // Order by idbin (common practice in scorecards)
 std::vector<std::pair<double,std::string>> ordered;
 ordered.reserve(grp_counts.size());
 for (const auto& e : grp_counts) {
   ordered.push_back({grp_id[e.first], e.first});
 }
 std::sort(ordered.begin(), ordered.end(),
           [](const std::pair<double,std::string>& a,
              const std::pair<double,std::string>& b){
             return a.first < b.first;
           });
 
 const int m = (int)ordered.size();
 
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
   const std::string& key = ordered[i].second;
   bin_labels[i] = key;
   group_ids[i]  = ordered[i].first;
   
   count_pos[i] = grp_counts[key].first;
   count_neg[i] = grp_counts[key].second;
   counts[i]    = count_pos[i] + count_neg[i];
   
   total_pos += count_pos[i];
   total_neg += count_neg[i];
   total_count += counts[i];
 }
 
 const double overall_pos_rate = (total_count > 0.0) ? (total_pos / total_count) : NA_REAL;
 
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
   double total_odds = (total_neg == 0.0) ? NA_REAL : (total_pos / total_neg);
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