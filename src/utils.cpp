#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>

using namespace Rcpp;

// Helper function to calculate KL divergence
double kl_divergence(double p, double q) {
  if (p == 0) return 0;
  if (q == 0) return R_PosInf;
  return p * std::log(p / q);
}

//' Generates a Comprehensive Gains Table from Optimal Binning Results
//'
//' This function takes the result of the optimal binning process and generates a detailed gains table.
//' The table includes various metrics to assess the performance and characteristics of each bin.
//'
//' @param binning_result A list containing the binning results, which must include a data frame with
//' the following columns: "bin", "count", "count_pos", "count_neg", and "woe".
//'
//' @return A data frame containing the following columns for each bin:
//' \itemize{
//'   \item \code{bin}: The bin labels.
//'   \item \code{count}: Total count of observations in the bin.
//'   \item \code{pos}: Count of positive events in the bin.
//'   \item \code{neg}: Count of negative events in the bin.
//'   \item \code{woe}: Weight of Evidence (WoE) for the bin.
//'   \item \code{iv}: Information Value (IV) contribution for the bin.
//'   \item \code{total_iv}: Total Information Value (IV) across all bins.
//'   \item \code{cum_pos}: Cumulative count of positive events up to the current bin.
//'   \item \code{cum_neg}: Cumulative count of negative events up to the current bin.
//'   \item \code{pos_rate}: Rate of positive events within the bin.
//'   \item \code{neg_rate}: Rate of negative events within the bin.
//'   \item \code{pos_perc}: Percentage of positive events relative to the total positive events.
//'   \item \code{neg_perc}: Percentage of negative events relative to the total negative events.
//'   \item \code{count_perc}: Percentage of total observations in the bin.
//'   \item \code{cum_count_perc}: Cumulative percentage of observations up to the current bin.
//'   \item \code{cum_pos_perc}: Cumulative percentage of positive events up to the current bin.
//'   \item \code{cum_neg_perc}: Cumulative percentage of negative events up to the current bin.
//'   \item \code{cum_pos_perc_total}: Cumulative percentage of positive events relative to total observations.
//'   \item \code{cum_neg_perc_total}: Cumulative percentage of negative events relative to total observations.
//'   \item \code{odds_pos}: Odds of positive events in the bin.
//'   \item \code{odds_ratio}: Odds ratio of positive events compared to the total population.
//'   \item \code{lift}: Lift of the bin, calculated as the ratio of the positive rate in the bin to the overall positive rate.
//'   \item \code{ks}: Kolmogorov-Smirnov statistic, measuring the difference between cumulative positive and negative percentages.
//'   \item \code{gini_contribution}: Contribution to the Gini coefficient for each bin.
//'   \item \code{precision}: Precision of the bin.
//'   \item \code{recall}: Recall up to the current bin.
//'   \item \code{f1_score}: F1 score for the bin.
//'   \item \code{log_likelihood}: Log-likelihood of the bin.
//'   \item \code{kl_divergence}: Kullback-Leibler divergence for the bin.
//'   \item \code{js_divergence}: Jensen-Shannon divergence for the bin.
//' }
//'
//' @details
//' The function calculates various metrics for each bin:
//'
//' \itemize{
//'   \item Weight of Evidence (WoE): \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//'   \item Information Value (IV): \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'   \item Kolmogorov-Smirnov (KS) statistic: \deqn{KS_i = |F_1(i) - F_0(i)|}
//'     where \eqn{F_1(i)} and \eqn{F_0(i)} are the cumulative distribution functions for positive and negative classes.
//'   \item Odds Ratio: \deqn{OR_i = \frac{P(Y=1|X_i) / P(Y=0|X_i)}{P(Y=1) / P(Y=0)}}
//'   \item Lift: \deqn{Lift_i = \frac{P(Y=1|X_i)}{P(Y=1)}}
//'   \item Gini Contribution: \deqn{Gini_i = P(X_i|Y=1) \times F_0(i) - P(X_i|Y=0) \times F_1(i)}
//'   \item Precision: \deqn{Precision_i = \frac{TP_i}{TP_i + FP_i}}
//'   \item Recall: \deqn{Recall_i = \frac{\sum_{j=1}^i TP_j}{\sum_{j=1}^n TP_j}}
//'   \item F1 Score: \deqn{F1_i = 2 \times \frac{Precision_i \times Recall_i}{Precision_i + Recall_i}}
//'   \item Log-likelihood: \deqn{LL_i = n_{1i} \ln(p_i) + n_{0i} \ln(1-p_i)}
//'     where \eqn{n_{1i}} and \eqn{n_{0i}} are the counts of positive and negative cases in bin i,
//'     and \eqn{p_i} is the proportion of positive cases in bin i.
//'   \item Kullback-Leibler (KL) Divergence: \deqn{KL_i = p_i \ln\left(\frac{p_i}{p}\right) + (1-p_i) \ln\left(\frac{1-p_i}{1-p}\right)}
//'     where \eqn{p_i} is the proportion of positive cases in bin i and \eqn{p} is the overall proportion of positive cases.
//'   \item Jensen-Shannon (JS) Divergence: \deqn{JS_i = \frac{1}{2}KL(p_i || m) + \frac{1}{2}KL(q_i || m)}
//'     where \eqn{m = \frac{1}{2}(p_i + p)}, \eqn{p_i} is the proportion of positive cases in bin i,
//'     and \eqn{p} is the overall proportion of positive cases.
//' }
//'
//' @references
//' \itemize{
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
//'   \item Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 45(2), 171-186.
//'   \item Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. The Annals of Mathematical Statistics, 22(1), 79-86.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @examples
//' \dontrun{
//' binning_result <- OptimalBinning(target, feature)
//' gains_table <- OptimalBinningGainsTable(binning_result)
//' print(gains_table)
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTable(List binning_result) {
 DataFrame bin = as<DataFrame>(binning_result["woebin"]);
 
 CharacterVector bin_labels = bin["bin"];
 NumericVector counts = bin["count"];
 NumericVector count_pos = bin["count_pos"];
 NumericVector count_neg = bin["count_neg"];
 NumericVector woe = bin["woe"];
 
 int total_count = sum(counts);
 int total_pos = sum(count_pos);
 int total_neg = sum(count_neg);
 double overall_pos_rate = (double)total_pos / total_count;
 
 int n = bin_labels.size();
 NumericVector count_perc(n), cum_count_perc(n);
 NumericVector pos_rate(n), neg_rate(n);
 NumericVector pos_perc(n), neg_perc(n);
 NumericVector cum_pos(n), cum_neg(n);
 NumericVector cum_pos_perc(n), cum_neg_perc(n);
 NumericVector cum_pos_perc_total(n), cum_neg_perc_total(n);
 NumericVector ks(n), iv(n);
 NumericVector odds_pos(n), odds_ratio(n), lift(n), gini_contribution(n);
 NumericVector precision(n), recall(n), f1_score(n);
 NumericVector log_likelihood(n);
 NumericVector kl_divergence_metric(n), js_divergence_metric(n);
 
 double total_iv = 0.0;
 double cum_pos_sum = 0.0, cum_neg_sum = 0.0;
 
 for (int i = 0; i < n; i++) {
   count_perc[i] = (double)counts[i] / total_count;
   cum_count_perc[i] = (i == 0) ? count_perc[i] : cum_count_perc[i - 1] + count_perc[i];
   
   pos_rate[i] = (double)count_pos[i] / counts[i];
   neg_rate[i] = (double)count_neg[i] / counts[i];
   
   pos_perc[i] = (double)count_pos[i] / total_pos;
   neg_perc[i] = (double)count_neg[i] / total_neg;
   
   cum_pos_sum += count_pos[i];
   cum_neg_sum += count_neg[i];
   cum_pos[i] = cum_pos_sum;
   cum_neg[i] = cum_neg_sum;
   
   cum_pos_perc[i] = cum_pos[i] / total_pos;
   cum_neg_perc[i] = cum_neg[i] / total_neg;
   
   cum_pos_perc_total[i] = cum_pos[i] / total_count;
   cum_neg_perc_total[i] = cum_neg[i] / total_count;
   
   ks[i] = std::abs(cum_pos_perc[i] - cum_neg_perc[i]);
   
   iv[i] = (pos_rate[i] - neg_rate[i]) * woe[i];
   total_iv += iv[i];
   
   odds_pos[i] = (count_neg[i] == 0) ? R_PosInf : (double)count_pos[i] / count_neg[i];
   
   double total_odds = (double)total_pos / total_neg;
   odds_ratio[i] = odds_pos[i] / total_odds;
   
   lift[i] = pos_rate[i] / overall_pos_rate;
   
   gini_contribution[i] = pos_perc[i] * cum_neg_perc[i] - neg_perc[i] * cum_pos_perc[i];
   
   precision[i] = (double)count_pos[i] / counts[i];
   recall[i] = cum_pos_perc[i];
   f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]);
   
   log_likelihood[i] = count_pos[i] * std::log(pos_rate[i]) + count_neg[i] * std::log(neg_rate[i]);
   
   double kl_pos = kl_divergence(pos_rate[i], overall_pos_rate);
   double kl_neg = kl_divergence(neg_rate[i], 1 - overall_pos_rate);
   kl_divergence_metric[i] = kl_pos + kl_neg;
   
   double m_pos = (pos_rate[i] + overall_pos_rate) / 2;
   double m_neg = (neg_rate[i] + (1 - overall_pos_rate)) / 2;
   js_divergence_metric[i] = (kl_divergence(pos_rate[i], m_pos) + kl_divergence(neg_rate[i], m_neg)) / 2;
 }
 
 return DataFrame::create(
   Named("bin") = bin_labels,
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

//' Generates a Comprehensive Gains Table from Weight of Evidence (WoE) and Target Feature Data
//'
//' This function takes a numeric vector of Weight of Evidence (WoE) values and the corresponding binary target variable
//' to generate a detailed gains table. The table includes various metrics to assess the performance and characteristics of each WoE bin.
//'
//' @param feature_woe Numeric vector representing the Weight of Evidence (WoE) values for each observation.
//' @param target Numeric vector representing the binary target variable, where 1 indicates a positive event (e.g., default) and 0 indicates a negative event (e.g., non-default).
//'
//' @return A data frame containing the following columns for each unique WoE bin:
//' \itemize{
//'   \item \code{bin}: The bin labels.
//'   \item \code{count}: Total count of observations in each bin.
//'   \item \code{pos}: Count of positive events in each bin.
//'   \item \code{neg}: Count of negative events in each bin.
//'   \item \code{woe}: Weight of Evidence (WoE) value for each bin.
//'   \item \code{iv}: Information Value (IV) contribution for each bin.
//'   \item \code{total_iv}: Total Information Value (IV) across all bins.
//'   \item \code{cum_pos}: Cumulative count of positive events up to the current bin.
//'   \item \code{cum_neg}: Cumulative count of negative events up to the current bin.
//'   \item \code{pos_rate}: Rate of positive events in each bin.
//'   \item \code{neg_rate}: Rate of negative events in each bin.
//'   \item \code{pos_perc}: Percentage of positive events relative to the total positive events.
//'   \item \code{neg_perc}: Percentage of negative events relative to the total negative events.
//'   \item \code{count_perc}: Percentage of total observations in each bin.
//'   \item \code{cum_count_perc}: Cumulative percentage of observations up to the current bin.
//'   \item \code{cum_pos_perc}: Cumulative percentage of positive events up to the current bin.
//'   \item \code{cum_neg_perc}: Cumulative percentage of negative events up to the current bin.
//'   \item \code{cum_pos_perc_total}: Cumulative percentage of positive events relative to the total observations.
//'   \item \code{cum_neg_perc_total}: Cumulative percentage of negative events relative to the total observations.
//'   \item \code{odds_pos}: Odds of positive events in each bin.
//'   \item \code{odds_ratio}: Odds ratio of positive events in the bin compared to the total population.
//'   \item \code{lift}: Lift of the bin, calculated as the ratio of the positive rate in the bin to the overall positive rate.
//'   \item \code{ks}: Kolmogorov-Smirnov statistic, measuring the difference between cumulative positive and negative percentages.
//'   \item \code{gini_contribution}: Contribution to the Gini coefficient for each bin.
//'   \item \code{precision}: Precision of the bin.
//'   \item \code{recall}: Recall up to the current bin.
//'   \item \code{f1_score}: F1 score for the bin.
//'   \item \code{log_likelihood}: Log-likelihood of the bin.
//'   \item \code{kl_divergence}: Kullback-Leibler divergence for the bin.
//'   \item \code{js_divergence}: Jensen-Shannon divergence for the bin.
//' }
//'
//' @details
//' The function performs the following steps:
//' 1. Checks if \code{feature_woe} and \code{target} have the same length.
//' 2. Verifies that \code{target} contains only binary values (0 and 1).
//' 3. Groups the target values by unique WoE values.
//' 4. Computes various metrics for each group, including counts, rates, percentages, and statistical measures.
//' 5. Handles cases where positive or negative classes have no instances by returning zero counts and appropriate NA values for derived metrics.
//'
//' The function calculates the following key metrics:
//' \itemize{
//'   \item Weight of Evidence (WoE): \deqn{WoE_i = \ln\left(\frac{P(X_i|Y=1)}{P(X_i|Y=0)}\right)}
//'   \item Information Value (IV): \deqn{IV_i = (P(X_i|Y=1) - P(X_i|Y=0)) \times WoE_i}
//'   \item Kolmogorov-Smirnov (KS) statistic: \deqn{KS_i = |F_1(i) - F_0(i)|}
//'     where \eqn{F_1(i)} and \eqn{F_0(i)} are the cumulative distribution functions for positive and negative classes.
//'   \item Odds Ratio: \deqn{OR_i = \frac{P(Y=1|X_i) / P(Y=0|X_i)}{P(Y=1) / P(Y=0)}}
//'   \item Lift: \deqn{Lift_i = \frac{P(Y=1|X_i)}{P(Y=1)}}
//'   \item Gini Contribution: \deqn{Gini_i = P(X_i|Y=1) \times F_0(i) - P(X_i|Y=0) \times F_1(i)}
//'   \item Precision: \deqn{Precision_i = \frac{TP_i}{TP_i + FP_i}}
//'   \item Recall: \deqn{Recall_i = \frac{\sum_{j=1}^i TP_j}{\sum_{j=1}^n TP_j}}
//'   \item F1 Score: \deqn{F1_i = 2 \times \frac{Precision_i \times Recall_i}{Precision_i + Recall_i}}
//'   \item Log-likelihood: \deqn{LL_i = n_{1i} \ln(p_i) + n_{0i} \ln(1-p_i)}
//'     where \eqn{n_{1i}} and \eqn{n_{0i}} are the counts of positive and negative cases in bin i,
//'     and \eqn{p_i} is the proportion of positive cases in bin i.
//'   \item Kullback-Leibler (KL) Divergence: \deqn{KL_i = p_i \ln\left(\frac{p_i}{p}\right) + (1-p_i) \ln\left(\frac{1-p_i}{1-p}\right)}
//'     where \eqn{p_i} is the proportion of positive cases in bin i and \eqn{p} is the overall proportion of positive cases.
//'   \item Jensen-Shannon (JS) Divergence: \deqn{JS_i = \frac{1}{2}KL(p_i || m) + \frac{1}{2}KL(q_i || m)}
//'     where \eqn{m = \frac{1}{2}(p_i + p)}, \eqn{p_i} is the proportion of positive cases in bin i,
//'     and \eqn{p} is the overall proportion of positive cases.
//' }
//'
//' @examples
//' \dontrun{
//' feature_woe <- c(-0.5, 0.2, 0.2, -0.5, 0.3)
//' target <- c(1, 0, 1, 0, 1)
//' gains_table <- OptimalBinningGainsTableFeature(feature_woe, target)
//' print(gains_table)
//' }
//'
//' @references
//' \itemize{
//'   \item Siddiqi, N. (2006). Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring. John Wiley & Sons.
//'   \item Hand, D. J., & Till, R. J. (2001). A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 45(2), 171-186.
//'   \item Kullback, S., & Leibler, R. A. (1951). On Information and Sufficiency. The Annals of Mathematical Statistics, 22(1), 79-86.
//'   \item Lin, J. (1991). Divergence measures based on the Shannon entropy. IEEE Transactions on Information Theory, 37(1), 145-151.
//' }
//'
//' @export
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTableFeature(NumericVector feature_woe, NumericVector target) {
 if (feature_woe.size() != target.size()) {
   stop("feature_woe and target must have the same length");
 }
 
 // Check if target is binary
 std::set<double> unique_targets(target.begin(), target.end());
 if (unique_targets.size() != 2 || !unique_targets.count(0) || !unique_targets.count(1)) {
   stop("target must contain only binary values (0 and 1)");
 }
 
 // Create a map to group data by unique feature_woe values
 std::map<double, std::vector<double>> woe_groups;
 for (int i = 0; i < feature_woe.size(); ++i) {
   woe_groups[feature_woe[i]].push_back(target[i]);
 }
 
 // Initialize vectors to store the calculated metrics
 int n = woe_groups.size();
 NumericVector woe(n), counts(n), count_pos(n), count_neg(n);
 CharacterVector bins(n);
 
 // Calculate counts and sums for each bin
 int total_count = 0, total_pos = 0, total_neg = 0;
 int i = 0;
 for (const auto& group : woe_groups) {
   woe[i] = group.first;
   counts[i] = group.second.size();
   count_pos[i] = std::count(group.second.begin(), group.second.end(), 1.0);
   count_neg[i] = counts[i] - count_pos[i];
   
   total_count += counts[i];
   total_pos += count_pos[i];
   total_neg += count_neg[i];
   
   // Create bin label
   bins[i] = "Bin_" + std::to_string(i + 1);
   
   i++;
 }
 
 // Initialize remaining vectors
 NumericVector count_perc(n), cum_count_perc(n);
 NumericVector pos_rate(n), neg_rate(n);
 NumericVector pos_perc(n), neg_perc(n);
 NumericVector cum_pos(n), cum_neg(n);
 NumericVector cum_pos_perc(n), cum_neg_perc(n);
 NumericVector cum_pos_perc_total(n), cum_neg_perc_total(n);
 NumericVector ks(n), iv(n);
 NumericVector odds_pos(n), odds_ratio(n), lift(n), gini_contribution(n);
 NumericVector precision(n), recall(n), f1_score(n);
 NumericVector log_likelihood(n);
 NumericVector kl_divergence_metric(n), js_divergence_metric(n);
 
 double total_iv = 0.0;
 double cum_pos_sum = 0.0, cum_neg_sum = 0.0;
 double overall_pos_rate = (double)total_pos / total_count;
 
 // Calculate metrics for each bin
 for (int i = 0; i < n; i++) {
   // Basic percentages
   count_perc[i] = counts[i] / total_count;
   cum_count_perc[i] = (i == 0) ? count_perc[i] : cum_count_perc[i - 1] + count_perc[i];
   
   // Rates
   pos_rate[i] = counts[i] > 0 ? count_pos[i] / counts[i] : 0;
   neg_rate[i] = counts[i] > 0 ? count_neg[i] / counts[i] : 0;
   
   // Positive and negative percentages
   pos_perc[i] = total_pos > 0 ? count_pos[i] / total_pos : 0;
   neg_perc[i] = total_neg > 0 ? count_neg[i] / total_neg : 0;
   
   // Cumulative sums
   cum_pos_sum += count_pos[i];
   cum_neg_sum += count_neg[i];
   cum_pos[i] = cum_pos_sum;
   cum_neg[i] = cum_neg_sum;
   
   // Cumulative percentages
   cum_pos_perc[i] = total_pos > 0 ? cum_pos[i] / total_pos : 0;
   cum_neg_perc[i] = total_neg > 0 ? cum_neg[i] / total_neg : 0;
   
   // Cumulative percentages relative to total observations
   cum_pos_perc_total[i] = cum_pos[i] / total_count;
   cum_neg_perc_total[i] = cum_neg[i] / total_count;
   
   // Kolmogorov-Smirnov statistic
   ks[i] = fabs(cum_pos_perc[i] - cum_neg_perc[i]);
   
   // Information Value contribution
   iv[i] = (pos_perc[i] - neg_perc[i]) * woe[i];
   total_iv += iv[i];
   
   // Odds of positive cases
   odds_pos[i] = count_neg[i] > 0 ? count_pos[i] / (double)count_neg[i] : R_PosInf;
   
   // Odds ratio
   double total_odds = total_neg > 0 ? total_pos / (double)total_neg : R_PosInf;
   odds_ratio[i] = total_odds > 0 ? odds_pos[i] / total_odds : R_NaN;
   
   // Lift
   lift[i] = overall_pos_rate > 0 ? pos_rate[i] / overall_pos_rate : R_NaN;
   
   // Gini contribution
   gini_contribution[i] = pos_perc[i] * cum_neg_perc[i] - neg_perc[i] * cum_pos_perc[i];
   
   // Precision
   precision[i] = count_pos[i] > 0 ? count_pos[i] / (double)counts[i] : 0;
   
   // Recall (cumulative)
   recall[i] = total_pos > 0 ? cum_pos[i] / (double)total_pos : 0;
   
   // F1 score
   f1_score[i] = (precision[i] + recall[i] > 0) ? 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) : 0;
   
   // Log-likelihood
   log_likelihood[i] = count_pos[i] * std::log(pos_rate[i] + 1e-10) + count_neg[i] * std::log(neg_rate[i] + 1e-10);
   
   // KL divergence
   double kl_pos = kl_divergence(pos_rate[i], overall_pos_rate);
   double kl_neg = kl_divergence(neg_rate[i], 1 - overall_pos_rate);
   kl_divergence_metric[i] = kl_pos + kl_neg;
   
   // JS divergence
   double m_pos = (pos_rate[i] + overall_pos_rate) / 2;
   double m_neg = (neg_rate[i] + (1 - overall_pos_rate)) / 2;
   js_divergence_metric[i] = (kl_divergence(pos_rate[i], m_pos) + kl_divergence(neg_rate[i], m_neg)) / 2;
 }
 
 // Create and return the gains table DataFrame
 return DataFrame::create(
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
