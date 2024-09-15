#include <Rcpp.h>
#include <algorithm>
#include <vector>
#include <cmath>
#include <map>

using namespace Rcpp;

//' Generates a Gains Table from the results of optimal binning
//'
//' This function takes the result of the optimal binning process and generates a detailed gains table. The gains table includes metrics such as the Weight of Evidence (WoE), Information Value (IV), cumulative positive and negative percentages, Kolmogorov-Smirnov (KS) statistic, odds ratio, lift, and Gini contribution for each bin.
//'
//' @param binning_result A list containing the binning results, which must include a data frame with the following columns: "bin", "count", "count_pos", "count_neg", and "woe".
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
//' }
//'
//' @examples
//' \dontrun{
//' binning_result <- OptimalBinning(target, feature)
//' gains_table <- OptimalBinningGainsTable(binning_result)
//' }
//'
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTable(List binning_result) {
  // Extract bin DataFrame from the binning_result list
  DataFrame bin = as<DataFrame>(binning_result["bin"]);

  // Extract columns from the bin DataFrame
  CharacterVector bin_labels = bin["bin"];
  NumericVector counts = bin["count"];
  NumericVector count_pos = bin["count_pos"];
  NumericVector count_neg = bin["count_neg"];
  NumericVector woe = bin["woe"];

  // Total counts
  int total_count = sum(counts);
  int total_pos = sum(count_pos);
  int total_neg = sum(count_neg);

  // Initialize vectors to store the calculated metrics
  int n = bin_labels.size();
  NumericVector count_perc(n), cum_count_perc(n);
  NumericVector pos_rate(n), neg_rate(n);
  NumericVector pos_perc(n), neg_perc(n);
  NumericVector cum_pos(n), cum_neg(n);
  NumericVector cum_pos_perc(n), cum_neg_perc(n);
  NumericVector cum_pos_perc_total(n), cum_neg_perc_total(n);
  NumericVector ks(n), iv(n);
  NumericVector odds_pos(n), odds_ratio(n), lift(n), gini_contribution(n);

  double total_iv = 0.0;

  // Loop through each bin and calculate the metrics
  double cum_pos_sum = 0.0, cum_neg_sum = 0.0;
  for (int i = 0; i < n; i++) {
    // Basic percentages
    count_perc[i] = counts[i] / total_count;
    cum_count_perc[i] = (i == 0) ? count_perc[i] : cum_count_perc[i - 1] + count_perc[i];

    // Rates
    pos_rate[i] = count_pos[i] / counts[i];
    neg_rate[i] = count_neg[i] / counts[i];

    // Positive and negative percentages
    pos_perc[i] = count_pos[i] / total_pos;
    neg_perc[i] = count_neg[i] / total_neg;

    // Cumulative sums
    cum_pos_sum += count_pos[i];
    cum_neg_sum += count_neg[i];
    cum_pos[i] = cum_pos_sum;
    cum_neg[i] = cum_neg_sum;

    // Cumulative percentages
    cum_pos_perc[i] = cum_pos[i] / total_pos;
    cum_neg_perc[i] = cum_neg[i] / total_neg;

    // Cumulative percentages relative to total observations
    cum_pos_perc_total[i] = cum_pos[i] / total_count;
    cum_neg_perc_total[i] = cum_neg[i] / total_count;

    // Kolmogorov-Smirnov statistic
    ks[i] = fabs(cum_pos_perc[i] - cum_neg_perc[i]);

    // Information Value contribution
    iv[i] = (pos_perc[i] - neg_perc[i]) * woe[i];
    total_iv += iv[i];

    // Odds of positive cases
    odds_pos[i] = (count_neg[i] == 0) ? NA_REAL : count_pos[i] / count_neg[i];

    // Odds ratio
    double total_odds = (total_neg == 0) ? NA_REAL : total_pos / (double)total_neg;
    odds_ratio[i] = (total_odds == 0 || NumericVector::is_na(odds_pos[i])) ? NA_REAL : odds_pos[i] / total_odds;

    // Lift
    lift[i] = (total_pos == 0 || total_count == 0) ? NA_REAL : pos_rate[i] / (total_pos / (double)total_count);

    // Gini contribution
    gini_contribution[i] = pos_perc[i] * cum_neg_perc[i] - neg_perc[i] * cum_pos_perc[i];
  }

  // Create and return the gains table DataFrame
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
    Named("gini_contribution") = gini_contribution
  );
}


//' Generates a Gains Table from the Weight of Evidence (WoE) and target feature data
//'
//' This function takes a numeric vector of Weight of Evidence (WoE) values and the corresponding binary target variable to generate a gains table. The table includes key metrics such as counts, event rates, cumulative sums, Kolmogorov-Smirnov (KS) statistic, Information Value (IV), odds ratio, lift, and Gini contribution for each unique WoE bin.
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
//' }
//'
//' @details
//' The function assumes that both \code{feature_woe} and \code{target} have the same length. It groups the target values by the unique WoE values, computes various metrics for each group, and returns a comprehensive gains table.
//'
//' @examples
//' \dontrun{
//' feature_woe <- c(-0.5, 0.2, 0.2, -0.5, 0.3)
//' target <- c(1, 0, 1, 0, 1)
//' gains_table <- OptimalBinningGainsTableFeature(feature_woe, target)
//' }
// [[Rcpp::export]]
DataFrame OptimalBinningGainsTableFeature(NumericVector feature_woe, NumericVector target) {
  if (feature_woe.size() != target.size()) {
    stop("feature_woe and target must have the same length");
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

  double total_iv = 0.0;
  double cum_pos_sum = 0.0, cum_neg_sum = 0.0;

  // Calculate metrics for each bin
  for (int i = 0; i < n; i++) {
    // Basic percentages
    count_perc[i] = counts[i] / total_count;
    cum_count_perc[i] = (i == 0) ? count_perc[i] : cum_count_perc[i - 1] + count_perc[i];

    // Rates
    pos_rate[i] = count_pos[i] / counts[i];
    neg_rate[i] = count_neg[i] / counts[i];

    // Positive and negative percentages
    pos_perc[i] = count_pos[i] / total_pos;
    neg_perc[i] = count_neg[i] / total_neg;

    // Cumulative sums
    cum_pos_sum += count_pos[i];
    cum_neg_sum += count_neg[i];
    cum_pos[i] = cum_pos_sum;
    cum_neg[i] = cum_neg_sum;

    // Cumulative percentages
    cum_pos_perc[i] = cum_pos[i] / total_pos;
    cum_neg_perc[i] = cum_neg[i] / total_neg;

    // Cumulative percentages relative to total observations
    cum_pos_perc_total[i] = cum_pos[i] / total_count;
    cum_neg_perc_total[i] = cum_neg[i] / total_count;

    // Kolmogorov-Smirnov statistic
    ks[i] = fabs(cum_pos_perc[i] - cum_neg_perc[i]);

    // Information Value contribution
    iv[i] = (pos_perc[i] - neg_perc[i]) * woe[i];
    total_iv += iv[i];

    // Odds of positive cases
    odds_pos[i] = (count_neg[i] == 0) ? NA_REAL : count_pos[i] / count_neg[i];

    // Odds ratio
    double total_odds = (total_neg == 0) ? NA_REAL : total_pos / (double)total_neg;
    odds_ratio[i] = (total_odds == 0 || NumericVector::is_na(odds_pos[i])) ? NA_REAL : odds_pos[i] / total_odds;

    // Lift
    lift[i] = (total_pos == 0 || total_count == 0) ? NA_REAL : pos_rate[i] / (total_pos / (double)total_count);

    // Gini contribution
    gini_contribution[i] = pos_perc[i] * cum_neg_perc[i] - neg_perc[i] * cum_pos_perc[i];
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
    Named("gini_contribution") = gini_contribution
  );
}
