# require(Rcpp);library(testthat);require(ggplot2);rm(list = ls(), inherits = TRUE)
# sourceCpp("src/algo.cpp")
# sourceCpp("src/dataprep.cpp")
# sourceCpp("src/utils.cpp")

## --------------------------------------------------------------------------------------------- ##
## OptimalBinningCategoricalMIP
## --------------------------------------------------------------------------------------------- ##
# library(woecpp)


# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalMIP(target, feature)
# 
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalMOB
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalMOB(target, feature)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalChiMerge
# ## --------------------------------------------------------------------------------------------- ##
# library(testthat)
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalChiMerge(target, feature, pvalue_threshold = 0.05)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalMDLP
# ## --------------------------------------------------------------------------------------------- ##
# library(testthat)
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalMDLP(target, feature)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalCAIM
# ## --------------------------------------------------------------------------------------------- ##
# library(testthat)
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalCAIM(target, feature)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalIV
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- sample(LETTERS[1:10], N, replace = TRUE)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningCategoricalIV(target, feature)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericMIP
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(123)
# N <- 1000
# feature <- rnorm(N)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningNumericMIP(target, feature)
# 
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericMOB
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(1234)
# N <- 10000
# feature <- rnorm(N)
# target <- rbinom(N, 1, prob = 0.3)
# 
# # Call the function
# result <- OptimalBinningNumericMOB(target, feature)
# 
# test_that("Output structure is correct", {
#   expect_true("feature_woe" %in% names(result))
#   expect_true("bin" %in% names(result))
#   expect_true("woe" %in% names(result))
#   expect_true("iv" %in% names(result))
#   expect_true("pos" %in% names(result))
#   expect_true("neg" %in% names(result))
# })
# 
# test_that("Feature WoE length matches input length", {
#   expect_equal(length(result$feature_woe), length(target))
# })
# 
# test_that("Bins are monotonic", {
#   bin_df <- result$bin
#   event_rates <- bin_df$count_pos / (bin_df$count_pos + bin_df$count_neg)
#   is_increasing <- all(diff(event_rates) >= 0)
#   is_decreasing <- all(diff(event_rates) <= 0)
#   expect_true(is_increasing || is_decreasing)
# })
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericChiMerge
# ## --------------------------------------------------------------------------------------------- ##
# 
# set.seed(123)
# n <- 1000
# feature <- rnorm(n)
# target <- rbinom(n, 1, prob = 1 / (1 + exp(-feature)))
# 
# result <- OptimalBinningNumericChiMerge(feature, target)
# 
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericMDLP
# ## --------------------------------------------------------------------------------------------- ##
# 
# set.seed(123)
# n <- 1000
# feature <- rnorm(n)
# prob <- 1 / (1 + exp(-feature))
# target <- rbinom(n, 1, prob)
# 
# result <- OptimalBinningNumericMDLP(feature, target)
# 
# 
# test_that("IV is non-negative", {
#   expect_true(result$iv >= 0)
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericCAIM
# ## --------------------------------------------------------------------------------------------- ##
# 
# set.seed(123)
# n <- 1000
# feature <- rnorm(n)
# prob <- 1 / (1 + exp(-feature))
# target <- rbinom(n, 1, prob)
# 
# result <- OptimalBinningNumericCAIM(feature, target)
# 
# da <- data.frame(x = feature, xw = result$feature_woe)
# 
# 
# test_that("IV is non-negative", {
#   expect_true(result$iv >= 0)
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalMIP
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(123)
# N <- 10000
# feature <- rnorm(N)
# target <- rbinom(N, 1, prob = 1 / (1 + exp(-feature)))
# 
# # Apply the binning function
# result <- OptimalBinningNumericPAVA(feature, target)
# 
# # Tests
# test_that("Output structure is correct", {
#   expect_true(is.list(result))
#   expect_true(all(c("feature_woe", "bin", "woe", "iv", "pos", "neg") %in% names(result)))
# })
# 
# test_that("WoE values are numeric", {
#   expect_true(is.numeric(result$woe))
# })
# 
# test_that("Feature WoE has correct length", {
#   expect_equal(length(result$feature_woe), length(feature))
# })
# 
# test_that("Bins have correct counts", {
#   total_count <- sum(result$bin$count)
#   expect_equal(total_count, length(feature))
# })
# 
# # Simulate data
# set.seed(123)
# N <- 10000
# feature <- rnorm(N)
# target <- ifelse(feature + rnorm(N) > 0, 1, 0)
# 
# # Apply the binning function
# result <- OptimalBinningNumericPAVA(feature, target, monotonicity_direction = "increasing")
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericPAVA
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Simulate data
# set.seed(123)
# N <- 10000
# feature <- rnorm(N)
# target <- rbinom(N, 1, prob = 1 / (1 + exp(-feature)))
# 
# # Apply the binning function
# result <- OptimalBinningNumericPAVA(feature, target)
# 
# # Tests
# test_that("Bin intervals are correctly formatted", {
#   expected_format <- "^\\[.*;.*\\)|^\\(.*;.*\\)$|^\\(.*;.*\\]$"
#   expect_true(all(grepl(expected_format, result$bin$bin)))
# })
# 
# test_that("Numbers are formatted to six decimal places", {
#   number_pattern <- "-?\\d+\\.\\d{6}"
#   expect_true(all(grepl(number_pattern, gsub("[^0-9.-]", "", result$bin$bin))))
# })
# 
# # Additional tests
# test_that("Output structure is correct", {
#   expect_true(is.list(result))
#   expect_true(all(c("feature_woe", "bin", "woe", "iv", "pos", "neg") %in% names(result)))
# })
# 
# test_that("WoE values are numeric", {
#   expect_true(is.numeric(result$woe))
# })
# 
# test_that("Feature WoE has correct length", {
#   expect_equal(length(result$feature_woe), length(feature))
# })
# 
# test_that("Bins have correct counts", {
#   total_count <- sum(result$bin$count)
#   expect_equal(total_count, length(feature))
# })
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningNumericTree
# ## --------------------------------------------------------------------------------------------- ##
# # Generate sample data with a clear relationship between feature and target
# set.seed(123)
# n_samples <- 10000
# feature <- rnorm(n_samples, mean = 50, sd = 10)
# # Introduce some relationship between feature and target
# target <- rbinom(n_samples, 1, plogis((feature - 50)/10))
# 
# # Perform optimal binning
# result <- OptimalBinningNumericTree(
#   feature = feature,
#   target = target,
#   lambda = 0.1,
#   min_bin_size = 0.05,
#   max_bins = 10,
#   min_iv_gain = 0.01,
#   monotonicity_direction = "increase",
#   max_depth = 10
# )
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningCategoricalMIP
# ## --------------------------------------------------------------------------------------------- ##
# test_that("OptimalBinningDataPreprocessor works correctly with numeric data", {
#   set.seed(123)
#   target <- sample(c(0,1), 1000, replace = TRUE)
#   feature <- rnorm(1000, mean = 50, sd = 10)
# 
#   # Introduce missing values
#   feature[sample(1:1000, 50)] <- NA
# 
#   # Introduce outliers
#   outlier_indices <- sample(1:1000, 10)
#   feature[outlier_indices] <- feature[outlier_indices] + 100
# 
#   result <- OptimalBinningDataPreprocessor(
#     target = target,
#     feature = feature,
#     num_miss_value = -999,
#     outlier_method = "iqr",
#     outlier_process = TRUE,
#     preprocess = "both",
#     iqr_k = 1.5
#   )
# 
#   expect_true("preprocess" %in% names(result))
#   expect_true("report" %in% names(result))
# 
#   preprocess_df <- result$preprocess
#   report_df <- result$report
# 
#   expect_equal(nrow(preprocess_df), 1000)
#   expect_equal(ncol(preprocess_df), 2)
# 
#   # Check that missing values are replaced
#   expect_false(any(is.na(preprocess_df$feature_preprocessed)))
# 
#   # Check report
#   expect_equal(report_df$variable_type, "numeric")
#   expect_true(report_df$missing_count >= 50)
#   expect_true(report_df$outlier_count >= 10)
#   expect_true(grepl("min:", report_df$original_stats))
#   expect_true(grepl("mean:", report_df$preprocessed_stats))
# })
# 
# test_that("OptimalBinningDataPreprocessor works correctly with categorical data", {
#   set.seed(456)
#   target <- sample(c(0,1), 500, replace = TRUE)
#   categories <- c("A", "B", "C", "D")
#   feature <- sample(categories, 500, replace = TRUE)
# 
#   # Introduce missing values
#   feature[sample(1:500, 20)] <- NA
# 
#   result <- OptimalBinningDataPreprocessor(
#     target = target,
#     feature = feature,
#     char_miss_value = "N/A",
#     preprocess = "both"
#   )
# 
#   expect_true("preprocess" %in% names(result))
#   expect_true("report" %in% names(result))
# 
#   preprocess_df <- result$preprocess
#   report_df <- result$report
# 
#   expect_equal(nrow(preprocess_df), 500)
#   expect_equal(ncol(preprocess_df), 2)
# 
#   # Check that missing values are replaced
#   expect_false(any(is.na(preprocess_df$feature_preprocessed)))
#   expect_true(all(preprocess_df$feature_preprocessed %in% c(categories, "N/A")))
# 
#   # Check report
#   expect_equal(report_df$variable_type, "categorical")
#   expect_equal(report_df$missing_count, 20)
#   expect_true(is.na(report_df$outlier_count))
#   expect_true(grepl("unique_count:", report_df$preprocessed_stats))
# })
# 
# test_that("OptimalBinningDataPreprocessor handles non-binary target correctly", {
#   target <- c(0,1,2,1,0)
#   feature <- c(10, 20, 30, 40, 50)
# 
#   expect_error(
#     OptimalBinningDataPreprocessor(
#       target = target,
#       feature = feature
#     ),
#     "Target variable must be binary."
#   )
# })
# 
# # test_that("OptimalBinningDataPreprocessor handles invalid outlier method", {
# #   set.seed(789)
# #   target <- sample(c(0,1), 100, replace = TRUE)
# #   feature <- rnorm(100)
# #
# #   expect_error(
# #     OptimalBinningDataPreprocessor(
# #       target = target,
# #       feature = feature,
# #       outlier_method = "invalid_method"
# #     ),
# #     "Invalid outlier_method. Choose from 'iqr', 'zscore', or 'grubbs'."
# #   )
# # })
# 
# test_that("OptimalBinningDataPreprocessor handles Grubbs' method correctly", {
#   set.seed(101)
#   target <- sample(c(0,1), 200, replace = TRUE)
#   feature <- rnorm(200, mean = 0, sd = 1)
# 
#   # Introduce a single outlier
#   feature[1] <- 10
# 
#   result <- OptimalBinningDataPreprocessor(
#     target = target,
#     feature = feature,
#     outlier_method = "grubbs",
#     outlier_process = TRUE,
#     preprocess = "both",
#     grubbs_alpha = 0.05
#   )
# 
#   report_df <- result$report
#   expect_true(report_df$outlier_count >=1)
# 
#   # Check that the outlier has been replaced
#   expect_true(result$preprocess$feature_preprocessed[1] != 10)
# })
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningGainsTable
# ## --------------------------------------------------------------------------------------------- ##
# 
# # Define test case for OptimalBinningGainsTable
# test_that("OptimalBinningGainsTable calculates correct metrics for OptimalBinningCategoricalMIP output", {
#   # Simulate data
#   set.seed(123)
#   N <- 1000
#   feature <- sample(LETTERS[1:10], N, replace = TRUE)
#   target <- rbinom(N, 1, prob = 0.3)
# 
#   # Call the OptimalBinningCategoricalMIP function to get binning results
#   binning_result <- OptimalBinningCategoricalMIP(target, feature, cat_cutoff = 0.05, min_bins = 2, max_bins = 5)
# 
#   # Call OptimalBinningGainsTable with the binning result
#   gains_table <- OptimalBinningGainsTable(binning_result)
# 
#   # Check that the gains table has the correct dimensions and structure
#   expect_equal(nrow(gains_table), nrow(binning_result$bin))
#   expect_equal(ncol(gains_table), 24)  # Ensure all required columns are present
# 
#   # Test some key metrics for consistency with known values
#   expect_equal(gains_table$count, binning_result$bin$count)
#   expect_equal(gains_table$pos, binning_result$bin$count_pos)
#   expect_equal(gains_table$neg, binning_result$bin$count_neg)
# 
#   # Check Information Value consistency
#   expect_equal(gains_table$total_iv[1], binning_result$iv)
# 
#   # Check that the cumulative metrics are correctly computed
#   expect_true(all(diff(gains_table$cum_count_perc) >= 0))
#   expect_true(all(diff(gains_table$cum_pos_perc_total) >= 0))
#   expect_true(all(diff(gains_table$cum_neg_perc_total) >= 0))
# 
#   # Check that ks (Kolmogorov-Smirnov) is computed correctly
#   expect_true(all(gains_table$ks >= 0))
# })
# 
# test_that("OptimalBinningGainsTable handles edge cases with zero counts", {
#   # Simulate binning result with zero counts for some bins
#   binning_result <- list(
#     bin = data.frame(
#       bin = c("A", "B", "C"),
#       woe = c(0.2, -0.1, 0.3),
#       iv_bin = c(0.05, 0.02, 0.08),
#       count = c(100, 0, 250),
#       count_pos = c(40, 0, 80),
#       count_neg = c(60, 0, 170)
#     ),
#     feature_woe = rnorm(500),
#     woe = c(0.2, -0.1, 0.3),
#     iv = 0.15,
#     pos = 120,
#     neg = 230
#   )
# 
#   # Call OptimalBinningGainsTable with the binning result
#   gains_table <- OptimalBinningGainsTable(binning_result)
# 
#   # Ensure that count percentages and cumulative percentages handle zero counts correctly
#   expect_equal(gains_table$count[2], 0)
#   expect_equal(gains_table$count_perc[2], 0)
#   expect_equal(gains_table$cum_count_perc[2], gains_table$cum_count_perc[1])
# 
#   # Ensure that odds_pos and other metrics handle zero counts properly (e.g., avoid division by zero)
#   expect_true(is.na(gains_table$odds_pos[2]))
#   expect_true(is.na(gains_table$odds_ratio[2]))
# })
# 
# test_that("OptimalBinningGainsTable returns correct lift values", {
#   # Simulate binning result
#   binning_result <- list(
#     bin = data.frame(
#       bin = c("A", "B", "C"),
#       woe = c(0.2, -0.1, 0.3),
#       iv_bin = c(0.05, 0.02, 0.08),
#       count = c(100, 150, 250),
#       count_pos = c(40, 60, 80),
#       count_neg = c(60, 90, 170)
#     ),
#     feature_woe = rnorm(500),
#     woe = c(0.2, -0.1, 0.3),
#     iv = 0.15,
#     pos = 180,
#     neg = 320
#   )
# 
#   # Call OptimalBinningGainsTable with the binning result
#   gains_table <- OptimalBinningGainsTable(binning_result)
# 
#   # Check lift values are calculated correctly
#   total_pos_rate <- 180 / 500
#   expected_lift_A <- (40 / 100) / total_pos_rate
#   expected_lift_B <- (60 / 150) / total_pos_rate
#   expected_lift_C <- (80 / 250) / total_pos_rate
# 
#   expect_equal(gains_table$lift[1], expected_lift_A)
#   expect_equal(gains_table$lift[2], expected_lift_B)
#   expect_equal(gains_table$lift[3], expected_lift_C)
# })
# 
# 
# 
# ## --------------------------------------------------------------------------------------------- ##
# ## OptimalBinningGainsTableFeature
# ## --------------------------------------------------------------------------------------------- ##
# 
# # test_that("OptimalBinningGainsTableFeature calculates correct metrics", {
# #   # Simulate data
# #   set.seed(123)
# #   N <- 1000
# #   feature <- sample(LETTERS[1:10], N, replace = TRUE)
# #   target <- rbinom(N, 1, prob = 0.3)
# #
# #   # Call the OptimalBinningCategoricalMIP function to get binning results
# #   binning_result <- OptimalBinningCategoricalMIP(target, feature, cat_cutoff = 0.05, min_bins = 2, max_bins = 5)
# #
# #   result <- OptimalBinningGainsTableFeature(binning_result$feature_woe, target)
# #
# #   expect_equal(nrow(result), 3)  # 3 unique WoE values
# #   expect_equal(result$woe, c(-0.5, 0.2, 0.4))
# #   expect_equal(result$count, c(2, 3, 2))
# #   expect_equal(result$pos, c(1, 2, 1))
# #   expect_equal(result$neg, c(1, 1, 1))
# #
# #   # Test case 2: Edge case - all positive targets
# #   feature_woe <- c(-1, 0, 1)
# #   target <- c(1, 1, 1)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_equal(nrow(result), 3)
# #   expect_true(all(is.na(result$neg_rate)))
# #   expect_true(all(result$pos_rate == 1))
# #
# #   # Test case 3: Edge case - all negative targets
# #   feature_woe <- c(-1, 0, 1)
# #   target <- c(0, 0, 0)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_equal(nrow(result), 3)
# #   expect_true(all(is.na(result$pos_rate)))
# #   expect_true(all(result$neg_rate == 1))
# #
# #   # Test case 4: Large dataset
# #   set.seed(123)
# #   feature_woe <- rnorm(1000)
# #   target <- sample(0:1, 1000, replace = TRUE)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_true(nrow(result) <= 1000)  # Should be less than or equal to 1000 unique WoE values
# #   expect_equal(sum(result$count), 1000)
# #   expect_equal(sum(result$pos), sum(target))
# #   expect_equal(sum(result$neg), sum(1 - target))
# #
# #   # Test case 5: Cumulative metrics
# #   feature_woe <- c(-1, -0.5, 0, 0.5, 1)
# #   target <- c(0, 1, 1, 0, 1)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_equal(result$cum_pos, c(0, 1, 2, 2, 3))
# #   expect_equal(result$cum_neg, c(1, 1, 1, 2, 2))
# #   expect_equal(result$cum_count_perc, c(0.2, 0.4, 0.6, 0.8, 1.0))
# #
# #   # Test case 6: Information Value
# #   feature_woe <- c(-1, -1, 0, 0, 1, 1)
# #   target <- c(0, 1, 0, 1, 0, 1)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_length(unique(result$total_iv), 1)  # Total IV should be the same for all rows
# #   expect_equal(sum(result$iv), unique(result$total_iv))
# #
# #   # Test case 7: KS statistic
# #   feature_woe <- c(-2, -1, 0, 1, 2)
# #   target <- c(0, 0, 1, 1, 1)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_true(all(result$ks >= 0 & result$ks <= 1))
# #   expect_equal(max(result$ks), max(abs(result$cum_pos_perc - result$cum_neg_perc)))
# #
# #   # Test case 8: Error handling - mismatched lengths
# #   feature_woe <- c(-1, 0, 1)
# #   target <- c(0, 1)
# #   expect_error(OptimalBinningGainsTableFeature(feature_woe, target), "feature_woe and target must have the same length")
# #
# #   # Test case 9: Monotonicity of cumulative metrics
# #   feature_woe <- runif(100, -1, 1)
# #   target <- sample(0:1, 100, replace = TRUE)
# #   result <- OptimalBinningGainsTableFeature(feature_woe, target)
# #
# #   expect_true(all(diff(result$cum_pos) >= 0))
# #   expect_true(all(diff(result$cum_neg) >= 0))
# #   expect_true(all(diff(result$cum_count_perc) >= 0))
# # })
