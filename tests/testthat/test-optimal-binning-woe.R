library(OptimalBinningWoE)
library(data.table)
library(testthat)

# Função auxiliar para criar dados de teste
create_test_data <- function(n = 1000) {
  set.seed(123)  # Para reprodutibilidade
  data.table(
    target = sample(0:1, n, replace = TRUE),
    num_feat1 = rnorm(n),
    num_feat2 = runif(n, 0, 100),
    cat_feat1 = sample(letters[1:5], n, replace = TRUE),
    cat_feat2 = sample(c("A", "B", "C", "D"), n, replace = TRUE, prob = c(0.4, 0.3, 0.2, 0.1)),
    date_feat = as.Date("2023-01-01") + sample(0:365, n, replace = TRUE),
    missing_feat = sample(c(1:10, NA), n, replace = TRUE)
  )
}

# Testes para OptimalBinningWoE
test_that("OptimalBinningWoE funciona corretamente", {
  
  # Teste 1: Funcionamento básico
  dt <- create_test_data()
  result <- OptimalBinningWoE(dt, target = "target")
  expect_type(result, "list")
  expect_named(result, c("woe_feature", "woe_woebins", "prep_report"))
  expect_true(all(paste0(c("num_feat1", "num_feat2", "cat_feat1", "cat_feat2", "missing_feat"), "_woe") %in% names(result$woe_feature)))
  
  # Teste 2: Método específico
  result_caim <- OptimalBinningWoE(dt, target = "target", method = "caim")
  expect_true("woe_feature" %in% names(result_caim))
  
  # Teste 3: Pré-processamento desativado
  result_no_preprocess <- OptimalBinningWoE(dt, target = "target", preprocess = FALSE)
  expect_true(nrow(result_no_preprocess$prep_report) == 0)
  
  # Teste 4: Feature específica
  result_single_feature <- OptimalBinningWoE(dt, target = "target", feature = "num_feat1")
  expect_true("num_feat1_woe" %in% names(result_single_feature$woe_feature))
  expect_false("num_feat2_woe" %in% names(result_single_feature$woe_feature))
  
  # Teste 5: Parâmetros de controle personalizados
  custom_control <- list(min_bads = 0.1, max_n_prebins = 15)
  result_custom_control <- OptimalBinningWoE(dt, target = "target", control = custom_control)
  expect_type(result_custom_control, "list")
  
  # Teste 6: Número mínimo e máximo de bins
  result_bins <- OptimalBinningWoE(dt, target = "target", min_bins = 3, max_bins = 5)
  expect_true(all(result_bins$woe_woebins[, .N, by = feature]$N >= 3 & result_bins$woe_woebins[, .N, by = feature]$N <= 5))
  
  # Teste 7: Tratamento de valores ausentes
  dt_with_na <- copy(dt)
  dt_with_na[sample(1:nrow(dt_with_na), 100), num_feat1 := NA]
  result_with_na <- OptimalBinningWoE(dt_with_na, target = "target")
  expect_true(any(grepl("Special", result_with_na$woe_woebins$bin)))
  
  # Teste 8: Variável alvo não binária
  dt_multi_target <- copy(dt)
  dt_multi_target[, target := sample(1:3, .N, replace = TRUE)]
  expect_error(OptimalBinningWoE(dt_multi_target, target = "target"))
  
  # Teste 9: Dados com poucas observações
  dt_small <- create_test_data(50)
  result_small <- OptimalBinningWoE(dt_small, target = "target")
  expect_type(result_small, "list")
  
  # Teste 10: Todos os valores iguais em uma feature
  dt_constant <- copy(dt)
  dt_constant[, constant_feat := 1]
  result_constant <- OptimalBinningWoE(dt_constant, target = "target")
  expect_true("constant_feat_woe" %in% names(result_constant$woe_feature))
  
  # Teste 11: Feature com muitas categorias
  dt_many_cat <- copy(dt)
  dt_many_cat[, many_cat := sample(letters, .N, replace = TRUE)]
  result_many_cat <- OptimalBinningWoE(dt_many_cat, target = "target")
  expect_true("many_cat_woe" %in% names(result_many_cat$woe_feature))
  
  # Teste 12: Teste de robustez com dados extremos
  dt_extreme <- copy(dt)
  dt_extreme[, extreme_num := c(rep(0, .N-1), 1e9)]
  result_extreme <- OptimalBinningWoE(dt_extreme, target = "target")
  expect_true("extreme_num_woe" %in% names(result_extreme$woe_feature))
  
  # Teste 13: Verificação de monotonia para método MOB
  result_mob <- OptimalBinningWoE(dt, target = "target", method = "mob")
  mob_woe_values <- result_mob$woe_woebins[feature == "num_feat1", woe]
  expect_true(all(diff(mob_woe_values) >= 0) | all(diff(mob_woe_values) <= 0))
  
  # Teste 14: Teste com diferentes tipos de dados
  dt_types <- data.table(
    target = sample(0:1, 1000, replace = TRUE),
    integer_feat = sample(1:100, 1000, replace = TRUE),
    factor_feat = as.factor(sample(letters[1:5], 1000, replace = TRUE)),
    logical_feat = sample(c(TRUE, FALSE), 1000, replace = TRUE),
    date_feat = as.Date("2023-01-01") + sample(0:365, 1000, replace = TRUE)
  )
  result_types <- OptimalBinningWoE(dt_types, target = "target")
  expect_true(all(paste0(c("integer_feat", "factor_feat", "logical_feat"), "_woe") %in% names(result_types$woe_feature)))
  
  # Teste 15: Teste de performance com conjunto de dados maior
  dt_large <- create_test_data(1e5)
  start_time <- Sys.time()
  result_large <- OptimalBinningWoE(dt_large, target = "target")
  end_time <- Sys.time()
  expect_true(as.numeric(end_time - start_time, units = "secs") < 60)  # Espera-se que execute em menos de 1 minuto
})
