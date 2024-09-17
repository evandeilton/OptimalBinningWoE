
<!-- README.md is generated from README.Rmd. Please edit that file -->

# OptimalBinningWoE

<!-- badges: start -->

[![R-CMD-check](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/evandeilton/OptimalBinningWoE/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

O pacote OptimalBinningWoE oferece uma implementação robusta e flexível
de binning ótimo e cálculo de Weight of Evidence (WoE) para análise de
dados e modelagem preditiva. Este pacote é particularmente útil para
preparação de dados em modelos de credit scoring, mas pode ser aplicado
em diversos contextos de modelagem estatística.

## Instalação

Você pode instalar a versão em desenvolvimento do OptimalBinningWoE do
[GitHub](https://github.com/) com:

``` r
# install.packages("devtools")
devtools::install_github("seu_usuario/OptimalBinningWoE")
```

## Visão Geral

O OptimalBinningWoE oferece as seguintes funcionalidades principais:

1.  Binning ótimo para variáveis categóricas e numéricas
2.  Cálculo de Weight of Evidence (WoE)
3.  Seleção automática do melhor método de binning
4.  Pré-processamento de dados, incluindo tratamento de valores ausentes
    e outliers

## Algoritmos Suportados

O OptimalBinningWoE suporta os seguintes algoritmos de binning:

1.  CAIM (Class-Attribute Interdependence Maximization): Aplicável a
    variáveis categóricas e numéricas.
2.  ChiMerge: Aplicável a variáveis categóricas e numéricas.
3.  MDLP (Minimum Description Length Principle): Aplicável a variáveis
    categóricas e numéricas.
4.  MIP (Minimum Information Pure): Aplicável a variáveis categóricas e
    numéricas.
5.  MOB (Monotone Optimal Binning): Aplicável a variáveis categóricas e
    numéricas.
6.  IV (Information Value): Aplicável apenas a variáveis categóricas.
7.  PAVA (Pool Adjacent Violators Algorithm): Aplicável apenas a
    variáveis numéricas.
8.  Tree-based binning: Aplicável apenas a variáveis numéricas.

Cada algoritmo tem suas próprias forças e pode performar diferentemente
dependendo da natureza dos dados. A opção de seleção automática de
método testa os algoritmos aplicáveis e escolhe o que produz o maior
Information Value.

## Parâmetros de Controle

O pacote oferece diversos parâmetros de controle para ajustar o
comportamento do binning e pré-processamento:

- `min_bads`: Proporção mínima de casos “ruins” em cada bin (padrão:
  0.05)
- `pvalue_threshold`: Limiar de p-valor para testes estatísticos
  (padrão: 0.05)
- `max_n_prebins`: Número máximo de pré-bins antes da otimização
  (padrão: 20)
- `monotonicity_direction`: Direção da monotonia (“increase” ou
  “decrease”) (padrão: “increase”)
- `lambda`: Parâmetro de regularização para métodos baseados em árvore
  (padrão: 0.1)
- `min_bin_size`: Proporção mínima de casos em cada bin (padrão: 0.05)
- `min_iv_gain`: Ganho mínimo de IV para criar uma nova divisão (padrão:
  0.01)
- `max_depth`: Profundidade máxima para métodos baseados em árvore
  (padrão: 10)
- `num_miss_value`: Valor para representar valores numéricos ausentes
  (padrão: -999.0)
- `char_miss_value`: Valor para representar valores categóricos ausentes
  (padrão: “N/A”)
- `outlier_method`: Método para detecção de outliers (“iqr”, “zscore”,
  ou “grubbs”) (padrão: “iqr”)
- `outlier_process`: Se deve processar outliers (padrão: FALSE)
- `iqr_k`: Fator para o método IQR (padrão: 1.5)
- `zscore_threshold`: Limiar para o método Z-score (padrão: 3)
- `grubbs_alpha`: Nível de significância para o teste de Grubbs (padrão:
  0.05)

## Exemplos de uso

``` r
library(OptimalBinningWoE)
library(data.table)
library(scorecard)
library(janitor)

# Carregar os dados
data("germancredit")
dt <- as.data.table(germancredit) %>% 
  janitor::clean_names()

# Definir a variável target
dt[, default := ifelse(creditability == "bad", 1, 0)]
dt$creditability <- NULL

# Executar OptimalBinningWoE com seleção automática de método
result <- OptimalBinningWoE(dt, target = "default", method = "auto")

# Visualizar resultados
head(result$woe_feature)
head(result$woe_woebins)
head(result$prep_report)
```

## Exemplos Detalhados

### 1. Binning de variável numérica com método específico

``` r
# Usando o método MDLP para a variável 'age_in_years'
numeric_result <- OptimalBinningWoE(dt, target = "default", feature = "age_in_years", method = "mdlp")
print(numeric_result$woe_woebins)
```

### 2. Binning de variável categórica

``` r
# Usando o método ChiMerge para a variável 'purpose'
categorical_result <- OptimalBinningWoE(dt, target = "default", feature = "purpose", method = "chimerge")
print(categorical_result$woe_woebins)
```

### 3. Tratamento de valores ausentes no pré-processamento

Para este exemplo, vamos adicionar alguns valores ausentes
artificialmente:

``` r
# Adicionar valores ausentes
set.seed(123)
dt[sample(1:nrow(dt), 50), age_in_years := NA]
dt[sample(1:nrow(dt), 30), credit_amount := NA]

# Executar OptimalBinningWoE com pré-processamento
result_with_missing <- OptimalBinningWoE(dt, target = "default", feature = c("age_in_years", "credit_amount"), 
                                         preprocess = TRUE,
                                         control = list(num_miss_value = -999, char_miss_value = "MISSING"))

# Verificar o relatório de pré-processamento
print(result_with_missing$prep_report)
```

### 4. Tratamento de outliers

``` r
# Adicionar alguns outliers aos dados
dt[sample(1:nrow(dt), 10), credit_amount := rnorm(10, mean = 100000, sd = 10000)]

# Executar OptimalBinningWoE com tratamento de outliers
result_with_outliers <- OptimalBinningWoE(dt, target = "default", feature = "credit_amount", preprocess = TRUE,
                                          control = list(outlier_method = "iqr", outlier_process = TRUE))

# Verificar o relatório de pré-processamento
print(result_with_outliers$prep_report)
```

### 5. Comparação de diferentes métodos

``` r
methods <- c("caim", "chimerge", "mdlp", "mip", "mob")
results <- list()

for (method in methods) {
  results[[method]] <- OptimalBinningWoE(dt, target = "default", feature = "duration_in_month", method = method)
}

# Comparar o número de bins e IV total para cada método
comparison <- data.frame(
  Method = methods,
  Num_Bins = sapply(results, function(x) nrow(x$woe_woebins)),
  Total_IV = sapply(results, function(x) sum(x$woe_woebins$iv))
)

print(comparison)
```

### 6. Processamento de múltiplas variáveis

``` r
# Selecionar múltiplas features para processamento
selected_features <- c("age_in_years", "credit_amount", "duration_in_month", "present_residence_since", "number_of_existing_credits_at_this_bank")

# Executar OptimalBinningWoE para múltiplas features
multi_feature_result <- OptimalBinningWoE(dt, target = "default", feature = selected_features, method = "auto")

# Visualizar resultados resumidos
summary_results <- data.frame(
  Feature = selected_features,
  Num_Bins = sapply(multi_feature_result$woe_woebins, function(x) nrow(x)),
  Total_IV = sapply(multi_feature_result$woe_woebins, function(x) sum(x$iv))
)

print(summary_results)
```

### 7. Análise de uma variável ordinal

``` r
# Analisar a variável 'present_residence_since'
ordinal_result <- OptimalBinningWoE(dt, target = "default", feature = "present_residence_since", method = "mob")
print(ordinal_result$woe_woebins)
```

Estes exemplos demonstram o uso do OptimalBinningWoE com o conjunto de
dados German Credit, incluindo:

1.  Uso de variáveis reais de um conjunto de dados de credit scoring.
2.  Tratamento de variáveis numéricas, categóricas e ordinais.
3.  Lidar com valores ausentes e outliers (artificialmente introduzidos
    para demonstração).
4.  Comparação de diferentes métodos de binning.
5.  Processamento de múltiplas variáveis simultaneamente.
6.  Análise específica de uma variável ordinal.

`Esta versão dos exemplos usa o conjunto de dados German Credit, que é um conjunto de dados real e amplamente utilizado em estudos de credit scoring. Os exemplos cobrem uma variedade de cenários e tipos de variáveis presentes neste conjunto de dados, proporcionando uma demonstração mais realista e relevante do uso do pacote OptimalBinningWoE.`

## Considerações Finais

O pacote OptimalBinningWoE oferece uma solução abrangente para binning e
cálculo de WoE, com suporte para diversos algoritmos e opções de
pré-processamento. Ao usar este pacote, considere os seguintes pontos:

1.  A seleção automática de método (`method = "auto"`) pode ser útil
    quando você não tem certeza sobre qual algoritmo usar, mas pode ser
    computacionalmente intensiva para grandes conjuntos de dados.

2.  O pré-processamento de dados, incluindo o tratamento de valores
    ausentes e outliers, pode ter um impacto significativo nos
    resultados do binning. Ajuste os parâmetros de controle conforme
    necessário para seu conjunto de dados específico.

3.  Diferentes métodos de binning podem produzir resultados
    significativamente diferentes. É uma boa prática comparar os
    resultados de vários métodos antes de fazer uma escolha final.

4.  Para conjuntos de dados muito grandes, considere usar uma amostra
    representativa para determinar os bins ótimos e depois aplicar esses
    bins ao conjunto de dados completo.

5.  O pacote fornece flexibilidade para lidar com diferentes tipos de
    dados e cenários de modelagem. Experimente diferentes configurações
    para encontrar a melhor abordagem para seus dados específicos.

Para mais detalhes sobre as opções disponíveis e interpretação dos
resultados, consulte a documentação completa do pacote.

## Contribuindo

Contribuições para o OptimalBinningWoE são bem-vindas! Por favor,
consulte o arquivo CONTRIBUTING.md para diretrizes sobre como contribuir
para este projeto.

## Licença

Este projeto está licenciado sob a Licença MIT - Veja o arquivo
LICENSE.md para detalhes.
