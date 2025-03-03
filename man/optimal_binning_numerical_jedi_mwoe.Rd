% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_numerical_jedi_mwoe}
\alias{optimal_binning_numerical_jedi_mwoe}
\title{Optimal Numerical Binning JEDI M-WOE (Multinomial Weight of Evidence)}
\usage{
optimal_binning_numerical_jedi_mwoe(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L,
  convergence_threshold = 1e-06,
  max_iterations = 1000L
)
}
\arguments{
\item{target}{IntegerVector de tamanho n, com valores de 0..(K-1) indicando a classe.}

\item{feature}{NumericVector de tamanho n, com os valores contínuos da feature.}

\item{min_bins}{Número mínimo de bins no resultado (>=2).}

\item{max_bins}{Número máximo de bins no resultado (>= min_bins).}

\item{bin_cutoff}{Frequência mínima relativa de um bin para não ser mesclado (0<bin_cutoff<1).}

\item{max_n_prebins}{Número máximo de pré-bins (fase inicial via quantis).}

\item{convergence_threshold}{Tolerância para parar iterações com base na variação do IV total.}

\item{max_iterations}{Número máximo de iterações permitidas.}
}
\value{
Uma lista com:
\itemize{
\item \code{bin}: vetor de rótulos dos bins (intervalos).
\item \code{woe}: matriz (n_bins x n_classes) de M-WOE para cada bin e classe.
\item \code{iv}: matriz (n_bins x n_classes) de IV por bin e classe.
\item \code{count}: vetor com contagem total por bin.
\item \code{class_counts}: matriz (n_bins x n_classes) com contagem por classe em cada bin.
\item \code{cutpoints}: pontos de corte (excluindo ±Inf).
\item \code{converged}: indica se houve convergência via \code{convergence_threshold}.
\item \code{iterations}: número de iterações realizadas.
\item \code{n_classes}: número de classes detectadas.
}
}
\description{
Versão multinomial do binning numérico JEDI, que estende o WOE/IV tradicional (binário)
para uma abordagem M-WOE, considerando várias classes simultaneamente.
}
\details{
Implementa a discretização de variáveis numéricas em múltiplas classes (K>2), calculando
o M-WOE e o M-IV (Information Value Multinomial), e forçando monotonicidade para cada classe
por mesclagem iterativa de bins adjacentes que violem a ordem (crescente ou decrescente) de WOE.

Fórmulas de M-WOE e M-IV para cada classe k em um bin i:
\deqn{M-WOE_{i,k} = \ln\left(\frac{ \frac{\text{count}_{i,k}}{ \text{Total}_k }}{ \frac{\sum_{j \neq k} \text{count}_{i,j}}{\sum_{j \neq k} \text{Total}_j}} \right)}

\deqn{IV_{i,k} = \Bigl(\frac{\text{count}_{i,k}}{\text{Total}_k} - \frac{\sum_{j \neq k}\text{count}_{i,j}}{\sum_{j \neq k}\text{Total}_j}\Bigr) \times M-WOE_{i,k}}

O IV total do bin i é \eqn{\sum_k IV_{i,k}} e o IV global é \eqn{\sum_i \sum_k IV_{i,k}}.
}
\examples{
\dontrun{
# Exemplo com 3 classes: 0, 1 e 2
target <- c(0,1,2,1,0,2,2,1,0,0,2)
feature <- c(1.1,2.2,3.5,2.7,1.0,4.2,3.9,2.8,1.2,1.0,3.6)
result <- optimal_binning_numerical_jedi_mwoe(target, feature,
               min_bins = 3, max_bins = 6, bin_cutoff = 0.05,
               max_n_prebins = 10, convergence_threshold = 1e-6,
               max_iterations = 100)
print(result)
}

}
