% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{optimal_binning_categorical_milp}
\alias{optimal_binning_categorical_milp}
\title{Optimal Binning for Categorical Variables using OBNP}
\usage{
optimal_binning_categorical_milp(
  target,
  feature,
  min_bins = 3L,
  max_bins = 5L,
  bin_cutoff = 0.05,
  max_n_prebins = 20L
)
}
\arguments{
\item{target}{An integer vector of binary target values (0 or 1).}

\item{feature}{A character vector of categorical feature values.}

\item{min_bins}{Minimum number of bins (default: 3).}

\item{max_bins}{Maximum number of bins (default: 5).}

\item{bin_cutoff}{Minimum proportion of observations in a bin (default: 0.05).}

\item{max_n_prebins}{Maximum number of pre-bins (default: 20).}
}
\value{
A list containing two elements:
\itemize{
\item woefeature: A numeric vector of Weight of Evidence (WoE) values for each observation
\item woebin: A data frame containing binning information, including bin names, WoE, Information Value (IV), and counts
}
}
\description{
This function performs optimal binning for categorical variables using the Optimal Binning Numerical Procedures (OBNP) approach.
The process aims to maximize the Information Value (IV) while maintaining a specified number of bins.
}
\details{
The algorithm works as follows:
\enumerate{
\item Merge rare categories: Categories with fewer observations than the specified bin_cutoff are merged into an "Other" category.
\item Create initial bins: Each unique category is assigned to its own bin, up to max_n_prebins.
\item Optimize bins:
a. Calculate WoE and IV for each bin.
b. Enforce monotonicity when possible, merging bins as needed, unless min_bins is reached.
c. Limit the number of bins to be within min_bins and max_bins.
\item Transform the feature: Assign WoE values to each observation based on its category.
}

The Weight of Evidence (WoE) is calculated as:
\deqn{WoE = \ln\left(\frac{\text{\% of events}}{\text{\% of non-events}}\right)}

The Information Value (IV) is calculated as:
\deqn{IV = (\text{\% of events} - \text{\% of non-events}) \times WoE}
}
\examples{
\dontrun{
# Create sample data
target <- sample(0:1, 1000, replace = TRUE)
feature <- sample(LETTERS[1:5], 1000, replace = TRUE)

# Run optimal binning
result <- optimal_binning_categorical_obnp(target, feature)

# View results
print(result$woebin)
}

}
\references{
\itemize{
\item Belotti, T., Crook, J. (2009). Credit Scoring with Macroeconomic Variables Using Survival Analysis.
Journal of the Operational Research Society, 60(12), 1699-1707.
\item Thomas, L. C. (2000). A survey of credit and behavioural scoring: forecasting financial risk of lending to consumers.
International Journal of Forecasting, 16(2), 149-172.
}
}
