% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/obwoe.R
\name{OBValidateInputs}
\alias{OBValidateInputs}
\title{Validate Inputs for Optimal Binning}
\usage{
OBValidateInputs(
  dt,
  target,
  features,
  method,
  preprocess,
  min_bins,
  max_bins,
  control,
  positive
)
}
\arguments{
\item{dt}{A data.table containing the dataset.}

\item{target}{The name of the target variable.}

\item{features}{Vector of feature names to process.}

\item{method}{The binning method to use.}

\item{preprocess}{Logical. Whether to preprocess the data before binning.}

\item{min_bins}{Minimum number of bins.}

\item{max_bins}{Maximum number of bins.}

\item{control}{A list of additional control parameters.}

\item{positive}{Character string specifying which category should be considered as positive.}
}
\value{
None. Throws an error if any input is invalid.
}
\description{
Validate Inputs for Optimal Binning
}
\keyword{internal}
