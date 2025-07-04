% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/obwoe.R
\name{OBPreprocessData}
\alias{OBPreprocessData}
\title{Preprocess Data for Optimal Binning}
\usage{
OBPreprocessData(dt, target, features, control, preprocess = "both")
}
\arguments{
\item{dt}{A data.table containing the dataset.}

\item{target}{Target name}

\item{features}{Vector of feature names to process.}

\item{control}{A list of control parameters.}

\item{preprocess}{Preprocess feature. 'both' feature and report. Can also be 'both' or 'feature'}
}
\value{
A list of preprocessed data for each feature.
}
\description{
Preprocess Data for Optimal Binning
}
