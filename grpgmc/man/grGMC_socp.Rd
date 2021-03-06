% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/grGMC_socp.R
\name{grGMC_socp}
\alias{grGMC_socp}
\title{Linear regression with group GMC as a SOCP}
\usage{
grGMC_socp(
  y,
  A,
  theta = 0.9,
  group = 1:ncol(A),
  lambda,
  group.multiplier,
  maxiters = 10000,
  tol = 1e-10,
  df.method = "group",
  returnX = FALSE,
  ShowTime = TRUE
)
}
\arguments{
\item{y}{The response variable}

\item{A}{The design matrix without an intercept. grGMC standardizes the data and includes an intercept by default.}

\item{theta}{The convexity-preserving parameter theta in the group GMC penalty}

\item{group}{A vector describing the grouping of the coefficients}

\item{lambda}{The tunning parameter in grGMC, usually a user supplied sequence of decrasing values.}

\item{group.multiplier}{The weight vector for the group GMC penalty}

\item{maxiters}{The max number of iterations for the SOCP solver.}

\item{tol}{The tolerance to terminate the SOCP solver.}

\item{df.method}{Method to compute degrees of freedom, "group", "active" or "Breheny"}

\item{returnX}{Whether to return the standardized data}

\item{ShowTime}{Whether to show the time cost for computing the solution}
}
\value{
\code{beta} The fitted matrix of coefficients.
The number of rows is equal to the number of coefficients,
and the number of columns is equal to the length of the sequence of lambda.

\code{family} Only "gaussian" is allowed for now.

\code{penalty} We call our penalty as grGMC.

\code{group} Same as above

\code{lambda} The sequence of lambda values in the path.

\code{theta} The theta parameter used in group GMC penalty.

\code{n} Number of observations.

\code{df} A vector containg the degrees of freedom for
 all the points along the regularization path.

\code{loss} A vector containing the residual sum of squares of
 the fitted model at each value of lambda.

\code{bb} The fitted matrix of coefficients for the standardized data.
 This return is used as a warm start when having a sequence of lambda.

\code{obj} The vector of objective values at each solution point

\code{group.multiplier} A named vector containing the multiplicative constant
 applied to each group's penalty.
}
\description{
\code{grGMC_socp} fit the group GMC path by cast the optimization problem as a SOCP
}
\author{
Xiaoqian Liu
}
