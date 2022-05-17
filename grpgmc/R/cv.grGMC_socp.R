#'Cross-validation for grouped variable selection with GMC penalization
#'
#' \code{cv.grGMC_socp} performs k-fold cross-validation for penalizaed linear regression models with the groupe GMC penalty
#' over a grid of values for the regularization parameter lambda using the SOCP solver.
#'
#' @param y The response variable
#' @param A The design matrix without an intercept. grGMC standardizes the data and includes an intercept by default.
#' @param theta The convexity-preserving parameter theta in the group GMC penalty
#' @param group A vector describing the grouping of the coefficients
#' @param lambda The tunning parameter in grGMC, usually a user supplied sequence of decrasing values.
#' @param group.multiplier The weight vector for the group GMC penalty
#' @param nfolds The number of cross-validation folds. Default is 10.
#' @param seed Users can set the seed of the random number generator to obtain reproducible results.
#' @param maxiters The maxiters parameter in grGMC
#' @param tol The tol parameter in grGMC
#' @param returnX Whether to return the standardized data
#' @param returnY Whether to return the fitted values from the cross-validation folds
#' @param trace Whether to inform the users the progress of the cross-validation by announcing the beginning of each CV fold
#' @return \code{cve} The error for each value of lambda, averaged across the cross-validation folds.
#' @return \code{cvse} The estimated standard error associated with each value of for cve.
#' @return \code{lambda} The sequence of regularization parameter values along which the cross-validation error was calculated.
#' @return \code{fit} The fitted grpreg object for the whole data.
#' @return \code{fold} The fold assignments for cross-validation for each observation.
#' @return \code{min} The index of lambda corresponding to lambda.min.
#' @return \code{min_1se} The index of lambda corresponding to lambda.1se.
#' @return \code{lambda.min} The value of lambda with respect to the minimum cross-validation error.
#' @return \code{lambda.1se} The value of lambda selected by the "one-standard-error" rule.
#' @author Xiaoqian Liu
#' @export

cv.grGMC_socp <- function(y, A, theta, group=1:ncol(A), lambda, group.multiplier,
                     nfolds=10, seed=1234, maxiters = 1e4, tol = 1e-5,
                     returnX=TRUE, returnY=FALSE, trace=FALSE) {

  # Complete data fit
  if(missing(lambda))
  {
    fit <- grpreg(A, y, group=group, penalty = "grLasso", returnX = TRUE)
  }else{
    fit <- grpreg(A, y, group=group, lambda=lambda, penalty = "grLasso", returnX = TRUE)
  }


  # Get standardized X, y
  XG <- fit$XG
  newA <- XG$X
  newy <- fit$y
  m <- attr(fit$y, "m")
  if (is.null(returnX) || !returnX) fit$XG <- NULL

  # Set up folds
  if (!missing(seed)) set.seed(seed)
  n <- length(y)
  fold <- sample(1:n %% nfolds)
  fold[fold==0] <- nfolds



  # Do cross-validation
  E <- Y <- matrix(NA, nrow=length(y), ncol=length(fit$lambda))

  cv.args <- list()
  cv.args$theta <- theta
  cv.args$lambda <-  fit$lambda
  cv.args$group <- XG$g
  cv.args$group.multiplier <- XG$m
  cv.args$returnX <- returnX
  cv.args$maxiters <- maxiters
  cv.args$tol <- tol
  cv.args$ShowTime <- FALSE


  for (i in 1:nfolds) {
    if (trace) cat("Starting CV fold #", i, sep="","\n")
    res <- cv_fold_socp(i, newy, newA, fold, cv.args)
    Y[fold==i, 1:res$nl] <- res$yhat
    E[fold==i, 1:res$nl] <- res$loss
  }

  # Eliminate saturated lambda values, if any
  ind <- which(apply(is.finite(E), 2, all))
  E <- E[, ind, drop=FALSE]
  Y <- Y[, ind]
  lambda <- fit$lambda[ind]

  # Return
  cve <- apply(E, 2, mean)
  cvse <- apply(E, 2, sd) / sqrt(n)
  min <- which.min(cve)


  #find the lambda.1se
  for (k in min:1) {
    if(cve[k]>cve[min]+cvse[min])
      break
  }
  lambda.1se <- lambda[k+1]
  min_1se <- k+1

  val <- list(cve=cve, cvse=cvse, lambda=lambda, fit=fit, fold=fold, min=min,
              min_1se=min_1se, lambda.min=lambda[min], lambda.1se=lambda.1se)
  if (returnY) {
    val$Y <- Y + attr(y, "mean")
  }
  structure(val, class="cv.grGMC")
}




cv_fold_socp <- function(i, y, A, fold, cv.args) {
  cv.args$A <- A[fold!=i, , drop=FALSE]
  cv.args$y <- y[fold!=i]
  fit.i <- do.call("grGMC_socp", cv.args)

  A2 <- A[fold==i, , drop=FALSE]
  y2 <- y[fold==i]

  L <- length(fit.i$lambda)
  loss <- yhat <- matrix(0, nrow = length(y2), ncol = L)

  for (l in 1:L) {
    yhat[, l] <-  A2%*%fit.i$beta[-1, l]
    loss[,l] <- (y2-yhat[, l])^2
  }

  list(loss=loss, nl=length(fit.i$lambda), yhat=yhat)
}
