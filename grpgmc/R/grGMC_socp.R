#'Linear regression with group GMC as a SOCP
#'
#' \code{grGMC_socp} fit the group GMC path by cast the optimization problem as a SOCP
#'
#' @param y The response variable
#' @param A The design matrix without an intercept. grGMC standardizes the data and includes an intercept by default.
#' @param theta The convexity-preserving parameter theta in the group GMC penalty
#' @param group A vector describing the grouping of the coefficients
#' @param lambda The tunning parameter in grGMC, usually a user supplied sequence of decrasing values.
#' @param group.multiplier The weight vector for the group GMC penalty
#' @param maxiters The max number of iterations for the SOCP solver.
#' @param tol The tolerance to terminate the SOCP solver.
#' @param df.method Method to compute degrees of freedom, "group", "active" or "Breheny"
#' @param returnX Whether to return the standardized data
#' @param ShowTime Whether to show the time cost for computing the solution
#' @return \code{beta} The fitted matrix of coefficients.
#' The number of rows is equal to the number of coefficients,
#' and the number of columns is equal to the length of the sequence of lambda.
#' @return \code{family} Only "gaussian" is allowed for now.
#' @return \code{penalty} We call our penalty as grGMC.
#' @return \code{group} Same as above
#' @return \code{lambda} The sequence of lambda values in the path.
#' @return \code{theta} The theta parameter used in group GMC penalty.
#' @return \code{n} Number of observations.
#' @return \code{df} A vector containg the degrees of freedom for
#'  all the points along the regularization path.
#' @return \code{loss} A vector containing the residual sum of squares of
#'  the fitted model at each value of lambda.
#' @return \code{bb} The fitted matrix of coefficients for the standardized data.
#'  This return is used as a warm start when having a sequence of lambda.
#' @return \code{obj} The vector of objective values at each solution point
#' @return \code{group.multiplier} A named vector containing the multiplicative constant
#'  applied to each group's penalty.
#' @author Xiaoqian Liu
#' @export
grGMC_socp <- function (y, A, theta=0.9, group = 1:ncol(A), lambda, group.multiplier,
          maxiters = 1e4, tol = 1e-10,  df.method = "group", returnX = FALSE,  ShowTime = TRUE)
{
  time <- proc.time()

  #construct new X and y as "grpreg"
  newy <- grpreg:::newY(y, family="gaussian")
  XG <- grpreg:::newXG(A, g=group, group.multiplier, attr(newy, 'm'), bilevel=FALSE)
  newA <- XG$X
  group1 <- XG$g      #new group vector
  K1 <- XG$m          #new group multiplier

  if (nrow(newA) != length(newy))
    stop("X and y do not have the same number of observations", call.=FALSE)


  # Fit
  n <- length(newy)
  p <- ncol(newA)

  NumLam <- length(lambda)
  Z <- theta * t(newA) %*% newA/n


  xhat <- matrix(NA, nrow = p, ncol = NumLam)
  # bb <- matrix(0, nrow = ncol(A), ncol = L)
  df <- rep(0, NumLam)
  loss <- rep(0, NumLam)
  #iter <- rep(0, NumLam)
  obj <- rep(0, NumLam)
  for (i in 1:NumLam) {

    #---------grGMC
    out_grGMC <- socp_solver(newA, newy, Z, theta, lambda=lambda[i], group1, K1, maxiters, tol)

    x0 <- as.vector(out_grGMC[[1]])
    xhat[, i] <- x0
    # Here compute df and loss=residual sum of squares
    df[i] <- df_grGMC(newy, newA, x0, group = group1, df.method)
    loss[i] <- norm(newy - newA %*% x0, "2")^2
   # iter[i] <- out_grGMC[[3]]
    obj[i] <- obj_socp(newy, newA, Z, x0, lambda=lambda[i], t =out_grGMC[[2]] , group1, K1)
 }

  b <- rbind(mean(y), xhat) #add the intercept

  # Unstandardize
  b <- grpreg:::unorthogonalize(b, XG$X, XG$g)
  if (XG$reorder)  b[-1,] <- b[1+XG$ord.inv, ]
  beta <- grpreg:::unstandardize(b, XG)

  # Names
  varnames <- c("(Intercept)", XG$names)
  dimnames(beta) <- list(varnames, round(lambda, digits = 4))

  val <- structure(list(beta = round(beta, 8),
                        family = "gaussian",
                        penalty = "grGMC",
                        group = factor(group),
                        lambda = lambda,
                        theta = theta,
                        n = n,
                        df = df,
                        loss = loss,
                        bb = xhat,
                        obj = obj,
                        group.multiplier = XG$m),
                        class = "grGMC")
  if (returnX) {
    val$XG <- XG
    val$y <- newy
    val$Z <- Z
  }
  if (ShowTime) {
    print(proc.time() - time)
  }
  return(val)
}
