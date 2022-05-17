
#' \code{objective_grGMC} computes the objective value of the group GMC problem at a specific solution point
#'
#' @param y The response variable
#' @param A The design matrix without an intercept.
#' @param Z The Z matrix in the group GMC problem
#' @param beta The specific solution of beta
#' @param v The specific solution of v
#' @param lambda The tunning parameter in grGMC
#' @param group A vector describing the grouping of the coefficients
#' @param group.multiplier The weight vector for the group GMC penalty
#' @author Xiaoqian Liu
#' @export
objective_grGMC <- function(y, A, Z, beta, v, lambda, group, group.multiplier){

  n <- nrow(A)
  p <- ncol(A)

  s1 <- norm(y-A%*%beta, "2")^2/(2*n)

  s2 <- 0
  s3 <- 0
  g <- unique(group)

  K <- group.multiplier
  J <- length(K)
  for (j in 1:J) {
    ind <- which(group==g[j])
    pj <- length(ind)
    Dj <- Matrix(0, nrow = pj, ncol = p, sparse = TRUE)
    for (i in 1:pj) {
      Dj[i, ind[i]] <- 1
    }
    s2 <- s2+ K[j]*norm(Dj%*%beta, "2")
    s3 <- s3+ K[j]*norm(Dj%*%v, "2")

  }

  s4 <- as.numeric(t(beta-v)%*%Z%*%(beta-v)/2)

  s <- as.numeric(s1+lambda*s2-lambda*s3-s4)
  return(s)
}
