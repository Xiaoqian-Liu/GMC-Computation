#' \code{obj_socp} computes the objective value of the group GMC problem as a SOCP
#'
#' @param y The response variable
#' @param A The design matrix without an intercept.
#' @param Z The Z matrix in the group GMC problem
#' @param beta The specific solution of beta
#' @param lambda The tunning parameter in grGMC
#' @param t The specific solution of t
#' @param group A vector describing the grouping of the coefficients
#' @param K The weight vector for the group GMC penalty = group.multiplier
#' @importFrom MASS ginv
#' @author Xiaoqian Liu
#' @export
obj_socp <- function(y, A, Z, beta, lambda, t, group, K){

  n <- length(y)
  p <- ncol(A)

  s1 <- norm(y-A%*%beta, "2")^2/(2*n)

  s2 <- 0

  g <- unique(group)
  J <- length(K)
  for (j in 1:J) {
    ind <- which(group==g[j])
    pj <- length(ind)
    Dj <- Matrix(0, nrow = pj, ncol = p, sparse = TRUE)
    for (i in 1:pj) {
      Dj[i, ind[i]] <- 1
    }
    s2 <- s2+ K[j]*norm(Dj%*%beta, "2")

  }

  Zinv <- ginv(Z)

  s3 <- as.numeric(t(beta)%*%Z%*%beta/2)

  a <- Z%*%beta
  s4 <- as.numeric(t(a-t)%*%Zinv%*%(a-t)/2)

  s <- s1+lambda*s2-s3+s4

  return(s)
}
