#' The SOCP solver for group GMC
#'
#' @param X The design matrix
#' @param y The response variable
#' @param Z The Z matrix in the group GMC problem
#' @param theta The convexity-preserving parameter theta in the group GMC penalty
#' @param lambda The tunning parameter in grGMC, usually a user supplied sequence of decrasing values.
#' @param group A vector describing the grouping of the coefficients
#' @param K The weight vector for the group GMC penalty = group.multiplier
#' @param maxiters The max number of iterations for the SOCP solver
#' @param tol The tolerance to terminate the SOCP solver
#' @importFrom MASS ginv Null
#' @import gurobi
#' @author Xiaoqian Liu
#' @export
socp_solver <- function(X, y, Z, theta, lambda, group, K, maxiters=1e4, tol=1e-12){

  # get the dimenions
  n <- length(y)
  p <- ncol(X)
  J <- length(K)


  Ip <- Diagonal(p, x=1)
  Zerop <-Diagonal(p, x=0)
  Q1 <- cbind(Ip, Zerop)
  Q2 <- cbind(Zerop, Ip)

  Zinv <- ginv(Z)

  M1 <- t(Q1)%*%t(X)%*%X%*%Q1/(2*n)
  M2 <- t(Q2)%*%Zinv%*%Z%*%Q1
  M3 <- t(Q2)%*%Zinv%*%Q2/2
  M <- M1 - M2 + M3
  Mbar <- bdiag(M, Diagonal(n=J, x=0))

  q <- c(as.vector(-t(Q1)%*%t(X)%*%y/n), lambda*K)

  # Q3 <- cbind(Z, -Ip)
  NZ <- Null(Z)
  if(ncol(NZ)==0){
    A <- Matrix(0, nrow = 1, ncol = 2*p+J)
  }else{

    # A = bdiag(t(NZ)%*%Q3, Diagonal(J, x=0))
     Q3 <- cbind(Z, -Ip, matrix(0, nrow = p, ncol = J))
     A <- Matrix(t(NZ)%*%Q3,sparse = TRUE)
  }



  model <- list()

  model$Q          <- Mbar
  model$A          <- A
  model$modelsense <- 'min'
  model$obj        <- q
  # model$rhs        <-
  model$sense      <- c('=')
  model$lb         <- c(rep(-Inf, 2*p), rep(0, J))
  #model$start     <- x0



  g <- unique(group)
  # Quadratic constraint for t and beta
  qc_t <- list()
  qc_b <- list()
  for (j in 1:J) {
    # Quadratic constraint for beta
    qc_b[[j]] <- list()

    ind <- which(group==g[j])
    pj <- length(ind)
    Dj <- Matrix(0, nrow = pj, ncol = p, sparse = TRUE)
    for (i in 1:pj) {
      Dj[i, ind[i]] <- 1
    }

    Aj <- Matrix(0, pj, 2*p+J, sparse = TRUE)
    Aj[, 1:(2*p)] <- Dj%*%Q1
    bj <- c(rep(0, 2*p+j-1), 1, rep(0, J-j))

    qc_b[[j]]$Qc <- t(Aj)%*%Aj - bj%*%t(bj)
    qc_b[[j]]$q <- rep(0, 2*p+J)
    qc_b[[j]]$rhs <- 0
    qc_b[[j]]$sense <- '<='  #could be "=", which is a nonconvex constratint, runtime increases a lot


    # Quadratic constraint for t
    qc_t[[j]] <- list()


    Cj <- Matrix(0, pj, 2*p+J)
    Cj[, 1:(2*p)] <- Dj%*%Q2


    qc_t[[j]]$Qc <- t(Cj)%*%Cj
    qc_t[[j]]$q <- rep(0, 2*p+J)
    qc_t[[j]]$rhs <- (lambda*K[j])^2
    qc_t[[j]]$sense <- '<='

  }


  model$quadcon <- list()
  for (j in 1:J) {
    model$quadcon[[j]] <- qc_b[[j]]
    model$quadcon[[J+j]] <- qc_t[[j]]
  }

  #==== output for "=" constraints for beta
  # params <- list(OutputFlag=0)
  # params$NonConvex <- 2
  # result <- gurobi(model, params)

  #== Normal output
  params <- list(OutputFlag=0, BarIterLimit = maxiters, BarQCPConvTol= tol)
  #===== optimization parameters in gurobi
  # params <- list(TimeLimit = 100, MIPGap = 1e-6, BarIterLimit = 1e4)
  # params <- list(TimeLimit = 100, FeasibilityTol=1e-9, OptimalityTol=1e-9, OutputFlag=0)
 # params <- list(OutputFlag=0, TimeLimit = 100, BarIterLimit = 1e4, BarQCPConvTol=1e-15)
 # params$NonConvex <- 2
  result <- gurobi(model, params)



  #print(result$objval)
  #print(result$x)
  beta <- Q1%*%(result$x[1:(2*p)])
  t <- Q2%*%(result$x[1:(2*p)])
  s <- result$x[(2*p+1): (2*p+J)]
  return(list(beta=as.vector(beta), t=as.vector(t), s=as.vector(s)))
}
