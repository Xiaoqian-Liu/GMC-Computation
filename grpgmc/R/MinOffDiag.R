#' \code{Proj_PSM} Project a matrix to the positive semidefinite matrix set
#'
#' @param D  the matrix we want to deal with
#' @param tol the minimum eigenvalue = 0
#' @return \code{D_new} The projected matrix

Proj_PSM <- function(D, tol=0){
  n <- nrow(D)

  # eigendecomposition of D
  result <- eigen(D, symmetric=TRUE)
  d <- result$values
  # cut the eigenvalues to make D positive semidefinite
  d_hat <- ifelse(d<tol,tol,d)

  U <- result$vectors
  D_new <- U%*%Diagonal(n,d_hat)%*%t(U)
  return(D_new)
}



#' \code{Proj_C2} Project a matrix D to the set M-D is positive semidefinite
#'
#' @param D the matrix we want to deal with
#' @param M the fixed M matrix
#' @param tol the minimum eigenvalue = 0
#' @return \code{D_new} The projected matrix
Proj_C2 <- function(D, M, tol=0){
  Y <- M-D
  Y_new <- Proj_PSM(Y, tol=tol)
  D_new <- M - Y_new
  return(D_new)
}



#' \code{Proj_C3} Project a matrix D onto the set where M-D <= epsilon for off-diagonal elements
#'
#' @param D the matrix we want to deal with
#' @param M the fixed M matrix
#' @param epsilon the largest value of the off-diagonal elements
#' @return \code{D_new} The projected matrix
Proj_C3 <- function(D, M, epsilon){
  y <- as.vector(M-D)
  p <- nrow(D)
  for (j in 1:p) {
    y[(j-1)*p + j] <- NA
  }
 # y <- y[!is.na(y)]

  y <- ifelse(abs(y)>epsilon, sign(y)*epsilon, y)

  d <- diag(M-D)
  for (j in 1:p) {
    y[(j-1)*p + j] <- d[j]
  }

  Y_new <- matrix(y, p, p)
  D_new <- M - Y_new
  return(D_new)
}



#' \code{MinOffDiag} The new method to find a feasible matrix D by controlling the off-diagonal elements
#'
#' @param D0 the initial matrix we want to deal with
#' @param U10 the initial value
#' @param U20 the initial value
#' @param U30 the initial value
#' @param M the fixed M matrix
#' @param epsilon the largest value of the off-diagonal elements
#' @param maxiters the maximal number of iteration
#' @param tol the tolerance to terminate the algorithm
#' @return \code{D} the feasible matrix D
#' @return \code{iters} the number of iteration used
#' @return \code{diff} the difference when terminate the algorithm
MinOffDiag <- function(D0, U10, U20, U30, M, epsilon, maxiters =1e3, tol = 1e-3){

  U1 <- U10
  U2 <- U20
  U3 <- U30

  D_avg <- D0
  p <- nrow(D0)

  for (k in 1:maxiters) {

    D1 <- Proj_PSM(D_avg - U1, tol=0)
    D2 <- Proj_C2(D_avg - U2, M, tol=0)
    D3 <- Proj_C3(D_avg - U3, M, epsilon)

    diff <- (norm(D1-D_avg, "2")+norm(D2-D_avg, "2")+norm(D3-D_avg, "2"))/norm(D_avg)
    if(diff>tol){
      U1 <- U1 + D1 - D_avg
      U2 <- U2 + D2 - D_avg
      U3 <- U3 + D3 - D_avg

      D_avg <- (D1+D2+D3)/3
    }else{
      break()
    }

  }

  return(list(D = D_avg, iters = k, diff= diff))


}
