#'Linear regression with group GMC
#'
#' \code{grGMC} fit regularization paths for linear regression models with the groupe GMC penalty
#' over a grid of values for the regularization parameter lambda.
#'
#' @param y The response variable
#' @param A The design matrix without an intercept. grGMC standardizes the data and includes an intercept by default.
#' @param theta The convexity-preserving parameter theta in the group GMC penalty
#' @param group A vector describing the grouping of the coefficients
#' @param lambda The tunning parameter in grGMC, usually a user supplied sequence of decrasing values.
#' @param group.multiplier The weight vector for the group GMC penalty
#' @param x0  Initial value for x, suggest use the last solution when having a sequence of lambda
#' @param v0 Initial value for v, similar with x0
#' @param Z.method Which method to set a matrix parameter Z = lambda B^TB/n
#' @param Z.epsilon The epsilon parameter in the new method of geenrating Z
#' @param maxiters The max number of iterations for the PDHG algorithm. The solution is computed by the
#'                adaptive PDHG algorithm using all the default values for the parameters in adaptive_pdhg function.
#' @param tol The relative decrease in the residuals before the adaptive PDHG algorithm stops
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
#' @return \code{iter} A vector containg the number of iterations
#' for the PDHG algorithm for each solution
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
#' @examples
#' set.seed(1234)
#' ## set the ture signal
#' x_star=c(0, 3, 9, 0, 0, 15, 0, 4, 5, 0)
#' p=length(x_star)
#' ##group the variables
#' group=c(2, 1, 1, 3, 2, 1, 3, 4, 4, 3)
#' ##set sample size
#' n=15
#' ## generate the covariate matrix
#' A=matrix(rnorm(n*p, mean = 1, sd=1), nrow = n)
#' ## set the group weights
#' gp=unique(group)
#' J=length(gp)
#' K=rep(0,J)
#' for (j in 1:J) {
#' K[j]=sqrt(length(which(group==gp[j])))
#' }
#' ## response vector
#' y=as.vector(A%*%x_star+rnorm(n, mean=0, sd = 0.5))
#' lambda=1
#' ## set initial values
#' x0=v0=double(p)
#' ## fit with grGMC function
#' fit_GMC <- grGMC(y=y, A=A, theta=0.9, group=group, lambda=lambda, group.multiplier=K, maxiters=1e3)
grGMC <- function( y, A, theta=0.9,  group=1:ncol(A),  lambda,  group.multiplier, x0, v0,
                   Z.method = "theta", Z.epsilon = 1e-3, maxiters = 1e3, tol = 1e-3,
                   df.method="group", returnX=FALSE, ShowTime=TRUE){

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


  if(Z.method == "theta"){
    Z <- theta*t(newA)%*%newA/n
  }else{
    M <- t(newA)%*%newA
    D0 <- matrix(rnorm(p*p), nrow = p, ncol = p)
    U10 <- matrix(0, nrow = p, ncol = p)
    U20 <- matrix(0, nrow = p, ncol = p)
    U30 <- matrix(0, nrow = p, ncol = p)

    res <- MinOffDiag(D0, U10, U20, U30, M, epsilon=Z.epsilon, maxiters, tol = 1e-8)
    D_new <- res$D
    # check
    values_D <- eigen(D_new)$values
    values_Diff <- eigen(M -D_new)$values
    if(all(round(values_D, 4) >=0) && all(round(values_Diff, 4) >=0)){
      Z <- D_new/n
    }else{
      stop("The Z matrix is not valid", call.=FALSE)
    }
  }
  #print(proc.time()-time)


  # initial value for the PDHG algorithm
  if(missing(x0) | missing(v0)){

    #x0 <- v0 <- rep(0, length(XG$nz))
    x0 <- v0 <- rep(0, p)

  } else{
    # x0 <- x0[XG$nz]
    # v0 <- v0[XG$nz]
    x0 <- x0[1:p]
    v0 <- v0[1:p]
  }



  xhat <- matrix(NA, nrow = p, ncol = NumLam)
  # bb <- matrix(0, nrow = ncol(A), ncol = L)
  df <- rep(0, NumLam)
  loss <- rep(0, NumLam)
  iter <- rep(0, NumLam)
  obj <- rep(0, NumLam)
  for (i in 1:NumLam) {

    #---------grGMC
    out_grGMC <- adaptive_pdhg(x0, v0, y=newy, A = newA, Z, group1, K1, lambda = lambda[i],
                          maxiters, tol)
    x0  <- as.vector(out_grGMC$x)
    v0 <- as.vector(out_grGMC$v)
    xhat[, i] <- x0
    # Here compute df and loss=residual sum of squares
    df[i] <- df_grGMC(newy, newA, x0, group=group1, df.method)
    loss[i] <- norm(newy-newA%*%x0, "2")^2

    iter[i] <- out_grGMC$iters
    obj[i] <- objective_grGMC(newy, newA, Z, x0, v0, lambda=lambda[i], group1, K1)
    # bb[XG$nz, i] <- x0  ## only for warm start if using  a sequence of lambda outside

  }

  b <- rbind(mean(y), xhat)   #add the intercept

  # Unstandardize
  b <- grpreg:::unorthogonalize(b, XG$X, XG$g)
  if (XG$reorder)  b[-1,] <- b[1+XG$ord.inv, ]
  beta <- grpreg:::unstandardize(b, XG)

  # Names
  varnames <- c("(Intercept)", XG$names)
  dimnames(beta) <- list(varnames, round(lambda, digits=4))

  val <- structure(list(beta = round(beta, 8),
                        family="gaussian",
                        penalty = "grGMC",
                        group = factor(group),
                        lambda = lambda,
                        theta=theta,
                        n = n,
                        iter=iter,
                        df=df,
                        loss=loss,
                        bb=xhat,
                        obj = obj,
                        group.multiplier = XG$m),
                        class="grGMC")

  if (returnX) {
    val$XG <- XG
    val$y <- newy
    val$Z <- Z
  }

  if(ShowTime){
    print(proc.time()-time)
  }

  return(val)

}
