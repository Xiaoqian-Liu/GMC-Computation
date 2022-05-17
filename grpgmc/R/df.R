#%%%%%%%%-------------------------------------------------------------------
#'  DOF function in "Penalized methods for bi-level variable selection"
#'  Patrick Breheny and Jian Huang, 2009
#'  Modified version
#'
#' @param y response, "centered"
#' @param A covariate matrix, "standardized"
#' @param x the current estimate, intercept is not included
#' @param group A vector describing the grouping of the coefficients
#' @param df.method which method to compute the degrees of freedom,
#' "group" means df=number of nonzero groups,
#' "active" means df=number of nonzero coefficients,
#' "Breheny" means using the formula in Breheny(2009)
#' @export

df_grGMC=function(y, A, x, group, df.method="group"){

  n=length(y)   # sample size
  p=length(x)   # dimension of the problem

  if(df.method=="group") {
    id=group[which(abs(x)>0)]
    df=length(unique(id))   #number of nonzero groups
  }

  if(df.method=="active"){
    df=length(which(abs(x)>0))    # number of nonzero coefficients
  }

  if (df.method=="Breheny" | missing(df.method)){
    b=rep(0,p)
    m=rep(0,p)
    for (i in 1:p) {

      #compute current partial residual
      y_tilde=y-A[, -i]%*%x[-i]

      m[i]=t(A[,i])%*%y_tilde/n

      b[i]=abs(x[i]/m[i])

      #truncate b into [-1,1]
      if(b[i]>1) b[i]=1
    }

    df=sum(b)
  }

  return(df+1)   # add the intercept
}
