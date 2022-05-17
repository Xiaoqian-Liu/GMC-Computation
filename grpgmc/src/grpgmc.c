#include <R.h>
#include <Rmath.h>
#include <math.h>
#include <stdio.h>
#include <R_ext/BLAS.h>
#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

typedef struct
{
    int Nrow, Ncol;
    double *data;
} matrix;

/************************************ Basic functions ********************************/
/* calculate the norm of a vector x */
void vec_norm(double *x, int *length, double *xnorm);

/*Inner procudt of vector x and y with length n*/
void innerprod(double *x, double *y, int *n, double *prod);

/* Matrix-vector multiplication, prod=Ax */
void mvprod(matrix A, double *x, double *prod);

/*Matrix transpose-vector multiplication, prod=A'x */
void mtvprod(matrix A, double *x, double *prod);

/* 
Get all the distinct elements in the array :group 
group -  an array of int, which presents the group index for each predictor in group GMC
p - dimension of group
d - dimension of the output
*/
void unique(int *group, int *p, int *d, int *g);

/* 
Get indices of elements  equal to a specific value in an array
group -  an array of int, which presents the group index for each predictor in group GMC
p - dimension of group
j - the specific value we want to seach the indices for 
l - dimension of the output
*/
void getindices(int *group, int *p, int *j, int *l, int *ind);

/* Proximal operator of f=tau*||x||_2 */
void prox_L2(double *x, double *tau, int *length, double *x_prox);

/************************************ Objective/gradient functions ********************************/
/*
Objective function f1(x)=1/2n*\|y-Ax\|_2^2 - x'Zx/2 + 1/2 param* \|x-var_hat\|_2^2
x - the current estimate, of length p
y - the response variable, of length n
A - the  A matrix in grGMC, n-by-p
Z - the Z matrix in grGMC penalty, p-by-p
var_hat - xhat in PDHG update, of length p
param - parameter tau_k for updating x_{k+1}
fx - the value of the objective function at x (output)
*/
void F1(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *fx);

/*
Gradient function gradf1(x)=1/n*A'(Ax-y) - Zx + 1/param* (x-var_hat)
x - the current estimate, of length p
y - the response variable, of length n
A - the  A matrix in grGMC, n-by-p
Z - the Z matrix in grGMC penalty, p-by-p
var_hat - xhat in PDHG update, of length p
param - parameter tau_k for updating x_{k+1}
gradfx - the value of the objective function at x (output)
*/
void GradF1(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *gradfx);

/*
Objective function f2(x)=x'Zx/2 + 1/2 param* \|x-var_hat\|_2^2
x - the current estimate for x, of length p
A - the  A matrix in grGMC, n-by-p
Z - the Z matrix in grGMC penalty, p-by-p
var_hat - xhat in PDHG update, of length p
param - parameter tau_k for updating x_{k+1}
fv - the value of the objective function at x (output)
*/
void F2(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *fx);

/*
Gradient function gradf2(x)=Zx + 1/param* (x-var_hat)
x - the current estimate, of length p
y - the response variable, of length n
A - the  A matrix in grGMC, n-by-p
Z - the Z matrix in grGMC penalty, p-by-p
var_hat - xhat in PDHG update, of length p
param - parameter tau_k for updating x_{k+1}
gradfx - the value of the objective function at x (output)
*/
void GradF2(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *gradfx);

/*
Objective function g(x)=lambda sum_{j=1}^J K_j \|x_j\|_2
x - the current estimate for x, of length p
group - the group indicator
p - the length  of x and group
K - the group weights or multipliers
lambda - the tuning parameter in group GMC
gx - the value of the objective function at x (output)
*/
void G1(double *x, int *group, int *p, double *K, double *lambda, double *gx);

/************************************ Proximal functions ********************************/
/*
Proximal operator of tau*g(x)=lambda sum_{j=1}^J K_j \|x_j\|_2
x - the current estimate for x, of length p
group - the group indicator
p - the length  of x and group
K - the group weights or multipliers
lambda - the tuning parameter in group GMC
tau - the multiplier of g(x)
proxgx - the value of the proximal operator at x (output)
*/
void ProxG1(double *x, int *group, int *p, double *K, double *lambda, double *tau, double *proxgx);

/************************************ FASTA  Algorithm ********************************/
/*
For the PDHG updates in group GMC

*/
void fasta(void (*f)(double *, double *, matrix, matrix, double *, double *, double *),
           void (*gradf)(double *, double *, matrix, matrix, double *, double *, double *),
           void (*g)(double *, int *, int *, double *, double *, double *),
           void (*proxg)(double *, int *, int *, double *, double *, double *, double *),
           double *x0, double *y, matrix A, matrix Z, double *var_hat, double *param,
           int *group, double *K, double *lambda,
           double *tau1, int *max_iters, int *w, int *backtrack,
           int *recordIterates, double *stepsizeShrink, double *eps_n,
           double *bestObjectiveIterate, double *objective, double *fVals,
           int *totalBacktracks, double *residual, double *taus, matrix iterates, int *NumTotalIters);

/*
 calculate the norm of a vector
 x - the vector
 length - the length of x
 xnorm - the norm of x
*/
void vec_norm(double *x, int *length, double *xnorm)
{

    if (*length <= 0)
    {
        fprintf(stderr, "%s", "length of the vector must be positive!\n");
    }
    const int one = 1;
    *xnorm = dnrm2_(length, x, &one);
}

void test_vecnorm(double *x, int *n, double *xnorm)
{
    vec_norm(x, n, xnorm);
}

/*
Inner procudt of vector x and y with length n
*/
void innerprod(double *x, double *y, int *n, double *prod)
{
    const int one = 1;
    *prod = ddot_(n, x, &one, y, &one);
}

void mvprod(matrix A, double *x, double *prod)
{
    int m = A.Nrow;
    int n = A.Ncol;

    const int one = 1;
    const char TRANS = 'N';
    double alpha = 1;
    double beta = 0;

    dgemv_(&TRANS, &m, &n, &alpha, A.data, &m, x, &one, &beta, prod, &one);
}

void test_mvprod(double *a, double *x, int *m, int *n, double *prod)
{
    matrix A;
    A.Nrow = *m;
    A.Ncol = *n;
    A.data = a;

    mvprod(A, x, prod);
}

void mtvprod(matrix A, double *x, double *prod)
{
    int m = A.Nrow;
    int n = A.Ncol;

    const int one = 1;
    const char TRANS = 'T';
    double alpha = 1;
    double beta = 0;

    dgemv_(&TRANS, &m, &n, &alpha, A.data, &m, x, &one, &beta, prod, &one);
}

void test_mtvprod(double *a, double *x, int *m, int *n, double *prod)
{
    matrix A;
    A.Nrow = *m;
    A.Ncol = *n;
    A.data = a;
    mtvprod(A, x, prod);
}



/* 
Get all the distinct elements in the array :group 
group -  an array of int, which presents the group index for each predictor in group GMC
p - dimension of group
d - useful dimension of the output
*/
void unique(int *group, int *p, int *d, int *g)
{

    g[0] = group[0];
    int k = 0;
    for (int i = 1; i < (*p); i++)
    {
        int j;
        for (j = 0; j < i; j++)
        {
            if (group[i] == group[j])
                break;
        }
        if (i == j)
        {
            g[k + 1] = group[i];
            k = k + 1;
        }
    }

    *d = k + 1;
}


/* 
Get indices of elements  equal to a specific value in an array
group -  an array of int, which presents the group index for each predictor in group GMC
p - dimension of group
j - the specific value we want to seach the indices for 
l - dimension of the output
*/
void getindices(int *group, int *p, int *j, int *l, int *ind)
{

    int k = 0;
    for (int i = 0; i < *p; i++)
    {
        if (group[i] == *j)
        {
            ind[k] = i;
            k = k + 1;
        }
    }
    *l = k;
}

/*
  Proximal operator of f=tau*||x||_2 
*/
void prox_L2(double *x, double *tau, int *length, double *x_prox)
{

    double c, xnorm;
    if (*tau < 0)
    {
        fprintf(stderr, "%s", "tau must be nonnegative!\n");
    }

    if (*length <= 0)
    {
        fprintf(stderr, "%s", "length of the vector must be positive!\n");
    }

    vec_norm(x, length, &xnorm);

    if (xnorm > *tau)
    {
        c = 1 - (*tau) / xnorm;
        for (int i = 0; i < *length; i++)
        {
            x_prox[i] = c * x[i];
        }
    }
    else
    {
        for (int i = 0; i < *length; i++)
            x_prox[i] = 0;
    }
}

/************************************ Objective/gradient/proximal functions ********************************/

void F1(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *fx)
{
    int n = A.Nrow;
    int p = A.Ncol;
    // get the matrix-vector multiplication Ax
    double *prod = (double *)calloc(n, sizeof(double));
    mvprod(A, x, prod);
    // get y-Ax
    double *a1 = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        a1[i] = y[i] - prod[i];
    }
    double f1, f2, f3;
    vec_norm(a1, &n, &f1);

    double *a2 = (double *)calloc(p, sizeof(double));
    mvprod(Z, x, a2);
    innerprod(x, a2, &p, &f2);

    double *a3 = (double *)calloc(p, sizeof(double));
    for (int i = 0; i < p; i++)
    {
        a3[i] = x[i] - var_hat[i];
    }
    vec_norm(a3, &p, &f3);

    double result = f1 * f1 / (2 * n) - 0.5 * f2 + f3 * f3 / (2 * (*param));
    *fx = result;

    free(prod);
    free(a1);
    free(a2);
    free(a3);
}

void test_f1(double *x, double *y, double *a, double *z, int *n, int *p,
             double *var_hat, double *param, double *fx)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = a;

    matrix Z;
    Z.Nrow = *p;
    Z.Ncol = *p;
    Z.data = z;

    F1(x, y, A, Z, var_hat, param, fx);
}

void GradF1(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *gradfx)
{
    int n = A.Nrow;
    int p = A.Ncol;
    // get the matrix-vector multiplication Ax
    double *prod = (double *)calloc(n, sizeof(double));
    mvprod(A, x, prod);
    // get Ax-y
    double *a1 = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        a1[i] = prod[i] - y[i];
    }
    // get the matrix transpose-vector multiplication A'(y-Ax)
    double *f1 = (double *)calloc(p, sizeof(double));
    mtvprod(A, a1, f1);

    // get the matrix-vector multiplication Zx
    double *f2 = (double *)calloc(p, sizeof(double));
    mvprod(Z, x, f2);

    for (int i = 0; i < p; i++)
    {
        gradfx[i] = f1[i] / n - f2[i] + (x[i] - var_hat[i]) / (*param);
    }

    free(prod);
    free(a1);
    free(f1);
    free(f2);
}

void test_gradf1(double *x, double *y, double *a, double *z, int *n, int *p,
                 double *var_hat, double *param, double *gradfx)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = a;

    matrix Z;
    Z.Nrow = *p;
    Z.Ncol = *p;
    Z.data = z;

    GradF1(x, y, A, Z, var_hat, param, gradfx);
}

/*
Objective function f2(x)=v'Zv/2 + 1/2 param* \|v-var_hat\|_2^2
v - the current estimate for v, of length p
A - the  A matrix in grGMC, n-by-p
Z - the Z matrix in grGMC penalty, p-by-p
var_hat - xhat in PDHG update, of length p
param - parameter tau_k for updating x_{k+1}
fv - the value of the objective function at x (output)
*/
void F2(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *fx)
{

    int p = A.Ncol;

    double f1, f2;

    double *a1 = (double *)calloc(p, sizeof(double));
    mvprod(Z, x, a1);
    innerprod(x, a1, &p, &f1);

    double *a2 = (double *)calloc(p, sizeof(double));
    for (int i = 0; i < p; i++)
    {
        a2[i] = x[i] - var_hat[i];
    }
    vec_norm(a2, &p, &f2);

    double result = 0.5 * f1 + f2 * f2 / (2 * (*param));
    *fx = result;

    free(a1);
    free(a2);
}

void test_f2(double *x, double *y, double *a, double *z, int *n, int *p,
             double *var_hat, double *param, double *fx)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = a;

    matrix Z;
    Z.Nrow = *p;
    Z.Ncol = *p;
    Z.data = z;

    F2(x, y, A, Z, var_hat, param, fx);
}

/*
Gradient function gradf2(x)=Zx + 1/param* (x-var_hat)
gradfx - the value of the objective function at x (output)
*/
void GradF2(double *x, double *y, matrix A, matrix Z, double *var_hat, double *param, double *gradfx)
{

    int p = A.Ncol;

    // get the matrix-vector multiplication Zx
    double *f1 = (double *)calloc(p, sizeof(double));
    mvprod(Z, x, f1);

    for (int i = 0; i < p; i++)
    {
        gradfx[i] = f1[i] + (x[i] - var_hat[i]) / (*param);
    }

    free(f1);
}

void test_gradf2(double *x, double *y, double *a, double *z, int *n, int *p,
                 double *var_hat, double *param, double *gradfx)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = a;

    matrix Z;
    Z.Nrow = *p;
    Z.Ncol = *p;
    Z.data = z;

    GradF2(x, y, A, Z, var_hat, param, gradfx);
}

void G1(double *x, int *group, int *p, double *K, double *lambda, double *gx)
{

    // get the distinct elements in group and its length d
    int d = 0;
    int *g = (int *)calloc(*p, sizeof(int));
    unique(group, p, &d, g);

    double *xnorm = (double *)calloc(d, sizeof(double));

    //=======================get out from for loop==========
    int *ind = (int *)calloc(*p, sizeof(int));
    double *x_temp = (double *)calloc(*p-d+1, sizeof(double)); // The maximal length of x_temp is p-d+1, since there are d distinct elements in x
    //======================================================
    int l = 0;
    for (int i = 0; i < d; i++)
    {
        // get all the indices of elements==g[i] and its length l
        // int l = 0;
        //=== int *ind = (int *)calloc(*p, sizeof(int));
        getindices(group, p, &g[i], &l, ind);
        // get the coresponding elements in x
        //====== double *x_temp = (double *)calloc(l, sizeof(double));
        for (int j = 0; j < l; j++)
        {
            x_temp[j] = x[ind[j]];
        }
        vec_norm(x_temp, &l, &xnorm[i]);

        //  free(x_temp);
       // free(ind); ///-------*******--------////
    }
    double prod = 0;
    innerprod(K, xnorm, &d, &prod);
    double result = 0;
    result = (*lambda) * prod;

    *gx = result;
    free(xnorm);
    free(g); ///-------*******--------////
    //=======================
    free(ind); ///-------*******--------////
    free(x_temp);
    //=======================
}

void test_g(double *x, int *group, int *p, double *K, double *lambda, double *gx)
{
    G1(x, group, p, K, lambda, gx);
}

void ProxG1(double *x, int *group, int *p, double *K, double *lambda, double *tau, double *proxgx)
{
    // get the distinct elements in group and its length d
    int d = 0;
    int *g = (int *)calloc(*p, sizeof(int));
    unique(group, p, &d, g);

    //=======================get out from for loop==========
    int *ind = (int *)calloc(*p, sizeof(int));
    double *x_temp = (double *)calloc(*p-d+1 , sizeof(double)); // The maximal length of x_temp is p-d+1, since there are d distinct elements in x
    double *x_temp_prox = (double *)calloc(*p-d+1, sizeof(double)); // x_temp_prox has the same length with x_temp
    //======================================================
    int l = 0;
    for (int i = 0; i < d; i++)
    {
        // get all the indices of elements==g[i] and its length l
        //=== int l = 0;
        //=== int *ind = (int *)calloc(*p, sizeof(int));
        getindices(group, p, &g[i], &l, ind);
        // get the coresponding elements in x
        //===== double *x_temp = (double *)calloc(l, sizeof(double));
        for (int j = 0; j < l; j++)
        {
            x_temp[j] = x[ind[j]];
        }
        //get the proximal operator of x_temp
        //=====  double *x_temp_prox = (double *)calloc(l, sizeof(double));
        double t = (*tau) * (*lambda) * K[i];
        prox_L2(x_temp, &t, &l, x_temp_prox);

        // assign the corresponding elements to proxgx
        for (int j = 0; j < l; j++)
        {
            proxgx[ind[j]] = x_temp_prox[j];
        }
        // free(x_temp);
        // free(x_temp_prox);
        //=== free(ind); ///-------*******--------////
    }

    free(g); ///-------*******--------////
    //=======================
    free(ind); ///-------*******--------////
    free(x_temp);
    free(x_temp_prox);
    //=======================
}

void test_proxg(double *x, int *group, int *p, double *K, double *lambda, double *tau, double *proxgx)
{
    ProxG1(x, group, p, K, lambda, tau, proxgx);
}


/************************************ FASTA  Algorithm ********************************/
void fasta(void (*f)(double *, double *, matrix, matrix, double *, double *, double *),
           void (*gradf)(double *, double *, matrix, matrix, double *, double *, double *),
           void (*g)(double *, int *, int *, double *, double *, double *),
           void (*proxg)(double *, int *, int *, double *, double *, double *, double *),
           double *x0, double *y, matrix A, matrix Z, double *var_hat, double *param,
           int *group, double *K, double *lambda,
           double *tau1, int *max_iters, int *w, int *backtrack,
           int *recordIterates, double *stepsizeShrink, double *eps_n,
           double *bestObjectiveIterate, double *objective, double *fVals,
           int *totalBacktracks, double *residual, double *taus, matrix iterates, int *TotalIters)
{

    // Allocate memory
    //*********----Those commented will be input------***********//
    //double *residual = (double *) calloc (*max_iters, sizeof(double));          // Residuals
    double *normalizedResid = (double *)calloc(*max_iters, sizeof(double)); // Normalized residuals
    //double *taus = (double *) calloc (*max_iters, sizeof(double));              // Stepsizes
    //double *fVals = (double *) calloc (*max_iters, sizeof(double));             // The value of 'f', the smooth objective term
    //double *objective = (double *) calloc (*max_iters+1, sizeof(double));       // The value of the objective function (f+g)
    //int totalBacktracks = 0;     // How many times was backtracking activated?
    int backtrackCount = 0; // Backtracks on this iterations

    // Intialize array values
    int p = A.Ncol; // dimension of the problem
    double *x1 = (double *)calloc(p, sizeof(double));
    double *d1 = (double *)calloc(p, sizeof(double));
    for (int j = 0; j < p; j++)
    {
        x1[j] = x0[j]; //  x1 <- x0
        d1[j] = x1[j]; //  d1 <- x1
    }

    double f1;
    f(d1, y, A, Z, var_hat, param, &f1); // f1  <- f(d1)
    fVals[0] = f1;                       // fVals[1] <- f1

    double *gradf1 = (double *)calloc(p, sizeof(double));
    gradf(d1, y, A, Z, var_hat, param, gradf1); // gradf1 <- gradf(d1)

    //matrix iterates;    --------******** Here is an input **********---------
    //iterates.Nrow = p;
    //iterates.Ncol = *max_iters+1;
    //iterates.data = (double *) calloc (p*iterates.Ncol, sizeof(double));
    if (*recordIterates)
    {
        // get the first column of iterates  , iterates[,1] <- x1
        for (int j = 0; j < p; j++)
            iterates.data[j] = x1[j];
    }
    else
    {
        iterates.data = NULL;
    }

    // The handle non-monotonicity
    double maxResidual = -INFINITY;      // Stores the maximum value of the residual that has been seen. Used to evaluate stopping conditions.
    double minObjectiveValue = INFINITY; // Stores the best objective value that has been seen.  Used to return best iterate, rather than last iterate

    double gx0;
    g(x0, group, &p, K, lambda, &gx0); // g(x0)
    objective[0] = f1 + gx0;           //  objective[1] <- f1 + g(x0)


    //====================
    double *gradf0 = (double *)calloc(p, sizeof(double));
    double *x1hat = (double *)calloc(p, sizeof(double));
    double *Dx = (double *)calloc(p, sizeof(double));
    double *dd = (double *)calloc(p, sizeof(double));
    double *Dg = (double *)calloc(p, sizeof(double));
    //====================
    // Begin Loop
    int i;
    for (i = 0; i < *max_iters; i++)
    {
        // Rename iterates relative to loop index.  "0" denotes index i, and "1" denotes index i+1
       //=== double *gradf0 = (double *)calloc(p, sizeof(double));
        for (int j = 0; j < p; j++)
        {
            x0[j] = x1[j];         // x_i <--- x_{i+1}
            gradf0[j] = gradf1[j]; //gradf0 <- matrix(gradf1);   gradf0 is now $\nabla f(x_i)$
        }
        double tau0 = *tau1; // tau0 <- tau1 ,   \tau_i <--- \tau_{i+1}

        // FBS step: obtain x_{i+1} from x_i
       //=== double *x1hat = (double *)calloc(p, sizeof(double));
        for (int j = 0; j < p; j++)
        {
            x1hat[j] = x0[j] - tau0 * gradf0[j]; //x1hat <- x0 - tau0 * c(gradf0),  Define \hat x_{i+1}
        }

        proxg(x1hat, group, &p, K, lambda, &tau0, x1); //x1 <- proxg(x1hat, tau0),  Define x_{i+1}

        // Non-monotone backtracking line search
       //=== double *Dx = (double *)calloc(p, sizeof(double));
        for (int j = 0; j < p; j++)
        {
            Dx[j] = x1[j] - x0[j]; // Dx <- matrix(x1 - x0)
            d1[j] = x1[j];         // d1 <- x1
        }
        f(d1, y, A, Z, var_hat, param, &f1); // f1  <- f(d1)

        if (*backtrack)
        {
            //M <- max( fVals[max(i-w,1):max(i-1,1)] )  # Get largest of last 10 values of 'f'
            int ind1 = max(i - *w, 0);
            int ind2 = max(i - 1, 0);
            double M = fVals[ind1];                       //give an initial value for M and then get the max
            for (int j = (ind1 + 1); j < (ind2 + 1); j++) //ind2  is included
            {
                M = max(M, fVals[j]);
            }

            backtrackCount = 0; //  backtrackCount <- 0
            //prop <- (f1 - 1e-12 > M + t(Dx)%*%gradf0 + 0.5*(norm(Dx,'f')**2)/tau0) && (backtrackCount < 20)
            double a1, a2, a3;
            innerprod(Dx, gradf0, &p, &a1); // a1= t(Dx)%*%gradf0
            vec_norm(Dx, &p, &a2);          // a2= norm(Dx,'f')
            a3 = f1 - 1e-12 - M - a1 - 0.5 * a2 * a2 / tau0;

            while ((a3 > 0) && (backtrackCount < 20)) // The backtrack loop
            {
                tau0 = tau0 * (*stepsizeShrink); // shrink stepsize
                for (int j = 0; j < p; j++)
                {
                    x1hat[j] = x0[j] - tau0 * gradf0[j]; //x1hat <- x0 - tau0*c(gradf0)  # redo the FBS
                }
                proxg(x1hat, group, &p, K, lambda, &tau0, x1); //x1 <- proxg(x1hat, tau0)
                for (int j = 0; j < p; j++)
                {
                    d1[j] = x1[j];         // d1 <- x1
                    Dx[j] = x1[j] - x0[j]; // Dx <- matrix(x1 - x0)
                }
                f(d1, y, A, Z, var_hat, param, &f1); // f1  <- f(d1)

                // update the condition for the next loop
                backtrackCount = backtrackCount + 1;             // update backtrackCount, backtrackCount <- backtrackCount + 1
                innerprod(Dx, gradf0, &p, &a1);                  // update a1, a1= t(Dx)%*%gradf0
                vec_norm(Dx, &p, &a2);                           // update a2, a2= norm(Dx,'f')
                a3 = f1 - 1e-12 - M - a1 - 0.5 * a2 * a2 / tau0; // update a3
            }                                                    // end while

            *totalBacktracks = *totalBacktracks + backtrackCount;
        } // end if(*backtrack)

        // Recorded information
        taus[i] = tau0; //  taus[i] <- tau0  # stepsize
        double Dx_norm;
        vec_norm(Dx, &p, &Dx_norm);
        residual[i] = Dx_norm / tau0;                //residual[i]<- norm(Dx,'f')/tau0,  # Estimate of the gradient, should be zero at solution
        maxResidual = max(maxResidual, residual[i]); // maxResidual <- max(maxResidual, residual[i])

        //compute normalizer
        double n1, n2;
        vec_norm(gradf0, &p, &n1); // n1=norm(gradf0,'f')
        //=== double *dd = (double *)calloc(p, sizeof(double));
        for (int j = 0; j < p; j++)
        {
            dd[j] = x1[j] - x1hat[j]; // dd=x1-x1hat
        }
        vec_norm(dd, &p, &n2);                           // n2 = norm(as.matrix(x1 - x1hat),'f')
        double normalizer = max(n1, n2 / tau0) + *eps_n; // normalizer=max(norm(gradf0,'f'),norm(as.matrix(x1- x1hat),'f')/tau0)+eps_n
        normalizedResid[i] = residual[i] / normalizer;   // Normalized residual:  size of discrepancy between the two derivative terms, divided by the size of the terms
        fVals[i] = f1;

        // record function values
        double gx1;
        g(x1, group, &p, K, lambda, &gx1);           // g(x1)
        objective[i + 1] = f1 + gx1;                 // objective[i+1] <- f1 + g(x1)
        double newObjectiveValue = objective[i + 1]; // newObjectiveValue <- objective[i+1]

        if (*recordIterates) //  record iterate values
        {                    // get the (i+1)-th col of iterates and assign
            for (int j = 0; j < p; j++)
            {
                iterates.data[(i + 1) * p + j] = x1[j]; //iterates[,i+1] <- x1
            }
        } // end if(recordIterates)

        //Methods is non-monotone:  Make sure to record best solution
        if (newObjectiveValue < minObjectiveValue)
        {
            //double  *bestObjectiveIterate = (double *) calloc (p, sizeof(double)); --------*******is an input********---------
            for (int j = 0; j < p; j++)
            {
                bestObjectiveIterate[j] = x1[j]; // bestObjectiveIterate <- x1
            }
            minObjectiveValue = min(minObjectiveValue, newObjectiveValue); //minObjectiveValue <- min(minObjectiveValue, newObjectiveValue)
        }                                                                  // end if(newObjectiveValue <)

        // Compute stepsize needed for next iteration using BB/spectral method
        gradf(d1, y, A, Z, var_hat, param, gradf1); // gradf1 <- gradf(d1)
        //=== double *Dg = (double *)calloc(p, sizeof(double));
        for (int j = 0; j < p; j++)
        {
            Dg[j] = gradf1[j] + (x1hat[j] - x0[j]) / tau0; // Dg <- matrix(gradf1 + (x1hat - x0)/tau0), #Delta_g, note that Delta_x was recorded above during backtracking
        }
        double dotprod;
        innerprod(Dx, Dg, &p, &dotprod); // dotprod <- t(Dx)%*%Dg

        double Dg_norm;
        vec_norm(Dx, &p, &Dx_norm);
        vec_norm(Dg, &p, &Dg_norm);

        double tau_s, tau_m;
        tau_s = Dx_norm * Dx_norm / dotprod;   // tau_s <- norm(Dx,'f')^2 / dotprod,  #  First BB stepsize rule
        tau_m = dotprod / (Dg_norm * Dg_norm); // tau_m <- dotprod / norm(Dg,'f')^2,  #  Alternate BB stepsize rule
        tau_m = max(tau_m, 0);                 // tau_m <- max(tau_m,0)

        if (fabs(dotprod) < 1e-15)
            break;

        if (2 * tau_m > tau_s) //  Use "Adaptive"  combination of tau_s and tau_m
        {
            *tau1 = tau_m;
        }
        else
        {
            *tau1 = tau_s - 0.5 * tau_m; // Experiment with this param
        }

        if ((*tau1 <= 0) || isinf(*tau1) || isnan(*tau1)) //?
        {
            *tau1 = tau0 * 1.5;
        }

        //==== free(gradf0);
        //====free(x1hat);
        //====free(Dx);
        //====free(dd);
        //====free(Dg);

    } // end for loop

    // After for loop, we only assign the NumTotalIters, and output the whole objective, residual,...
    // No need to change the column of iterates
    // selecting the results will be done in R
    //*TotalIters = i; // when stop, i is actually max_iters, so no need to +1
    if (i == *max_iters)
    {
        *TotalIters = i;
    }
    else
    {

        *TotalIters = i + 1; // if break, need +1;
    }

    free(normalizedResid);
    free(x1);
    free(d1);
    free(gradf1);
    //========================
    free(gradf0);
    free(x1hat);
    free(Dx);
    free(dd);
    free(Dg);
    //========================
}

void test_fasta(double *x0, double *y, double *A_data, double *Z_data, int *n, int *p,
                double *var_hat, double *param, int *group, double *K, double *lambda,
                double *tau1, int *max_iters, int *w, int *backtrack, int *recordIterates,
                double *stepsizeShrink, double *eps_n, double *bestObjectiveIterate,
                double *objective, double *fVals, int *totalBacktracks, double *residual, double *taus,
                double *iterates_data, int *TotalIters)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = A_data;

    matrix Z;
    Z.Ncol = *p;
    Z.Nrow = *p;
    Z.data = Z_data;

    matrix iterates;
    iterates.Nrow = *p;
    iterates.Ncol = *max_iters + 1;
    iterates.data = iterates_data;

    fasta(F1, GradF1, G1, ProxG1, x0, y, A, Z, var_hat, param, group, K, lambda, tau1,
          max_iters, w, backtrack, recordIterates, stepsizeShrink, eps_n,
          bestObjectiveIterate, objective, fVals, totalBacktracks, residual,
          taus, iterates, TotalIters);
}

void test_fasta2(double *x0, double *y, double *A_data, double *Z_data, int *n, int *p,
                 double *var_hat, double *param, int *group, double *K, double *lambda,
                 double *tau1, int *max_iters, int *w, int *backtrack, int *recordIterates,
                 double *stepsizeShrink, double *eps_n, double *bestObjectiveIterate,
                 double *objective, double *fVals, int *totalBacktracks, double *residual, double *taus,
                 double *iterates_data, int *TotalIters)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = A_data;

    matrix Z;
    Z.Ncol = *p;
    Z.Nrow = *p;
    Z.data = Z_data;

    matrix iterates;
    iterates.Nrow = *p;
    iterates.Ncol = *max_iters + 1;
    iterates.data = iterates_data;

    fasta(F2, GradF2, G1, ProxG1, x0, y, A, Z, var_hat, param, group, K, lambda, tau1,
          max_iters, w, backtrack, recordIterates, stepsizeShrink, eps_n,
          bestObjectiveIterate, objective, fVals, totalBacktracks, residual,
          taus, iterates, TotalIters);
}









void PDHG(double *x, double *v, double *y, matrix A, matrix Z,  int *group, double *K, double *lambda,
          int *maxiters, double *tol, int *adaptive, int *backtrack, double *tau, double *sigma,
          double *L, double *a, double *eta, double *Delta, double *gamma, double *b, int *stopRule, 
          double *estimate_of_x, double *estimate_of_v, int *Num_of_iters, double *Presidual, double *Dresidual,
          double *tau_seq, double *sigma_seq, int *updates)
{


    int p = A.Ncol;
    //======================= Initialize some values =====================
    *updates = 0;
    double *Zx = (double *)calloc(p, sizeof(double));
    mvprod(Z, x, Zx); // Zx= Z %*% x
    double *Ztv = (double *)calloc(p, sizeof(double));
    mtvprod(Z, v, Ztv); // Ztv = t(Z) %*% v

    double maxPrimal = -INFINITY;
    double maxDual = -INFINITY;

  

    //=========prepare some variables for FASTA in the loop
    double tau1 = 5;
    int max_iters_fasta = 100;
    int w = 10;
    int backtrack_fasta = 1;
    int recordIterates = 0;
    double stepsizeShrink = 0.5;
    double eps_n = 1e-15;
    int toltalBacktracks = 0;
    double *objective = (double *)calloc(max_iters_fasta + 1, sizeof(double));
    double *fVals = (double *)calloc(max_iters_fasta, sizeof(double));
    double *residual = (double *)calloc(max_iters_fasta, sizeof(double));
    double *taus = (double *)calloc(max_iters_fasta, sizeof(double));
    matrix iterates;
    iterates.Nrow = p;
    iterates.Ncol = max_iters_fasta + 1;
    iterates.data = NULL;
    int TotalIters = 0;



    // ============ Some itermediate values used in the iterations
    double *x0 = (double *)calloc(p, sizeof(double));
    double *v0 = (double *)calloc(p, sizeof(double));
    double *Zx0 = (double *)calloc(p, sizeof(double));
    double *Ztv0 = (double *)calloc(p, sizeof(double));

    double *xhat = (double *)calloc(p, sizeof(double));
    double *vhat = (double *)calloc(p, sizeof(double));
    double *x0_fasta = (double *)calloc(p, sizeof(double));
    double *v0_fasta = (double *)calloc(p, sizeof(double));
    double *Zxh = (double *)calloc(p, sizeof(double));

    double *dx = (double *)calloc(p, sizeof(double));
    double *dv = (double *)calloc(p, sizeof(double));
    double *r1 = (double *)calloc(p, sizeof(double));
    double *r2 = (double *)calloc(p, sizeof(double));
    double *d1 = (double *)calloc(p, sizeof(double));
    double *d2 = (double *)calloc(p, sizeof(double));
    double *r_diff = (double *)calloc(p, sizeof(double));
    double *d_diff = (double *)calloc(p, sizeof(double));
    
    double primal;  // the primal residual in each iteration
    double dual;  // the dual residual in each iteration
    int stop; // the stop sign in each iteration


    double prodx, prodx0, Zxv;
    double dxnorm, dvnorm, Hnorm;
    double decay;



     //======================= Begin iteration =====================
    int iter; // for saving Num_of_iters
    for (iter = 0; iter < *maxiters; iter++)
    {  
        // store old iterates
          for (int j = 0; j < p; j++)
        {
           x0[j] = x[j]; //  x0 = x
           v0[j] = v[j]; //  v0 = v
        }
        mvprod(Z, x, Zx0); // Zx0 = Z %*% x
        mtvprod(Z, v, Ztv0); // Ztv0 = t(Z) %*% v
        

        //======================= primal update =====================
        for (int j = 0; j < p; j++)
        {
            xhat[j] = x[j] - *tau * Ztv[j]; //xhat = x - tau*Ztv
            x0_fasta[j] = xhat[j];      // set the initial value of x in fasta as xhat
        }
        // x = res1$x
        fasta(F1, GradF1, G1, ProxG1, x0_fasta, y, A, Z, xhat, tau, group, K, lambda, &tau1,
              &max_iters_fasta, &w, &backtrack_fasta, &recordIterates, &stepsizeShrink, &eps_n,
              x, objective, fVals, &toltalBacktracks, residual, taus, iterates, &TotalIters);

        mvprod(Z, x, Zx); // Zx = Z %*% x
        for (int j = 0; j < p; j++)
        {   
            Zxh[j] = 2 * Zx[j] - Zx0[j]; //Zxh = 2*Zx - Zx0
        }



        //======================= dual update =====================
        for (int j = 0; j < p; j++)
        {
            vhat[j] = v[j] + *sigma * Zxh[j]; //vhat = v + sigma*Zxh
            v0_fasta[j] = vhat[j];      // set the initial value of v in fasta as vhat
        }
        // v = res2$x
        fasta(F2, GradF2, G1, ProxG1, v0_fasta, y, A, Z, vhat, sigma, group, K, lambda, &tau1,
              &max_iters_fasta, &w, &backtrack_fasta, &recordIterates, &stepsizeShrink, &eps_n,
              v, objective, fVals, &toltalBacktracks, residual, taus, iterates, &TotalIters);

        mtvprod(Z, v, Ztv); // Ztv = t(Z) %*% v
       
        
       //======================= compute and store residuals =====================
        for (int j = 0; j < p; j++)
        { //printf("%f \n", v[j]);
            dx[j] = x[j] - x0[j]; //dx = x - x0
            dv[j] = v[j] - v0[j]; //dv = v - v0
            r1[j] = dx[j]/(*tau) + Ztv0[j];  //r1 = dx/tau + Ztv0
            r2[j] = Ztv[j];   // r2 = Ztv
            d1[j] = dv[j]/(*sigma) + Zxh[j];  //r2 = dv/sigma + Zxh ////////////////////////////////////
            d2[j] = Zx[j];   // d2 = Zx

            r_diff[j] = r1[j] - r2[j];
            d_diff[j] = d1[j] - d2[j]; ////////////////////////////////////
        }
        
        vec_norm(r_diff, &p, &primal);  // primal = norm(r1-r2, "2")
        vec_norm(d_diff, &p, &dual);  // dual = norm(d1-d2, "2")
        
        maxPrimal = max(primal, maxPrimal);  // maxPrimal = max(primal, maxPrimal)
        maxDual = max(dual, maxDual);   // maxDual = max(dual, maxDual)
        Presidual[iter] = primal;   // outs$p[iter] = primal
        Dresidual[iter] = dual;   //  outs$d[iter] =  dual
        tau_seq[iter] = *tau;     //  outs$tau[iter] = tau
        sigma_seq[iter] = *sigma_seq;   // outs$sigma[iter] = sigma

      
        //======================= test stopping conditions =====================
        if(*stopRule == 1){
            stop = primal/maxPrimal < *tol && dual/maxDual < *tol;  
        }
        if(*stopRule == 2){
            stop = primal < *tol && dual < *tol;   
        }
        if(*stopRule == 3){
            stop = iter > *maxiters; 
        }
        // if stopping rule is satisfied
        if((stop && iter >= 5) || iter >= *maxiters-1){
           for (int j = 0; j < p; j++)
           {
               estimate_of_x[j] = x[j];   // outs$x = x
               estimate_of_v[j] = v[j];   // outs$v = v
           }

            *Num_of_iters = iter+1;   // outs$iters = iter, iter  stars from 0, check all iters!!!!!!
        }


        //======================= test the backtracking/stability condition =====================
        innerprod(Zx, dv, &p, &prodx);  // prodx = t(Zx) %*% dv
        innerprod(Zx0, dv, &p, &prodx0);  //  prodx0 = t(Zx0) %*% dv
        Zxv = 2*(prodx  - prodx0);  // Zxv = 2*t(Zx - Zx0)%*%dv = 2*(prodx - prodx0)

        vec_norm(dx, &p, &dxnorm); // dxnorm = norm(dx, "2")
        vec_norm(dv, &p, &dvnorm); // dvnorm = norm(dv, "2")
        Hnorm = dxnorm*dxnorm/(*tau) +  dvnorm*dvnorm/(*sigma); //  Hnorm = norm(dx, "2")^2/tau + norm(dv, "2")^2/sigma
        
        if(backtrack && (*gamma)*Hnorm<Zxv ){
           
            for (int j = 0; j < p; j++)
            {
               x[j] = x0[j]; //  x = x0
               v[j] = v0[j]; //  v = v0
               Zx[j] = Zx0[j];  // Zx = Zx0
               Ztv[j] = Ztv0[j];  // Ztv = Ztv0
            }

            decay = (*b)*(*gamma)*Hnorm/Zxv;
            *tau = (*tau) * decay;
            *sigma = (*sigma) * decay;
            *L = (*L) * decay * decay;

        }

        //======================= perform adaptive update =====================
        if(adaptive && iter >0 && max(primal, dual) < max(Presidual[iter-1], Dresidual[iter-1])){

            if(primal > (*Delta) * dual){
                *tau = *tau/(1- *a);
                *sigma = *L/ *tau;
                *a = (*a) * (*eta);
                *updates = *updates +1;
            }

            if(primal <  dual/(*Delta)){
                *tau = (*tau) *(1- *a);
                *sigma = *L/ *tau;
                *a = (*a) * (*eta);
                *updates = *updates +1;
            }

        }// end adaptive


    } // end for loop

    // Free Memory.......
    free(Zx);
    free(Ztv);

    free(objective);
    free(fVals);
    free(residual);
    free(taus);
    

    free(x0);
    free(v0);
    free(Zx0);
    free(Ztv0);

    free(xhat);
    free(vhat);
    free(x0_fasta);
    free(v0_fasta);
    free(Zxh);

    free(dx);
    free(dv);
    free(r1);
    free(r2);
    free(d1);
    free(d2);
    free(r_diff);
    free(d_diff);
}









void test_PDHG(double *x, double *v, double *y, double *A_data, double *Z_data, int *n, int *p, 
               int *group, double *K, double *lambda, int *maxiters, double *tol, int *adaptive, int *backtrack,
               double *tau, double *sigma, double *L, double *a,  double *eta, double *Delta, double *gamma, double *b,
               int *stopRule,  double *estimate_of_x, double *estimate_of_v,  int *Num_of_iters, double *Presidual, double *Dresidual,
               double *tau_seq, double *sigma_seq, int *updates)
{

    matrix A;
    A.Nrow = *n;
    A.Ncol = *p;
    A.data = A_data;

    matrix Z;
    Z.Ncol = *p;
    Z.Nrow = *p;
    Z.data = Z_data;

    PDHG(x, v, y, A, Z, group, K, lambda, maxiters, tol, adaptive, backtrack, tau, sigma, L, a, eta, Delta, gamma, b, stopRule,
         estimate_of_x, estimate_of_v, Num_of_iters, Presidual, Dresidual, tau_seq, sigma_seq, updates);
}



