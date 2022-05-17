# GMC-Computation
A new computational method for the GMC-penalized least squares problem.

The R package grGMC includes all code for computing (group) GMC penalized least squares using the PDHG method. An additional method which transform the original problem into standard convex programmings (QP or QCQP) is also included. It performs better when computing a single solution but worse than PDHG when computing a solution path since there is no warm start for QP or QCQP in gurobi.
