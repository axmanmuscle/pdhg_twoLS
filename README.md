# Introduction
My attempt at composing two different line searches for Chambolle and Pock's Primal Dual Hybrid Gradient (PDHG) method. The first uses the reuivalence between PDHG and the primal-dual formulation of Douglas-Rachford to write PDHG as an Averaged Operator Iteration, then uses Stephen Boyd's AOI linesearch. The second line search is by Malitsky and Pock and operates directly on the PDHG iterations.

Since the formulation is quite complicated I'll add some information about it here as well as link the final monograph when it is finished.

## To Do
We need to write the test for the line searches. That means running them with different parameters on the same problem. Set up a script that will run the various line searches with their different parameters and print out the objective value vs. iteration count for each.
