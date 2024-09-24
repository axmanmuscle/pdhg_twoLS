# Introduction
My attempt at composing two different line searches for Chambolle and Pock's Primal Dual Hybrid Gradient (PDHG) method. The first uses the reuivalence between PDHG and the primal-dual formulation of Douglas-Rachford to write PDHG as an Averaged Operator Iteration, then uses Stephen Boyd's AOI linesearch. The second line search is by Malitsky and Pock and operates directly on the PDHG iterations.

Since the formulation is quite complicated I'll add some information about it here as well as link the final monograph when it is finished.
