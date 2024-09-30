function run_ls_tests()
%run_ls_tests LS parameter tests
%   We have three different line searches for the PDHG method to solve
%   problems of the form
%   min f(x) + g(Ax)
%    x
%   where f, g are proxable but not necessarily differentiable
%
%   for a first problem let's do f(x) = 0.5 * || . ||_2^2 and 
%   g(x) = || x ||_1

%   compare DR to PDHG wLS, our line search, and PDHG -> AOI line search?
%   want to compare params to convergence, etc.


f = @(x) 0.5 * norm(x, 2)^2;
g = @(x) norm(x, 1);

end