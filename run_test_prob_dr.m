function run_test_prob_dr()
%%%
%%% need to be able to run DR, AOI line search, Malitsky line search, our
%%% line search
%%% with multiple parameters WHY DO WE STILL HAVE TO TUNE TAU
%%% Convergence to solution for each method
%%% plots
%%% write this tonight to let it run
%%% make a function like this that takes in tau maybe?
%%% need to save out obj value vs. iteration count for every method
%%% in primal dual dr aoi wls this is "gamma"
%%% omfg
m = 9;
n = 3;
b = 8*ones([n, 1]);
eps = 0.2;

%% problem is 
% min f(x) + g(Ax)
% f is an indicator function for an l2 ball around b with radius eps
% g is the 1 norm
A = randi(15, [m n]);
ta = 1.01 * norm(A)^2;
theta = 1/ta;
tau = 0.1;
G = A*A';
Bt = chol((1/theta)*eye(size(G)) - G);
B = Bt';

sigma = theta/tau;

g = @(x) norm(x, 1);
ga = @(x) norm(A*x, 1);
f = @(x) 0;
obj = @(x) f(x) + ga(x);
proxf = @(x, t) proxF(x, b, eps);
proxg = @(x, t) proxL1Complex(x, t);

gtilde = @(x) norm(A*x(1:n) + B*x(n+1:end), 1);
ftilde = @(x) full_f(x(1:n), b, eps);
objmod = @(x) ftilde(x) + gtilde(x);
pf = @(x, t) [proxF(x(1:n), b, eps); zeros([m 1])];
pgstar = @(x, t) proxConjL1(x, t, 1);
% pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(sigma*(A*x(1:n) + B*x(n+1:end)), sigma);
pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(t*(A*x(1:n) + B*x(n+1:end)), t);
pgtildestar = @(x, t) x - pgtilde(x, t);

pg3 = @(x, t) tau * [A'; B'] * pgstar(sigma*(A*x(1:n) + B*x(n+1:end)), sigma);

% x0 = [-20;-10; -30; 0; 0; 0];
% x0 = [-20;-10;-30;zeros([m 1])];
x0 = [50*randn([n 1]); zeros([m 1])];
x0_dr = 50*randn([n 1]);

[xStar_mod, drModObjVals] = douglasRachford(x0, pf, pgtilde, 1, 'N', 500, 'f', ftilde, 'g', gtilde);
[xStar, drObjVals] = douglasRachford(x0_dr, proxf, proxg, 1, 'N', 500, 'f', f, 'g', ga);
[xStar_pddr, objVals_pddr] = primal_dual_dr_test(x0,pf,pgtildestar,1,objmod);
[xStar2, driters] = primal_dual_dr(x0, pf, pgtildestar);
[xStar3, dr2iters] = primal_dual_dr_2(x0, pf, pgtildestar);
[xStar4, aoiiters] = primal_dual_dr_aoi(x0, pf, pgtildestar);
[xStar6, aoiiters2, alphas] = primal_dual_dr_aoi_wls(x0, pf, pg3, f, g, 300);
% [xStar7, aoiiters] = primal_dual_dr_aoi_newls(x0, pf, pgtildestar);

% [xStar8, newiters, taus] = pddr_malitsky_test(A);

[xStar9, iters] = aoi_newls(x0,pf,pgstar, theta, A, B, n, m);
% [xStar10, ls_iters, alphas, objVals] = primal_dual_dr_aoi_newls(x0,pf,pgstar, ftilde, gtilde, 100, theta, A, B, 1);

[xStar11, objVals_new, alphas_new] = gPDHG_wls(x0,pf,pgstar, ftilde, gtilde, 100, theta, A, B, 1, n, m);

pfnew = @(x, t) proxF(x(1:n), b, eps);
% [xStarpdhg, objvals, reldiffs] = pdhg(x0(1:3), pfnew, pgstar, tau, 'A', A, ...
%   'f', f, 'g', g, 'normA', norm(A), 'verbose', true, 'tol', 1e-10);
% [xStar5, dr3iters] = primal_dual_dr_3(x0(1:n), pfnew, pgstar, A, B, theta);
% 
% [xStarpdhgwls, objvalswls] = pdhgWLS(x0(1:3), pfnew, pgstar, 'tau', tau, 'A', A, ...
%   'f', f, 'g', g, 'verbose', true);


xStar

end

function out = full_f(x, b, eps)
  if norm(x - b, 2) <= eps
    out = 0;
  else
    out = Inf;
  end
end

function out = proxF(x, b, eps)
  if norm(x - b, 2) < eps
    out = x;
  else
    xperp = (x - b)/norm(x-b,2) * eps;
    out = b + xperp;
  end
end