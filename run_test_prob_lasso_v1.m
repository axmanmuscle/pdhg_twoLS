function run_test_prob_lasso_v1()
%%% This is another (!) test function for the new applyS functions we've
%%% written
%%%
%%% we'll solve the lasso problem 
%%% min 0.5*|| Ax - b ||_2^2 + lambda || x ||_1
%%%  x
%%% where A is R^(mxn)

rng(20241014);
n = 3;
m = 9;

noise_sig = 0.05;
lambda = 0.5;

A = randn(m, n);
x = randn(n, 1);
b = A*x + noise_sig * randn(m, 1);

f = @(in) 0.5 * norm(A*in - b, 2)^2;
g = @(in) lambda*norm(in, 1);

obj = @(in) f(in) + g(in);

proxf = @(in, t) proxL2Sq(in, t, b, A);
proxg = @(in, t) proxL1Complex(in, t*lambda);

proxfconj = @(in, t) in - t * proxf(in/t, 1/t);
proxgconj = @(in, t) in - t * proxg(in/t, 1/t);

%%% moreau test
xtest = randn(size(x));
gammatest = 0.25;
pftest = proxf(xtest, gammatest) + gammatest*proxfconj(xtest/gammatest, 1/gammatest);
pgtest = proxg(xtest, gammatest) + gammatest*proxgconj(xtest/gammatest, 1/gammatest);

x0 = zeros(size(x));

alpha = 0.5; % alpha for AOI steps
gamma = .25; % step size for prox operators

%% TO DO
% take a step of dr and pddr using the regular iterations and show these
% are the same
% and probably check that it converges to a solution idk
% then try it with option 2

%%% take a step for both DR and PDDR using AOI
x0_dr = x0;
x0_pddr = x0;

xaoi_dr = (1 - alpha)*x0_dr + alpha * applyS_dr(x0_dr, proxf, proxg, gamma);
xaoi_dr2 = (1 - alpha)*xaoi_dr + alpha * applyS_dr(xaoi_dr, proxf, proxg, gamma);

xaoi_pddr = (1 - alpha)*x0_pddr + alpha * applyS_pddr(x0_pddr, proxf, proxgconj, gamma);
xaoi_pddr2 = (1 - alpha)*xaoi_pddr + alpha * applyS_pddr(xaoi_pddr, proxf, proxgconj, gamma);

% here try using the resolvents
Jf = @(in, t) proxf(in, t);
Jg_inv = @(in, t) t * proxgconj(in/t, 1/t);

xaoi_pddr_resolve = (1 - alpha)*x0_pddr + alpha * applyS_pddr_resolve(x0_pddr, Jf, Jg_inv, gamma);
xaoi_pddr_resolve2 = (1 - alpha)*xaoi_pddr_resolve + alpha * applyS_pddr_resolve(xaoi_pddr_resolve, Jf, Jg_inv, gamma);

%%% take dr step using dr iters
x_dr = proxf(x0_dr, gamma);
y_dr = proxg(2*x_dr - x0_dr, gamma);
z_dr = x0_dr + 2*alpha*(y_dr - x_dr);

%%% one more dr step
x_dr2 = proxf(z_dr, gamma);
y_dr2 = proxg(2*x_dr2 - z_dr, gamma);
z_dr2 = z_dr + 2*alpha*(y_dr2 - x_dr2);

%%% pddr step using pddr iters
x_pddr = proxf(x0_pddr, gamma);
y_pddr = 2*x_pddr - x0_pddr - gamma * proxgconj((2*x_pddr - x0_pddr)/gamma, 1/gamma);
z_pddr = x0_pddr + 2*alpha*(y_pddr - x_pddr);

%%% second try at pddr
x_pddr2 = proxf(x0_pddr, gamma);
y_pddr2 = gamma * proxgconj((2*x_pddr - x0_pddr) / gamma, 1/gamma);
z_pddr2 = (1 - 2*alpha)*x0_pddr - 2*alpha*(y_pddr2 - x_pddr2);

%%% second step
x_pddr3 = proxf(z_pddr2, gamma);
y_pddr3 = gamma * proxgconj((2*x_pddr3 - z_pddr2) / gamma, 1/gamma);
z_pddr3 = (1 - 2*alpha)*z_pddr2 - 2*alpha*(y_pddr3 - x_pddr3);

%%% here xaoi_dr, xaoi_pddr, y_dr, and z_pddr should all be the same

%% Modified Problem
% now its min f(x) + g(Ax)
normA = norm(A);
theta = 0.9/normA^2;
G = A*A';
Bt = chol((1/theta)*eye(size(G)) - G);
B = Bt';

ftilde = @(x) f(x(1:n)) + 99999*norm(x(n+1:end));
gtilde = @(x) g( A*x(1:n) + B*x(n+1:end) );
obj2 = @(x) ftilde(x) + gtilde(x);

proxftilde = @(in, t) [proxf(in(1:n), t); zeros([m 1])];
proxgtilde = @(in, t) in - t * [A';B'] * proxgconj((theta/t) * [A B]*in, theta / t);
proxgtildeconj = @(in, t) in - proxgtilde(in, t);

%%% one step of PDHG is equiv to one step of PDDR applied to above
x0 = zeros([n+m 1]);

%%% lets do DR iters
x_dr = proxftilde(x0, gamma);
y_dr = proxgtilde(2*x_dr - x0, gamma);
z_dr = x0 + 2*alpha*(y_dr - x_dr);

%%% one more step
x_dr2 = proxftilde(z_dr, gamma);
y_dr2 = proxgtilde(2*x_dr2 - z_dr, gamma);
z_dr2 = z_dr + 2*alpha*(y_dr2 - x_dr2);

%%% pddr iters
x_pddr = proxftilde(x0, gamma);
y_pddr = proxgtildeconj(2*x_pddr - x0, gamma);
z_pddr = (1 - 2*alpha)*x0 + 2*alpha*(x_pddr - y_pddr);

%%% one more step to test
x_pddr2 = proxftilde(z_pddr, gamma);
y_pddr2 = proxgtildeconj(2*x_pddr2 - z_pddr, gamma);
z_pddr2 = (1 - 2*alpha)*z_pddr + 2*alpha*(x_pddr2 - y_pddr2);

%%% aoi iters
xaoi_dr = (1 - alpha)*x0 + alpha * applyS_dr(x0, proxftilde, proxgtilde, gamma);
xaoi_dr2 = (1 - alpha)*xaoi_dr + alpha * applyS_dr(xaoi_dr, proxftilde, proxgtilde, gamma);

xaoi_pddr = (1 - alpha)*x0 + alpha * applyS_pddr(x0, proxftilde, proxgtildeconj, gamma);
xaoi_pddr2 = (1 - alpha)*xaoi_pddr + alpha * applyS_pddr(xaoi_pddr, proxftilde, proxgtildeconj, gamma);

Jf = @(in, t) proxftilde(in, t);
Jg_inv = @(in, t) proxgtildeconj(in, t);

xaoi_pddr_resolve = (1 - alpha)*x0 + alpha * applyS_pddr_resolve(x0, Jf, Jg_inv, gamma);
xaoi_pddr2_resolve = (1 - alpha)*xaoi_pddr_resolve + alpha * applyS_pddr_resolve(xaoi_pddr_resolve, Jf, Jg_inv, gamma);

%%% pdhg iter
z0 = zeros([m 1]);
sigma = theta/gamma;
xbar_pdhg = proxf(x0_dr - gamma*A'*z0, gamma);
zbar_pdhg = proxgconj(z0 + sigma*A*(2*xbar_pdhg - x0_dr), sigma);
out_pdhg = [x0_dr; z0] + 2*alpha*[xbar_pdhg - x0_dr; zbar_pdhg - z0];
x_pdhg = out_pdhg(1:n);
z_pdhg = out_pdhg(n+1:end);

%%% second PDHG iter
xbar_pdhg2 = proxf(x_pdhg - gamma*A'*z_pdhg, gamma);
zbar_pdhg2 = proxgconj(z_pdhg + sigma*A*(2*xbar_pdhg2 - x_pdhg), gamma);
x_pdhg2 = (1-2*alpha)*x_pdhg + 2*alpha*xbar_pdhg2;
z_pdhg2 = (1-2*alpha)*z_pdhg + 2*alpha*zbar_pdhg2;

%%% second pdhg form
z0 = zeros([m 1]);
y_pdhg = proxgconj(z0 + sigma*A*x0_dr, sigma);
x_pdhg_new = proxf(x0_dr - gamma*A'*y_pdhg, gamma);
xbar_pdhg_new = x_pdhg_new + 1*(x_pdhg_new - x0_dr);

%%% idk what's going on anymore
[xStar] = pdhg(x0_dr, proxf, proxgconj, gamma, 'sigma', sigma, 'A', A, 'N', 3)

%%% after one step we should have
%%% resize(xaoi_pdhg, [n+m 1]) - gamma * [A'; B'] * zaoi_pdhg = z_pddr
[xaoi_pdhg, zaoi_pdhg] = applyS_pdhg(x0_dr, z0, proxf, proxgconj, gamma, sigma, alpha, A, B);
[xaoi_pdhg2, zaoi_pdhg2] = applyS_pdhg(xaoi_pdhg, zaoi_pdhg, proxf, proxgconj, gamma, sigma, alpha, A, B);

[sx, xnew, znew] = applyS_pdhg_full(x0_dr, z0, proxf, proxgconj, gamma, sigma, alpha, A, B);
zold = resize(x0_dr, [n+m 1]) - gamma * [A'; B'] * z0;
aoi_step1 = (1 - alpha)*zold + alpha * sx;
zold1 = resize(xnew, [n+m 1]) - gamma * [A'; B'] * znew;
[sx1, xnew1, znew1] = applyS_pdhg_full(xnew, znew, proxf, proxgconj, gamma, sigma, alpha, A, B);
aoi_step2 = (1 - alpha)*zold1 + alpha*sx1;

%%% this now works
%%% relationship b/w pdhg and pddr:
%%% x_pddr = x_pdhg
%%% sigma * [A B] * y_pddr = zbar_pdhg;
%%% or gamma * [A^T; B^T] * zbar_pdhg = y_pddr;

%%% pdhg wls aoi iteration
tau0 = 1;
theta0 = 1;
[sx2, xOut, zOut, tauk, thetak, xNew] = applyS_pdhgwLS(x0_dr, z0, proxf, proxgconj, gamma/0.8, 0, alpha, A, B);
xaoi_pdhg_new = (1 - alpha)*x0 + alpha*sx;

applyA = @(in) A*in;
At = @(in) A'*in;
Bt = @(in) B'*in;
[xOut_n, zOut_n, tau_n, nsxmx, xNewn] = applyS_pdhgwLS_op_new(x0_dr, z0, proxf, proxgconj, 1, 1, alpha, applyA, At, Bt);


% %%% eqs 36
% ubar_k = [proxf(x0_dr, gamma); zeros([m 1])];
% wbar_k = gamma * [A'; B'] * proxgconj(sigma*[A B]*(2*ubar_k - x0), sigma);
% u_k = (1 - 2*alpha)*zeros([n+m 1]) + 2*alpha*ubar_k;
% y_k = (1 - 2*alpha)*x0 + 2*alpha*(ubar_k - wbar_k);
% w_k = u_k - y_k;
% 
% %%% eqs 37
% 
% ubar_k2 = [proxf(x0_dr, gamma); zeros([m 1])];
% wbar_k2 = gamma * [A';B']*proxgconj(sigma*[A B]*zeros(size(x0)) + sigma*[A B] *(2*ubar_k2 - x0), sigma);
% u_k2 = (1 - 2*alpha)*zeros([n+m 1]) + 2*alpha*ubar_k2;
% w_k2 = (1 - 2*alpha)*x0 + 2*alpha*wbar_k2;






end