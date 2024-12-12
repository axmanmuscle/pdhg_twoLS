function run_test_prob_lasso_v2()
%%% Code for the paper
%%% create noisy signal b
%%% solve
%%%   min 1/2 * || x - b ||_2^2 + lambda*||Ax||_1
%%%    x
%%% where A is n x n and random

rng(20241014);
n = 1000;

A = randn(n);
normA = norm(A);
b = 8 * ones([n 1]);

theta = 0.9/normA^2;
Bt = chol((1/theta)*eye(n) - A*A');
B = Bt';

clear Bt;

f = @(in) 0.5 * norm(in - b, 2)^2;

proxf = @(in, t) proxL2Sq(in, t, b);
proxfconj = @(in, t) in - t * proxf(in/t, 1/t);

Rftilde = @(in, t) [proxf(in(1:n), t); zeros([n 1])];

z0 = zeros([n 1]);

% lambdas = [0.01 0.1 0.5 1 10];
taus = 2.^(-4:0.5:4);
lambdas = [3];

num_lambda = numel(lambdas);
num_tau = numel(taus);
maxIter = 2000;

objVals_gpdhg_all = zeros([num_lambda num_tau maxIter]);
objVals_pdhg_all = zeros([num_lambda num_tau maxIter]);
objVals_pdhgWls_all = zeros([num_lambda num_tau maxIter+1]);
objVals_aoi_all = zeros([num_lambda num_tau maxIter]);

x_gpdhg_all = zeros([num_lambda num_tau n]);
x_pdhg_all = zeros([num_lambda num_tau n]);
x_pdhgWls_all = zeros([num_lambda num_tau n]);
x_aoi_all = zeros([num_lambda num_tau n]);

for lambda_idx = 1:num_lambda
    lambda = lambdas(lambda_idx);
    disp(lambda_idx);

    proxg = @(in, t) proxL1Complex(in, t*lambda);
    g = @(in) lambda*norm(in, 1);
    ga = @(in) lambda*norm(A*in, 1);

    obj = @(in) f(in) + g(A*in);
    proxgconj = @(in, t) in - t * proxg(in/t, 1/t);

    objtilde = @(in) f(in(1:n)) + g(A*in(1:n));

    proxgtilde = @(x, t) x - t*[A';B']*proxgconj((theta/t)*(A*x(1:n) + B*x(n+1:end)), theta/t);
    proxgtildeconj = @(x, t) x - proxgtilde(x, t);
    Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
    Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;

    parfor tau_idx = 1:num_tau
        x0 = zeros([n+n 1]);
        tau = taus(tau_idx);

        [xStar_gpdhg, objVals_gpdhg] = gPDHG_wls(z0, proxf, proxgconj, f, g, A, ...
            B, 'maxIter', maxIter, 'tau0', tau, 'beta0', 1, 'verbose', true);

        [xStar_pdhg, objVals_pdhg] = pdhg(z0, proxf, proxgconj, tau, 'f', f, ...
            'g', g, 'A', A, 'normA', normA, 'N', maxIter, 'verbose', false, 'tol', 1e-15);

        [xStar_pdhgWLS, objVals_pdhgWLS] = pdhgWLS(z0, proxf, proxgconj, 'beta', 1, ...
            'tau', tau, 'f', f, 'g', g, 'A', A, 'N', maxIter, 'verbose', true);

        S_pdDR = @(in) -tau * Rgtildeconj( Rftilde( in, tau) / tau, 1/tau);

        [xStar_aoi,objVals_aoi] = avgOpIter_wLS( x0(:), S_pdDR, 'N', maxIter, ...
                'objFunction', objtilde, 'verbose', false, 'printEvery', 20, 'doLineSearchTest', true );

        objVals_gpdhg_all(lambda_idx, tau_idx, :) = objVals_gpdhg;
        objVals_pdhg_all(lambda_idx, tau_idx, :) = objVals_pdhg;
        objVals_pdhgWls_all(lambda_idx, tau_idx, :) = objVals_pdhgWLS;
        objVals_aoi_all(lambda_idx, tau_idx, :) = objVals_aoi;

        x_gpdhg_all(lambda_idx, tau_idx, :) = xStar_gpdhg;
        x_pdhg_all(lambda_idx, tau_idx, :) = xStar_pdhg;
        x_pdhgWls_all(lambda_idx, tau_idx, :) = xStar_pdhgWLS;
        x_aoi_all(lambda_idx, tau_idx, :) = xStar_aoi(1:n);

    end
end

save 1211_fixed_lasso.mat


end