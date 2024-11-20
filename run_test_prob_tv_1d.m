rng(20241024)
num_segments = 5;
len_segments = 400;

C = 0.9*rand([len_segments 1]) + randi([-5 5]);
for i = 1:num_segments-1
    A = 0.9*rand([len_segments 1]) + randi([-2 5]);
    C = [C; A];
end

x = 1:num_segments*len_segments;
% figure; plot(x, C); hold on;
n = numel(C);

A = @(in) circshift(in, -1) - in;

Amat = zeros(n);
for i = 1:n
    ei = zeros([n 1]);
    ei(i) = 1;
    Amat(:, i) = A(ei);
end

theta = 0.9/norm(Amat)^2;
Bt = chol((1/theta)*eye(n) - Amat*Amat');
B = Bt';

clear Bt;

f = @(in) 0.5*norm(in - C, 2)^2;
proxf = @(x, t) proxL2Sq(x, t, C);

lambdas = [0.01, 0.1, 1, 2, 5, 10, 100, 1000, 1.0e4, 1.0e5];

taus = 10.^(-6:0.5:4);
betas = 10.^(-1:0.5:1);
maxIter = 5000;

obj_vals = zeros([numel(lambdas) numel(taus) numel(betas)]);
obj_vals(:, :, :, maxIter) = 0;

final_vals = zeros([numel(lambdas) numel(taus) numel(betas)]);
final_vals(:, :, :, n) = 0;

for lambda_idx = 1:numel(lambdas)
    lambda = lambdas(lambda_idx);

    g = @(in) lambda * norm(in, 1);
    ga = @(in) lambda * norm(A(in), 1);
    proxgconj = @(in, t) proxConjL1(in, t, lambda);

    z0 = zeros(size(C));

    for tau_idx = 1:numel(taus)
        tau = taus(tau_idx)

        for beta_idx = 1:numel(betas)
            beta = betas(beta_idx)

            [xStar, objVals, alphas] = gPDHG_wls(z0, proxf, proxgconj, ...
                f, g, Amat, B, 'maxIter', maxIter, 'tau0', tau, 'verbose', true);
    
            obj_vals(lambda_idx, tau_idx, beta_idx, :) = objVals;
            final_vals(lambda_idx, tau_idx, beta_idx, :) = xStar;
        end
    end
        % figure; plot(xStar_new);
end
save tv_1d.mat