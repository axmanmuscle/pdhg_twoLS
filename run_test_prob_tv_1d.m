rng(20241025)
num_segments = 5;

len_segments = 400;

gt = zeros([num_segments*len_segments 1]);

ri = randi([-5 5], [num_segments 1]);
C = 0.25*randn([len_segments 1]) + ri(1);
gt(1:len_segments) = ri(1);
for i = 1:num_segments-1
    A = 0.5*randn([len_segments 1]) + ri(i+1);
    C = [C; A];
    gt(i*len_segments+1 : (i+1)*len_segments) = ri(i+1);
end

x = 1:num_segments*len_segments;
% figure; plot(x, C, 'DisplayName', 'Noised'); hold on;
% plot(x, gt, 'DisplayName', 'gt', 'LineWidth', 2)
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
m = size(B, 1);

clear Bt;

f = @(in) 0.5*norm(in - C, 2)^2;
proxf = @(x, t) proxL2Sq(x, t, C);

Rftilde = @(in, t) [proxf(in(1:n), t); zeros([m 1])];

%lambdas = [0.01, 0.1, 1, 2, 5, 10, 100, 1000];
lambdas = [10];

taus = 10.^(-5:0.5:3);
%betas = 10.^(-1:0.5:1);
% taus = [10^-.5];
betas = [1];
maxIter = 1000;

obj_vals = zeros([numel(taus) ]);
obj_vals( :, maxIter) = 0;

final_vals = zeros([numel(taus) ]);
final_vals(:, n) = 0;

num_taus = numel(taus);
%num_betas = numel(betas);

for lambda_idx = 1:numel(lambdas)
    lambda = lambdas(lambda_idx);

    g = @(in) lambda * norm(in, 1);
    ga = @(in) lambda * norm(A(in), 1);
    proxgconj = @(in, t) proxConjL1(in, t, lambda);

    gtilde = @(in) lambda*g(A(in(1:n)) + B*in(n+1:end));
    objtilde = @(in) f(in(1:n)) + g(in(1:n));

    proxgtilde = @(x, t) x - t*[Amat';B']*proxgconj((theta/t)*(Amat*x(1:n) + B*x(n+1:end)), theta/t);
    proxgtildeconj = @(x, t) x - proxgtilde(x, t);
    Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
    Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;

    z0 = zeros(size(C));

    for tau_idx = 1:num_taus
        tau = taus(tau_idx);
        disp(tau_idx)

        x0 = zeros([n+m 1]);
        % 
        % [xStar, objVals, alphas] = gPDHG_wls(z0, proxf, proxgconj, ...
        %     f, g, Amat, B, 'maxIter', maxIter, 'tau0', tau,'beta0', 1, 'verbose', true);
        % [xStar, objVals, alphas] = pdhg(z0, proxf, proxgconj, tau,...
        %     'f', f, 'g', g, 'A', Amat, 'normA', 2, 'N', maxIter, 'verbose', false, 'tol', 1e-15);

        S_pdDR = @(in) -tau * Rgtildeconj( Rftilde( in, tau ) / tau , 1/tau );

        [xStar,objVals,alphas_aoi] = avgOpIter_wLS( x0(:), S_pdDR, 'N', maxIter, ...
            'objFunction', objtilde, 'verbose', true, 'printEvery', 20, 'doLineSearchTest', true );

        % [xStar, objVals] = pdhgWLS(z0, proxf, proxgconj, 'beta', beta, 'tau', tau,...
        %     'A', Amat, 'f', f, 'g', g, 'N', maxIter, 'verbose', true);

        obj_vals(tau_idx, :) = objVals;
        final_vals(tau_idx, :) = xStar(1:n);
    end
        % figure; plot(xStar_new);
end
save tv_1d_aoi_lambda10.mat