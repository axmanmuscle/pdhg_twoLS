rng(2024);
m = 2500;
n = 500;

A = randn([m n]);
x = randn([n 1]);

b = A*x;

%%% minimize 0.5*|| Ax - b ||^2 st |xi| > 0 for all i

f = @(in) 0.5*norm(A*in - b)^2;
g = @(in) indicatorFunction(in, [0 Inf]);

proxf = @(in, t) proxL2Sq(in, t, b, A);
proxg = @(in, t) max(in, 0);
proxgconj = @(in, t) in - t * proxg(in/t, 1/t);

x0 = zeros([n 1]);

theta = 0.9 / norm(A)^2;
Bt = chol((1/theta) * eye(m) - A*A');
B = Bt';

% tau0 = 1/norm(A);
% for i = -3:5
%     tau = tau0 * 10^i;
%     [xStar_pdhg, objVals_pdhg] = pdhg(x0, proxf, proxgconj, tau, 'A', A, 'f', f, 'g', g, 'normA', norm(A), 'N', 10000);
%     xStar_pdhg = max(xStar_pdhg, 0);
%     objVals(i+4) = f(xStar_pdhg);
% end

betas = 10.^(-6:6);
taus = 10.^(-6:6) / norm(A);
objVals = zeros([numel(betas) numel(taus)]);
for bidx = 1:numel(betas)
    beta = betas(bidx)
    tauObjValues = zeros([numel(taus) 1]);
    parfor tidx = 1:numel(taus)
        tau = taus(tidx)
        % [xStar_pdhgwls, objVals_pdhgwls] = pdhgWLS(x0, proxf, proxgconj, 'A', A, 'f', f, 'g', g, 'tau', tau, 'N', 1000, 'beta', beta);
        [xStar, objVals] = gPDHG_wls(x0, proxf, proxgconj, f, g, A, B, 'maxIter', 750, 'beta0', beta, 'tau0', tau);
        xStar = max(xStar, 0);
        tauObjValues(tidx) = f(xStar);
    end
    objVals(bidx, :) = tauObjValues;
end

tauObjValues = zeros([numel(taus) 1]);
for tidx = 1:numel(taus)
    tau = taus(tidx);
    % sigma = 0.99 / (tau * norm(A)^2);
    % beta = 0.99 / (tau*norm(A))^2;
    beta = 1;
    [xStar, objVals_new] = gPDHG_wls(x0, proxf, proxgconj, f, g, A, B, 'maxIter', 1000, 'beta0', beta, 'tau0', tau);
    xStar = max(xStar, 0);
    tauObjValues(tidx) = f(xStar);
end
% [xStar, objVals] = gPDHG_wls(x0, proxf, proxgconj, f, g, theta, A, B, 'maxIter', 1000);
% [xStar_pdhgwls, objVals_pdhgwls] = pdhgWLS(x0, proxf, proxgconj, 'A', A, 'f', f, 'g', g, 'verbose', true, 'N', 10000, 'beta', 100);