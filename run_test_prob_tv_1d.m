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
B = sparse(B);
Amat = sparse(Amat);

f = @(in) 0.5*norm(in - C, 2)^2;
proxf = @(x, t) proxL2Sq(x, t, C);

proxftilde = @(in, t) [proxf(in(1:n), t); zeros([m 1])];

Rftilde = @(in, t) 2*proxftilde(in, t) - in;


%lambdas = [0.01, 0.1, 1, 2, 5, 10, 100, 1000];
lambda = 10;

taus = 10.^(-5:0.5:3);
%betas = 10.^(-1:0.5:1);
% taus = [10^-.5];
% taus = [10.^3]
betas = [1];
maxIter = 1000;

num_taus = numel(taus);
%num_betas = numel(betas);
g = @(in) lambda * norm(in, 1);
ga = @(in) lambda * norm(A(in), 1);
proxgconj = @(in, t) proxConjL1(in, t, lambda);

gtilde = @(in) lambda*g(A(in(1:n)) + B*in(n+1:end));


proxgtilde = @(x, t) x - t*[Amat';B']*proxgconj((theta/t)*(Amat*x(1:n) + B*x(n+1:end)), theta/t);
proxgtildeconj = @(x, t) x - proxgtilde(x, t);
Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;

z0 = zeros(size(C));

%% optimal value
[xStar_optimal, objVals_optimal, alphas] = pdhg(z0, proxf, proxgconj, 10^-0.5,...
        'f', f, 'g', g, 'A', Amat, 'normA', 2, 'N', 75000, 'verbose', true, 'tol', 10^-32);

save 2_22_1d_optimal xStar_optimal objVals_optimal

% aoi_objvals = zeros([num_taus maxIter]);
% aoi_finalvals = zeros([ num_taus n]);
% 
% pdhg_objvals = zeros([ num_taus maxIter]);
% pdhg_finalvals = zeros([ num_taus n]);
% 
% gpdhg_objvals = zeros([ num_taus maxIter]);
% gpdhg_finalvals = zeros([ num_taus n]);
% 
% pdhgwls_objvals = zeros([ num_taus maxIter+1]);
% pdhgwls_finalvals = zeros([ num_taus n]);
% 
% gpdhg2_objvals = zeros([ num_taus maxIter]);
% gpdhg2_finalvals = zeros([ num_taus n]);
% 
% 
% parfor tau_idx = 1:num_taus
%     tau = taus(tau_idx);
%     disp(tau_idx)
%     objtilde = @(in) f(proxf(in(1:n), tau)) + gtilde(proxftilde(in, tau));
% 
%     x0 = zeros([n+m 1]);
% 
%     [xStar_gpdhg, objVals_gpdhg, alphas] = gPDHG_wls(z0, proxf, proxgconj, ...
%         f, g, Amat, B, 'maxIter', maxIter, 'tau0', tau,'beta0', 1, 'verbose', true);
% 
%     [xStar_gpdhg2, objVals_gpdhg2, alphas] = test_gpdhg(z0, proxf, proxgconj, ...
%         f, g, Amat, B, 'maxIter', maxIter, 'tau0', tau,'beta0', 1, 'verbose', true);
% 
%     [xStar_pdhg, objVals_pdhg, alphas] = pdhg(z0, proxf, proxgconj, tau,...
%         'f', f, 'g', g, 'A', Amat, 'normA', 2, 'N', maxIter, 'verbose', false, 'tol', 1e-15);
% 
%     S_pdDR = @(in) -tau * Rgtildeconj( Rftilde( in, tau ) / tau , 1/tau );
% 
%     [xStar_aoi,objVals_aoi,alphas_aoi] = avgOpIter_wLS( x0(:), S_pdDR, 'N', maxIter, ...
%         'objFunction', objtilde, 'verbose', true, 'printEvery', 20, 'doLineSearchTest', true );
% 
%     [xStar_pdhgWLS, objVals_pdhgWLS] = pdhgWLS(z0, proxf, proxgconj, 'beta', 1, 'tau', tau,...
%         'A', Amat, 'f', f, 'g', g, 'N', maxIter, 'verbose', true);
% 
%     xStar_aoi_final = proxftilde(xStar_aoi, tau);
% 
%     aoi_objvals(tau_idx, :) = objVals_aoi;
%     pdhg_objvals(tau_idx, :) = objVals_pdhg;
%     gpdhg_objvals(tau_idx, :) = objVals_gpdhg;
%     pdhgwls_objvals(tau_idx, :) = objVals_pdhgWLS;
%     gpdhg2_objvals(tau_idx, :) = objVals_gpdhg2;
% 
%     aoi_finalvals(tau_idx, :) = xStar_aoi_final(1:n);
%     pdhg_finalvals(tau_idx, :) = xStar_pdhg;
%     gpdhg_finalvals(tau_idx, :) = xStar_gpdhg;
%     pdhgwls_finalvals(tau_idx, :) = xStar_pdhgWLS;
%     gpdhg2_finalvals(tau_idx, :) = xStar_gpdhg2;
% end
%         % figure; plot(xStar_new);
% save 221_tv1d_new_gpdhg.mat