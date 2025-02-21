function run_test_prob_tv_all()
%%% let's a total variation denoising problem
rng(20241125)
vers = version('-release');
vnum = vers(end-1);
im = imread('cameraman.tif');
% im = imread('barbara.png');

minim = min(double(im), [], 'all');

scaled_im = double(im) - minim;
mim = max(scaled_im, [], 'all');
im =  scaled_im ./ mim;

im = imresize(im, 0.3);

noise_sig = 0.05;
noise = noise_sig*randn(size(im));
noised_im = im + noise;

n = size(noised_im(:), 1);

%%% salt and pepper noise
% num_snp = round(0.1*n); % 10 % of samples
% indices = randperm(n);
% indices = indices(1:num_snp);
% snp = round(rand([num_snp 1]));
% 
% noised_im(indices) = snp;

% show noised image
figure; imshowscale(im, 8); title('original image');
figure; imshowscale(noised_im, 8); title('noised image')

% info for saving variables
% dirStr = 'E:\matlab\tvRunData';
% dirStr = '/home/alex/Documents/MATLAB/tvRunData_new1121';

% compute stuff for PDDR line searches

normA = powerIteration(@computeGradient, noised_im);
A = makeTVMx(size(noised_im));
ta = 1.1 * normA^2;
theta = 1/ta;
G = A*A';
Bt = chol((1/theta)*eye(size(G)) - G);
B = Bt';
m = size(B, 1);

clear G Bt;
A = sparse(A);
B = sparse(B);
% clear B;

sizex = size(computeGradient(noised_im));

%% params
% lambdas = linspace(0.001, 5, 100);
% lambdas = 2.^(1:1:8);
% lambdas = [0.01 0.05 0.1 0.15 0.175 0.2]
% lambdas = [0.01 0.02 0.03 0.05 0.2];
lambda = 0.03;
gammas = 10.^(-4:0.5:4);
%gammas = [10^-0.5];
% gammas = [10.^4];
maxIter = 1000;

num_gammas = numel(gammas);
%save tv2d_rundata_1122.mat;

aoi_objvals = zeros([num_gammas maxIter]);
aoi_finalvals = zeros([ num_gammas n]);

pdhg_objvals = zeros([ num_gammas maxIter]);
pdhg_finalvals = zeros([ num_gammas n]);

gpdhg_objvals = zeros([ num_gammas maxIter]);
gpdhg_finalvals = zeros([ num_gammas n]);

pdhgwls_objvals = zeros([ num_gammas maxIter+1]);
pdhgwls_finalvals = zeros([ num_gammas n]);

gpdhg2_objvals = zeros([ num_gammas maxIter]);
gpdhg2_finalvals = zeros([ num_gammas n]);

x0 = zeros([n + m, 1]);
lstr = sprintf('lambda %f', lambda);
disp(lstr);

f = @(x) 0.5*norm(x - noised_im, 'fro')^2;
g = @(x) lambda*tvNorm(reshape(x, sizex));
ga = @(x) lambda * tvNorm(computeGradient(x));
obj = @(x) f(x) + g(x);
obja = @(x) f(x) + ga(x);

fflat = @(x) 0.5*norm(x - noised_im(:))^2;

objaf = @(x) fflat(x) + ga(x);

delta_y = @(x) deltay(x);
ftilde = @(x) fflat(x(1:n)) + delta_y(x(n+1: end));
gtilde = @(x) g( reshape(A*x(1:n) + B*(x(n+1:end)), sizex ) );

proxf = @(x, t) proxL2Sq(x, t, noised_im);
proxgconj = @(x, t) reshape(proxConjL2L1(reshape(x, sizex), t, lambda), [], 1);

proxf_flat = @(x, t) proxL2Sq(x, t, noised_im(:));

proxftilde = @(x, t) [proxf_flat(x(1:n), t); zeros(size(x(n+1:end)))];


proxgtilde = @(x, t) x - t*[A';B']*proxgconj((theta/t)*(A*x(1:n) + B*x(n+1:end)), theta/t);
proxgtildeconj = @(x, t) x - proxgtilde(x, t);
Rftilde = @(x, t) 2*proxftilde(x,t) - x;
Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;

best_idx = 0;
best_obj = Inf;

parfor gamma_idx = 1:num_gammas
    z0 = zeros([n 1]);
    disp(gamma_idx);
    gamma = gammas(gamma_idx);
    tau = gamma/normA;
    objtilde = @(x) ftilde(proxftilde(x, tau)) + gtilde(proxftilde(x, tau));

    [xStar_gpdhg, objVals_gpdhg] = gPDHG_wls(z0, proxf_flat, proxgconj, fflat, g, A, ...
        B, 'maxIter', maxIter, 'tau0', tau, 'verbose', true);

    [xStar_gpdhg2, objVals_gpdhg2] = test_gpdhg(z0, proxf_flat, proxgconj, fflat, g, A, ...
        B, 'maxIter', maxIter, 'tau0', tau, 'verbose', true);

    [xStar_pdhg, objVals_pdhg] = pdhg(z0, proxf_flat, proxgconj, tau, 'f', fflat, ...
        'g', g, 'A', A, 'normA', normA, 'N', maxIter, 'verbose', true, 'tol', 1e-32);

    [xStar_pdhgWLS, objVals_pdhgWLS] = pdhgWLS(z0, proxf_flat, proxgconj, 'beta', 1, ...
        'tau', tau, 'f', fflat, 'g', g, 'A', A, 'N', maxIter, 'verbose', false);

    S_pdDR = @(in) -tau * Rgtildeconj( Rftilde( in, tau) / tau, 1/tau);

    [xStar_aoi,objVals_aoi] = avgOpIter_wLS( x0(:), S_pdDR, 'N', maxIter, ...
            'objFunction', objtilde, 'verbose', false, 'printEvery', 20, 'doLineSearchTest', true );

    xStar_aoi_final = proxftilde(xStar_aoi, tau);

    aoi_objvals(gamma_idx, :) = objVals_aoi;
    pdhg_objvals(gamma_idx, :) = objVals_pdhg;
    gpdhg_objvals(gamma_idx, :) = objVals_gpdhg;
    pdhgwls_objvals(gamma_idx, :) = objVals_pdhgWLS;
    gpdhg2_objvals(gamma_idx, :) = objVals_gpdhg2;

    aoi_finalvals(gamma_idx, :) = xStar_aoi_final(1:n);
    pdhg_finalvals(gamma_idx, :) = xStar_pdhg;
    gpdhg_finalvals(gamma_idx, :) = xStar_gpdhg;
    pdhgwls_finalvals(gamma_idx, :) = xStar_pdhgWLS;
    gpdhg2_finalvals(gamma_idx, :) = xStar_gpdhg2;
end


clear A B;
save 1213_tv2d_fixed.mat aoi_objvals pdhg_objvals gpdhg_objvals pdhgwls_objvals gpdhg2_objvals aoi_finalvals pdhg_finalvals gpdhg_finalvals pdhgwls_finalvals gpdhg2_finalvals

end

function out = deltay(x)
    if any(~x, 'all')
        out = 0;
    else
        out = Inf;
    end
end

%%
% for i = 1:numel(lambdas)
% f1 = squeeze(pdhg_finalvals(i, :, :));
% figure; imshowscale(reshape(f1, size(im)), 5); title(sprintf('lambda %f', lambdas(i)));
% end
