function run_test_prob_tv_all()
%%% let's a total variation denoising problem
rng(20241125)
vers = version('-release');
vnum = vers(end-1);
im = imread('cameraman.tif');
im = double(im) ./ 255;

im = imresize(im, 0.3);

noise_sig = 1e-1;
noise = noise_sig*randn(size(im));
noised_im = im + noise;

n = size(noised_im(:), 1);

%%% salt and pepper noise
num_snp = round(0.1*n); % 10 % of samples
indices = randperm(n);
indices = indices(1:num_snp);
snp = round(rand([num_snp 1]));
noised_im(indices) = snp;

% show noised image
figure; imshowscale(noised_im, 5);

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

sizex = size(computeGradient(noised_im));

%% params
% lambdas = linspace(0.001, 5, 100);
lambdas = 2.^(-2:1:2);
gammas = 10.^(-4:0.5:4);
maxIter = 800;

num_lambdas = numel(lambdas);
num_gammas = numel(gammas);
%save tv2d_rundata_1122.mat;

aoi_objvals = zeros([num_lambdas num_gammas maxIter]);
aoi_finalvals = zeros([num_lambdas num_gammas n]);

pdhg_objvals = zeros([num_lambdas num_gammas maxIter]);
pdhg_finalvals = zeros([num_lambdas num_gammas n]);

gpdhg_objvals = zeros([num_lambdas num_gammas maxIter]);
gpdhg_finalvals = zeros([num_lambdas num_gammas n]);

pdhgwls_objvals = zeros([num_lambdas num_gammas maxIter+1]);
pdhgwls_finalvals = zeros([num_lambdas num_gammas n]);


parfor lambda_idx = 1:num_lambdas
    x0 = zeros([n + m, 1]);
    lambda = lambdas(lambda_idx);
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
    proxgconj = @(x, t) proxConjL2L1(x, t, lambda);

    proxf_flat = @(x, t) proxL2Sq(x, t, noised_im(:));

    proxftilde = @(x, t) [proxf_flat(x(1:n), t); zeros(size(x(n+1:end)))];
    

    proxgtilde = @(x, t) x - t*[A';B']*proxgconj((theta/t)*(A*x(1:n) + B*x(n+1:end)), theta/t);
    proxgtildeconj = @(x, t) x - proxgtilde(x, t);
    Rftilde = @(x, t) 2*proxftilde(x,t) - x;
    Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
    Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;
    
    best_idx = 0;
    best_obj = Inf;
   
    for gamma_idx = 1:num_gammas
        z0 = zeros([n 1]);
        disp(gamma_idx);
        gamma = gammas(gamma_idx);
        tau = gamma/normA;

        objtilde = @(x) ftilde(proxftilde(x, tau)) + gtilde(proxftilde(x, tau));

        [xStar_gpdhg, objVals_gpdhg] = gPDHG_wls(z0, proxf_flat, proxgconj, fflat, g, A, ...
            B, 'maxIter', maxIter, 'tau0', tau, 'verbose', true);

        [xStar_pdhg, objVals_pdhg] = pdhg(z0, proxf_flat, proxgconj, tau, 'f', fflat, ...
            'g', g, 'A', @computeGradient, 'normA', normA, 'N', maxIter, 'verbose', true, 'tol', 1e-50);

        [xStar_pdhgWLS, objVals_pdhgWLS] = pdhgWLS(z0, proxf_flat, proxgconj, 'beta', 1, ...
            'tau', tau, 'f', fflat, 'g', g, 'A', @computeGradient, 'N', maxIter, 'verbose', false);

        S_pdDR = @(in) -tau * Rgtildeconj( Rftilde( in, tau) / tau, 1/tau);

        [xStar_aoi,objVals_aoi] = avgOpIter_wLS( x0(:), S_pdDR, 'N', maxIter, ...
                'objFunction', objtilde, 'verbose', false, 'printEvery', 20, 'doLineSearchTest', true );

        xStar_aoi_final = proxftilde(xStar_aoi, tau);

        aoi_objvals(lambda_idx, gamma_idx, :) = objVals_aoi;
        pdhg_objvals(lambda_idx, gamma_idx, :) = objVals_pdhg;
        gpdhg_objvals(lambda_idx, gamma_idx, :) = objVals_gpdhg;
        pdhgwls_objvals(lambda_idx, gamma_idx, :) = objVals_pdhgWLS;

        aoi_finalvals(lambda_idx, gamma_idx, :) = xStar_aoi_final(1:n);
        pdhg_finalvals(lambda_idx, gamma_idx, :) = xStar_pdhg;
        gpdhg_finalvals(lambda_idx, gamma_idx, :) = xStar_gpdhg;
        pdhgwls_finalvals(lambda_idx, gamma_idx, :) = xStar_pdhgWLS; 
    end

end
clear A B;
save tv2d_all_1202.mat
end

function out = deltay(x)
    if any(~x, 'all')
        out = 0;
    else
        out = Inf;
    end
end
