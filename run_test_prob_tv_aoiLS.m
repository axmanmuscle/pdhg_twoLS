function run_test_prob_tv_aoiLS()
%%% let's a total variation denoising problem
im = imread('cameraman.tif');
im = double(im) ./ 255;

im = imresize(im, 0.25);

noise = 0.08*randn(size(im));

noised_im = im + noise;

% show noised image
figure; imshowscale(noised_im, 5);

% compute stuff for PDDR line searches

normA = powerIteration(@computeGradient, noised_im);
A = makeTVMx(size(noised_im));
ta = 1.01 * normA^2;
theta = 1/ta;
tau = 0.1;
G = A*A';
Bt = chol((1/theta)*eye(size(G)) - G);
B = Bt';
% then write the functions for the line search


% lambdas = linspace(0.001, 5, 100);
lambdas = 2.^(-6:6);
for lambda_idx = 1:numel(lambdas)
    x0 = zeros(size(noised_im));
    lambda = lambdas(lambda_idx);
    lstr = sprintf('lambda %f', lambda);
    disp(lstr);
    f = @(x) 0.5*norm(x - noised_im, 'fro')^2;
    g = @(x) lambda*tvNorm(x);
    ga = @(x) lambda * tvNorm(computeGradient(x));
    obj = @(x) f(x) + g(x);
    obja = @(x) f(x) + ga(x);
    
    proxf = @(x, t) proxL2Sq(x, t, noised_im);
    proxgconj = @(x, t) proxConjL2L1(x, t, lambda);
    % proxgtildeconj = @(x, t) 
    
    best_idx = 0;
    best_obj = Inf;
    
    gamma_vals = 10.^(-8:4);
    for gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        %%% define S_pdDR here
        gamma = gamma_vals(gamma_idx);
        S_pdDR = @(x) -gamma * Rg_tildeConj( Rf( in, gamma ) / gamma , 1/gamma );
        
        [xStar,objValues,alphas] = primal_dual_dr_aoi_newls( x0, S_pdDR, 'N', 1000, ...
        'objFunction', objF, 'verbose', true, 'printEvery', 1, 'doLineSearchTest', true );
    
        xend = proxf(xStar, gamma);
        final_obj = obja(xend);
        if final_obj < best_obj
            best_idx = gamma_idx;
            xBest = xend;
            bestObjs = objVals_pdhg;
            best_obj = final_obj;
        end
    end
    
    figure; imshowscale(xBest); title(lstr)
    
    figure; plot(bestObjs); title(lstr);
    disp(best_idx);
    disp(gamma_vals(best_idx));

    pause(2);






end