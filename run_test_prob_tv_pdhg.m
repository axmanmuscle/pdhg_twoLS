function run_test_prob_tv_pdhg()
%%% let's a total variation denoising problem
im = imread('cameraman.tif');
im = double(im) ./ 256;

noise = 0.08*randn(size(im));

noised_im = im + noise;

figure; imshowscale(noised_im);


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
    
    best_idx = 0;
    best_obj = Inf;
    normA = powerIteration(@computeGradient, noised_im);
    gamma_vals = 10.^(-8:4);
    for gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        gamma = gamma_vals(gamma_idx);
        [xStar_pdhg, objVals_pdhg] = pdhg(x0, proxf, proxgconj, gamma, 'N', 200, ...
            'A', @computeGradient, 'f', f, 'g', g, 'normA', normA);
    
        xend = proxf(xStar_pdhg, gamma);
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






end