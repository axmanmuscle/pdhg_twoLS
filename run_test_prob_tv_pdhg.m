function run_test_prob_tv_pdhg()
%%% let's a total variation denoising problem
rng(20241125)

im = imread('cameraman.tif');
im = double(im) ./ 255;
im = imresize(im, 0.3);

noise = 0.08*randn(size(im));

noised_im = im + noise;

figure; imshowscale(noised_im);

dirStr = "E:\matlab\tvRunData\pdhg";

% lambdas = linspace(0.001, 5, 100);
% lambdas = 2.^(-6:6);
lambdas = [0.1];
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
    gamma_vals = 10.^(-4:0.5:4);
    parfor gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        gamma = gamma_vals(gamma_idx);
        [xStar_pdhg, objVals_pdhg] = pdhg(x0, proxf, proxgconj, gamma, 'N', 5000, ...
            'A', @computeGradient, 'f', f, 'g', g, 'normA', normA);
    
        % xend = proxf(xStar_pdhg, gamma);

        fstr_pdhg = sprintf('%s/lambda_%d_gamma_%d', dirStr, lambda_idx, gamma_idx);
        objStr_pdhg = sprintf('%s_obj.mat', fstr_pdhg);
        xStr_pdhg = sprintf('%s_x.mat', fstr_pdhg);

        parsave(objStr_pdhg, objVals_pdhg)
        parsave(xStr_pdhg, xStar_pdhg);

    end
    
    % figure; imshowscale(xBest); title(lstr)
    % 
    % figure; plot(bestObjs); title(lstr);
    % disp(best_idx);
    % disp(gamma_vals(best_idx));






end