function run_test_prob_tv_newls()
%%% let's a total variation denoising problem
im = imread('cameraman.tif');
im = double(im) ./ 255;

im = imresize(im, 0.2);

noise = 0.08*randn(size(im));
noised_im = im + noise;

n = size(noised_im(:), 1);

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
m = size(B, 1);

sizex = size(computeGradient(noised_im));
% then write the functions for the line search


% lambdas = linspace(0.001, 5, 100);
lambdas = 2.^(-6:2);
for lambda_idx = 1:numel(lambdas)
    x0 = zeros([n + m, 1]);
    lambda = lambdas(lambda_idx);
    lstr = sprintf('lambda %f', lambda);
    disp(lstr);

    f = @(x) 0.5*norm(x - noised_im, 'fro')^2;
    g = @(x) lambda*tvNorm(x);
    ga = @(x) lambda * tvNorm(computeGradient(x));
    obj = @(x) f(x) + g(x);
    obja = @(x) f(x) + ga(x);

    fflat = @(x) 0.5*norm(x - noised_im(:))^2;

    objaf = @(x) fflat(x) + g(resize(x, sizex));

    delta_y = @(x) deltay(x);
    ftilde = @(x) fflat(x(1:n)) + delta_y(x(n+1: end));
    gtilde = @(x) g( resize(A*x(1:n) + B*(x(n+1:end)), sizex ) );
    

    % proxftilde = @(x, t)
    % proxgtilde
    
    proxf = @(x, t) proxL2Sq(x, t, noised_im);
    proxgconj = @(x, t) proxConjL2L1(x, t, lambda);

    proxf_flat = @(x, t) proxL2Sq(x, t, noised_im(:));

    proxftilde = @(x, t) [proxf_flat(x(1:n), t); zeros(size(x(n+1:end)))];
    objtilde = @(x) ftilde(proxftilde(x, 1)) + gtilde(proxftilde(x, 1));
    
    best_idx = 0;
    best_obj = Inf;
    
    gamma_vals = 10.^(-8:4);
    for gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        gamma = gamma_vals(gamma_idx);
        maxIter = 50;
        % [xStar, iters, alphas, objVals_newls] = primal_dual_dr_aoi_newls(x0, ...
        %     proxftilde,proxgconj, ftilde, gtilde, maxIter, theta, A, B, gamma);
        
        [xStar, objVals_newls, alphas] = gPDHG_wls(x0, proxftilde,proxgconj, ftilde, ...
                        gtilde, maxIter, theta, A, B, gamma);
        xend = proxf_flat(xStar(1:n), gamma);
        final_obj = objaf(xend);
        if final_obj < best_obj
            best_idx = gamma_idx;
            xBest = xend;
            bestObjs = objVals_newls;
            best_obj = final_obj;
        end
    end
    
    figure; imshowscale(resize(xBest, size(noised_im)),5); title(lstr)
    
    figure; plot(bestObjs); title(lstr);
    disp(best_idx);
    disp(gamma_vals(best_idx));

    pause(2);

end
end

function out = deltay(x)
    if any(~x, 'all')
        out = 0;
    else
        out = Inf;
    end
end