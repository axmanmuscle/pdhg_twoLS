function run_test_prob_tv_aoiLS()
%%% let's a total variation denoising problem
im = imread('cameraman.tif');
im = double(im) ./ 255;

im = imresize(im, 0.25);

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
lambdas = 2.^(-6:6);
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

    objaf = @(x) fflat(x) + ga(x);
    delta_y = @(x) deltay(x);
    ftilde = @(x) fflat(x(1:n)) + delta_y(x(n+1: end));
    gtilde = @(x) g( resize(A*x(1:n) + B*(x(n+1:end)), sizex ) );
    

    % proxftilde = @(x, t)
    % proxgtilde
    
    proxf = @(x, t) proxL2Sq(x, t, noised_im);
    proxgconj = @(x, t) proxConjL2L1(x, t, lambda);

    proxf_flat = @(x, t) proxL2Sq(x, t, noised_im(:));

    proxftilde = @(x, t) [proxf_flat(x(1:n), t); zeros(size(x(n+1:end)))];
    proxgtilde = @(x, t) x - t*[A';B']*proxgconj((theta/t)*(A*x(1:n) + B*x(n+1:end)), theta/t);
    proxgtildeconj = @(x, t) x - proxgtilde(x, t);

    objtilde = @(x) ftilde(proxftilde(x, 1)) + gtilde(proxftilde(x, 1));

    Rftilde = @(x, t) 2*resize(proxftilde(x,t), size(x)) - x;
    Rgtilde = @(x, t) 2*proxgtilde(x,t) - x;
    Rgtildeconj = @(x, t) 2*proxgtildeconj(x, t) - x;
    
    best_idx = 0;
    best_obj = Inf;
    
    gamma_vals = 10.^(-8:4);
    for gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        %%% define S_pdDR here
        gamma = gamma_vals(gamma_idx);
        S_pdDR = @(in) -gamma * Rgtildeconj( Rftilde( in, gamma ) / gamma , 1/gamma );
        
        [xStar,objVals_pdhgaoi,alphas] = avgOpIter_wLS( x0(:), S_pdDR, 'N', 300, ...
        'objFunction', objtilde, 'verbose', true, 'printEvery', 1, 'doLineSearchTest', true );
    
        xend = proxf_flat(xStar(1:n), gamma);
        final_obj = objaf(xend);

        fstr = sprintf('/home/alex/Documents/MATLAB/tvRunData/pdhgAOIls/lambda_%d_gamma_%d', lambda_idx, gamma_idx);
        objStr = sprintf('%s_obj.mat', fstr);
        xStr = sprintf('%s_x.mat', fstr);

        save(objStr, "objVals_pdhgaoi");
        save(xStr, "xend");
        if final_obj < best_obj
            best_idx = gamma_idx;
            xBest = xend;
            bestObjs = objVals_pdhgaoi;
            best_obj = final_obj;
        end
    end
    
    figure; imshowscale(resize(xBest, size(noised_im))); title(lstr)
    
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