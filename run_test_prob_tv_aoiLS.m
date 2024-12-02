function run_test_prob_tv_aoiLS()
%%% let's a total variation denoising problem
rng(20241125)
im = imread('cameraman.tif');
im = double(im) ./ 255;
im = imresize(im, 0.3);

noise = 0.08*randn(size(im));

noised_im = im + noise;
n = size(noised_im(:), 1);

figure; imshowscale(noised_im);

%dirStr = "E:\matlab\tvRunData\aoiLs";
dirStr = '/home/alex/Documents/MATLAB/tvRunData_new1121/aoiLs/'
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

clear G Bt;


% lambdas = linspace(0.001, 5, 100);
lambdas = [0.1];
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
    
    gamma_vals = 10.^(-4:0.5:4);
    parfor gamma_idx = 1:numel(gamma_vals)
        disp(gamma_idx);
        %%% define S_pdDR here
        gamma = gamma_vals(gamma_idx);
        S_pdDR = @(in) -gamma * Rgtildeconj( Rftilde( in, gamma ) / gamma , 1/gamma );
        
        [xStar,objVals_pdhgaoi,alphas] = avgOpIter_wLS( x0(:), S_pdDR, 'N', 200, ...
        'objFunction', objtilde, 'verbose', true, 'printEvery', 1, 'doLineSearchTest', true );
    
        xend = proxf_flat(xStar(1:n), gamma);

        fstr_aoi = sprintf('%s_gamma_%d', dirStr, gamma_idx);
        objStr_aoi = sprintf('%s_obj.mat', fstr_aoi);
        xStr_aoi = sprintf('%s_x.mat', fstr_aoi);

        parsave(objStr_aoi, objVals_pdhgaoi);
        parsave(xStr_aoi, xend);
    end
    

end
end

function out = deltay(x)
    if any(~x, 'all')
        out = 0;
    else
        out = Inf;
    end
end