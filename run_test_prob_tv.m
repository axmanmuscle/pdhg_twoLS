function run_test_prob_tv()
%%% let's a total variation denoising problem
im = imread('cameraman.tif');
im = double(im) ./ 256;

noise = 0.5*randn(size(im));

noised_im = im + noise;

figure; imshowscale(noised_im);

x0 = zeros(size(noised_im));
lambda = 3;
f = @(x) norm(x - noised_im, 2)^2;
g = @(x) lambda * tvNorm(computeGradient(x));
obj = @(x) f(x) + g(x);

proxf = @(x, t) proxL2Sq(x, t, noised_im, eye(size(x, 1)));
proxg = @(x, t) tvProx(x, t, lambda);

gamma_vals = linspace(0.01, 8, 200);
best_idx = 0;
best_obj = Inf;
for gamma_idx = 1:numel(gamma_vals)
    disp(gamma_idx);
    gamma = gamma_vals(gamma_idx);
    [xStar, drObjVals] = douglasRachford(x0, proxf, proxg, gamma, 'N', 100, 'f', f, 'g', g);

    xend = proxf(xStar, gamma);
    final_obj = obj(xend);
    if final_obj < best_obj
        best_idx = gamma_idx;
        xBest = xend;
        bestObjs = drObjVals;
    end
end

figure; imshowscale(xBest);

figure; plot(bestObjs);
disp(best_idx);
disp(gamma_vals(best_idx));






end