function [xStar, iters, alphas, objVals] = primal_dual_dr_aoi_newls(x0,proxf,proxgconj, f, g, maxIter, theta, A, B, gamma)

%%% parameters
alpha_bar = 0.5; % alpha_bar
% gamma = 3; % gamma for prox operators
eps = 0.03; % eps for (1 - eps) || rbar_k || in linesearch
tol = 1e-7; % tolerance for exit criterion
alpha_change = 1/1.4; % factor for change in alpha during linesearch

Rf = @(phi) 2*proxf(phi, gamma) - phi;
% Rgconj = @(phi) 2*proxgconj(phi, gamma) - phi;
Rg2 = @(phi) 2*proxgconj(phi, 1/gamma) - phi;

% S = @(phi) Rgconj(Rf(phi));
S_old = @(phi) -gamma * Rg2( (1/gamma) * Rf(phi) );

S = @(phi, tauk, thetak) applyS(phi, proxf, proxgconj, tauk, thetak, theta, A, B);
S2 = @(phi, tauk, thetak, yk) applyS2(phi, proxf, proxgconj, yk, tauk, thetak, theta, A, B);

iters = zeros([maxIter size(x0)]);
alphas = zeros(maxIter, 1);
objVals = zeros([maxIter, 1]);

tau0 = 1;
theta0 = 1;

xk = x0;
tauk = tau0;
thetak = theta0;
yk = zeros(size(A, 1), 1);

for i = 1:maxIter
  [sxk, taukp1, thetakp1] = S(xk, tauk, thetak);
  [sxk2, taukp12, thetakp12, ykp12] = S2(xk, tauk, thetak, yk);
  rk = sxk - xk;
  % rk = S(xk) - xk;
  old_sxk = S_old(xk);
  old_rk = old_sxk - xk;
  xk_bar = xk + alpha_bar * rk;
  [sxk_bar, resid_tau, resid_theta] = S(xk_bar, taukp1, thetakp1);
  rk_bar = sxk_bar - xk_bar;
  old_rkbar = S_old(xk_bar) - xk_bar;
  % rk_bar = S(xk_bar) - xk_bar;

  alpha_k = 50;
  subiter = 0;
  sub_tau = taukp1;
  sub_theta = thetakp1;
  while true
    subiter = subiter + 1;
    fprintf('subiter %d\n', subiter);
    xkp1 = xk + alpha_k * rk;
    [sxkp1, sub_tau, sub_theta] = S(xkp1, sub_tau, sub_theta);
    rkp1 = sxkp1 - xkp1;
    % rkp1 = S(xkp1) - xkp1;
    if norm(rkp1) < (1-eps) * norm(rk_bar)
      xk = xkp1;
      alphas(i) = alpha_k;
      tauk = taukp1;
      thetak = thetakp1;
      yk=ykp12;
      break
    end
    alpha_k = alpha_k*alpha_change;
    if alpha_k < alpha_bar
      alpha_k = alpha_bar;
      xkp1 = xk + alpha_k*rk;
      xk = xkp1;
      alphas(i) = alpha_k;
      tauk = taukp1;
      thetak = thetakp1;
      yk = ykp12;
      break
    end
  end

  iters(i, :) = proxf(xk, gamma);

  objVals(i) = f(proxf(xk, gamma)) + g(proxf(xk, gamma));

  % if mod(i, 3) == 1
    fprintf('iter: %d   objVal: %d   res: %d  alpha: %d\n', i, objVals(i), norm(rkp1(:)), alpha_k );
  %end

end
xk = proxf(xk, gamma);
xStar = xk;
end