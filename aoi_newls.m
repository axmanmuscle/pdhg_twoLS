function [xStar, iters] = aoi_newls(x0,proxf,proxgconj, theta, A, B)

maxIter = 100;
iters = zeros([maxIter size(x0)]);

tau0 = 1;
theta0 = 1;

xk = x0;
tauk = tau0;
thetak = theta0;
for i = 1:maxIter
  [xOut, tau, thetaOut] = applyS(xk, proxf, proxgconj, tauk, thetak, theta, A, B);

  xkp1 = 0.5*xk + 0.5*xOut;
  xk = xkp1;

  iters(i, :) = proxf(xk, 1);

  tauk = tau;
  thetak = thetaOut;

end

xStar = proxf(xk);
end