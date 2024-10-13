function [xStar, iters] = aoi_newls(x0,proxf,proxgconj, theta, A, B, n, m)

if isnumeric(A) && isnumeric(B)
    applyA = @(x) A*x;
    applyAt = @(x) A'*x;
    applyB = @(x) B*x;
    applyBt = @(x) B'*x;
elseif isa(A, 'function_handle') && isa(B, 'function_handle')
    applyA = @(x) A(x, 'notransp');
    applyAt = @(x) A(x, 'transp');
    applyB = @(x) B(x, 'notransp');
    applyBt = @(x) B(x, 'transp');

else
    disp('both A, B must be numeric or functions');
    return
end



function out = applyAB( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyA( in(1:n) ) + applyB( in(n+1:end) );
    else
      out = zeros( n + m, 1 );
      out(1:n) = applyAt( in );
      out(n+1:end) = applyBt( in );
    end
end

maxIter = 100;
iters = zeros([maxIter size(x0)]);

tau0 = 1;
theta0 = 1;

xk = x0;
tauk = tau0;
thetak = theta0;

xkm1 = xk;
yk = zeros([m 1]);

for i = 1:maxIter
  [xOut, tau, thetaOut, yk, xkm1] = applyS(xk, proxf, proxgconj, yk, xkm1, tauk, thetak, theta, applyAt, @applyAB);

  xkp1 = 0.5*xk + 0.5*xOut;
  xk = xkp1;

  % iters(i, :) = proxf(xk, 1);
  iters(i, :) = xk;

  tauk = tau;
  thetak = thetaOut;

end

xStar = proxf(xk);
end