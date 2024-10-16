function [xOut, zOut] = applyS_pdhg(xIn, zIn, proxf, proxgconj, gamma, sigma, alpha, A, B)
% this applyS is going to resemble PDHG iterations but will give me back Sx
% as if this were the PDDR/AOI S.
% A, At need to be function handles ig

xbar_pdhg = proxf(xIn - gamma*A'*zIn, gamma);
zbar_pdhg = proxgconj(zIn + sigma*A*(2*xbar_pdhg - xIn), sigma);


xOut = (1-2*alpha)*xIn + 2*alpha*xbar_pdhg;
zOut = (1-2*alpha)*zIn + 2*alpha*zbar_pdhg;

% zOut = gamma *[A';B']*z_pdhg;
% 
% xLast = xIn - gamma*A'*zIn;
% xLast = resize(xLast, size(zOut));
% 
% xOut = (1-2*alpha)*xLast + 2*alpha*(resize(x_pdhg, size(zOut)) - zOut);
end


