function [sx, xOut, zOut, xNew, sxmx] = applyS_pdhg_full(xIn, zIn, proxf, proxgconj, gamma, sigma, alpha, A, B)
% this applyS is going to resemble PDHG iterations but will give me back Sx
% as if this were the PDDR/AOI S.
% A, At need to be function handles ig

n = size(xIn, 1);
m = size(zIn, 1);

xLast = resize(xIn, [n+m 1]) - gamma * [A'; B'] * zIn;

xbar_pdhg = proxf(xIn - gamma*A'*zIn, gamma);
zbar_pdhg = proxgconj(zIn + sigma*A*(2*xbar_pdhg - xIn), sigma);


xOut = (1-2*alpha)*xIn + 2*alpha*xbar_pdhg;
zOut = (1-2*alpha)*zIn + 2*alpha*zbar_pdhg;

xNew = resize(xOut, [n+m 1]) - gamma * [A'; B'] * zOut;
sx = (1/alpha) * xNew + (1 - 1/alpha) * xLast; % this isolates Sx, independent of alpha
sxmx = (1/alpha) * (xNew - xLast); % this isolates Sx - x, independent of alpha

end


