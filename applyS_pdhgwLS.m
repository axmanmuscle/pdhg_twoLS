function [sx, xOut, zOut, tau, theta, xNew] = applyS_pdhgwLS(xIn, zIn, proxf, proxgconj, taukm1, thetakm1, alpha, A, B)
% this applyS will do PDHG iterations with Malitsky's line search

%%% line search params
beta = 0.8;
delta = 0.99;
mu = 0.8;

n = size(xIn, 1);
m = size(zIn, 1);

xLast = resize(xIn, [n+m 1]) - taukm1 * [A'; B'] * zIn;

xk = proxf(xIn - taukm1*A'*zIn, taukm1);

% tauk = taukm1 * sqrt(1 + thetakm1);
tauk = taukm1;

accept = false;
while ~accept
    thetak = tauk/taukm1;
    xbar_k = xk + thetak*(xk - xIn);
    zkp1 = proxgconj(zIn + beta*tauk*A*xbar_k, beta*tauk);
    
    left_term = sqrt(beta)*tauk*norm(A'*zkp1 - A'*zIn);
    right_term = delta*norm(zkp1 - zIn);

    if left_term <= right_term
        accept = true;
    else
        tauk = tauk * mu;
    end
end

xOut = (1-2*alpha)*xIn + 2*alpha*xbar_k;
zOut = (1-2*alpha)*zIn + 2*alpha*zkp1;

xNew = resize(xOut, [n+m 1]) - taukm1 * [A'; B'] * zOut;
sx = (1/alpha) * xNew + (1 - 1/alpha) * xLast;

tau = tauk;
theta = thetak;

end


