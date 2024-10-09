function [xOut, tau, thetaOut] = applyS(xIn, pf, pgstar, tauk, thetak, theta, At, AB)

pgtilde = @(x, t, tauk) x - tauk * AB( pgstar(t*AB(x), t), 'transp');

% pgtildestar = @(x, t) x - pgtilde(x, t);

Rf = @(phi, t) 2*pf(phi, t) - phi;

mu = 0.8;
delta = 0.99;
beta = 0.8;



xhat = Rf(xIn, tauk);

tau_hat = rand;
tau_range = tauk * (sqrt(1 + thetak) - 1);
taukp1 = tauk + tau_hat*tau_range;

accept = false;
while ~accept
% just let yk = 0
    thetaOut = taukp1 / tauk;
    ykp1 = pgtilde(xhat, beta*taukp1, tauk);
    % ykp1 = pgtilde(xhat, beta*taukp1);

    sig = theta / taukp1;
    ytest = sig*AB(ykp1);

    left_term = sqrt(beta)*taukp1 * norm(At(ytest));
    right_term = delta * norm(ytest);

    % left_term = sqrt(beta)*taukp1 * norm(A'*ytest - A'*yk);
    % right_term = delta * norm(ytest - yk);

    if left_term <= right_term
        accept = true;
    else
        taukp1 = mu*taukp1;
    end
end

tau = taukp1;
xOut = 2*(xhat - ykp1) - xhat;
xOut = -1 * xOut;

end


