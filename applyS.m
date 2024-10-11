function [xOut, tau, thetaOut, yOut] = applyS(xIn, pf, pgstar, yk, tauk, thetak, theta, At, AB)

pgtilde = @(x, t, tauk) x - tauk * AB(pgstar(t*AB(x), t), 'transp');
Rf = @(phi, t) 2*pf(phi, t) - phi;

% linesearch params

mu = 0.8;
delta = 0.99;
beta = 0.8;

xhat = Rf(xIn, tauk);

tau_hat = rand;
tau_range = tauk * (sqrt(1 + thetak) - 1);
taukp1 = tauk + tau_hat*tau_range;

accept = false;
while ~accept
    thetaOut = taukp1 / tauk;
    ykp1 = pgtilde(xhat, beta*taukp1, tauk);

    sig = theta / taukp1;
    yhat = sig*AB(ykp1);
    if size(yk) ~= size(yhat)
        yk = reshape(yk, size(yhat));
    end

    left_term = sqrt(beta)*taukp1 * norm(At(yhat) - At(yk));
    right_term = delta * norm(yhat - yk);

    if left_term <= right_term
        accept = true;
    else
        taukp1 = mu*taukp1;
    end
end

tau = taukp1;
xOut = 2*(xhat - ykp1) - xhat;
xOut = -1 * xOut;
yOut = yhat;

end


