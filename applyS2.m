function [xOut, tau, thetaOut, yOut] = applyS2(xIn, pf, pgstar, yk, tauk, thetak, theta, A, B)
% b = [8;8;8];
% eps = 0.2;
n = size(A, 2);
%
% theta = 1/norm(A)^2 - 1e-6;
% tau = 0.1;
% Bt = chol((1/theta)*eye(9) - A*A');
% B = Bt';
% sigma = theta/tau;
%
% pf = @(x, t) [proxF(x(1:n), b, eps); zeros([9 1])];
% pgstar = @(x, t) proxConjL1(x, t, 1);
% % pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(sigma*(A*x(1:n) + B*x(n+1:end)), sigma);
% % pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(t*(A*x(1:n) + B*x(n+1:end)), t);
pgtilde = @(x, t, tauk) x - tauk * [A'; B'] * pgstar(t*(A*x(1:n) + B*x(n+1:end)), t);
%
% pgtildestar = @(x, t) x - pgtilde(x, t);


% x0 = [-20;-10;-30;zeros([9 1])];

% g = @(x) norm(x, 1);
% f = @(x) 0;

Rf = @(phi, t) 2*pf(phi, t) - phi;

% maxIter = 100;
% iters = zeros([maxIter size(x0)]);
% taus = zeros([maxIter 1]);

% linesearch params

mu = 0.8;
delta = 0.99;
beta = 0.8;
% theta0 = 1;
% tau0 = 1;
% y0 = zeros(size(A,1));

% xk = x0;
% yk = y0;
% tauk = tau0;
% thetak = theta0;


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
    yhat = sig*[A B] * ykp1;

    % left_term = sqrt(beta)*taukp1 * norm(A'*yhat);
    % right_term = delta * norm(yhat);

    left_term = sqrt(beta)*taukp1 * norm(A'*yhat - A'*yk);
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


