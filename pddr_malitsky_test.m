function [xStar, iters, taus] = pddr_malitsky_test(A)
m = 300;
n = 75;
b = 8*ones([n, 1]);
eps = 0.2;

%% problem is 
% min f(x) + g(Ax)
%A = [2 0 1;1 2 3; 8 4 6];
ta = 1.01 * norm(A)^2;
theta = 1/ta;
% theta = 2;
tau = 0.1;
G = A*A';
Bt = chol((1/theta)*eye(size(G)) - G);
B = Bt';

sigma = theta/tau;

pf = @(x, t) [proxF(x(1:n), b, eps); zeros([9 1])];
pgstar = @(x, t) proxConjL1(x, t, 1);
% pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(sigma*(A*x(1:n) + B*x(n+1:end)), sigma);
% pgtilde = @(x, t) x - tau * [A'; B'] * pgstar(t*(A*x(1:n) + B*x(n+1:end)), t);
pgtilde = @(x, t, tauk) x - tauk * [A'; B'] * pgstar(t*(A*x(1:n) + B*x(n+1:end)), t);

pgtildestar = @(x, t) x - pgtilde(x, t);


x0 = [-20;-10;-30;zeros([9 1])];

g = @(x) norm(x, 1);
f = @(x) 0;

Rf = @(phi, t) 2*pf(phi, t) - phi;
Rgconj = @(phi, t) 2*pgtildestar(phi, t) - phi;

% Sstar = @(phi, t) t*Rgconj(Rf(phi/t, 1/t), 1/t);
Sstar = @(phi, t) Rgconj(Rf(phi, t), t);

maxIter = 100;
iters = zeros([maxIter size(x0)]);
taus = zeros([maxIter 1]);

% linesearch params

mu = 0.8;
delta = 0.99;
beta = 0.8;
theta0 = 1;
tau0 = 1;
y0 = zeros(size(A,1));

xk = x0;
yk = y0;
tauk = tau0;
thetak = theta0;


for i = 1:maxIter

  xhat = Rf(xk, tauk);
    
  tau_hat = rand;
  tau_range = tauk * (sqrt(1 + thetak) - 1);
  taukp1 = tauk + tau_hat*tau_range;

  accept = false;
  while ~accept 
      thetak = taukp1 / tauk;
      ykp1 = pgtilde(xhat, beta*taukp1, tauk);

      sig = theta / taukp1;
      ytest = sig*[A B] * ykp1;

      left_term = sqrt(beta)*taukp1 * norm(A'*ytest - A'*yk);
      right_term = delta * norm(ytest - yk);

      if left_term <= right_term
          accept = true;
      else
          taukp1 = mu*taukp1;
      end
  end

  taus(i) = taukp1;

  xnew = 2*(xhat - ykp1) - xhat;

  yk = ytest;

  xkp1 = 0.5*xk - 0.5*xnew;

  xk = xkp1;


  % xkp1 = 0.5*xk - 0.5*Sstar(xk, 1);
  % xk = pf(xkp1, 1);

  iters(i, :) = pf(xk, 1);

end

xStar = pf(xk, 1);
end


