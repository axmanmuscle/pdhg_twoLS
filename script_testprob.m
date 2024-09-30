%%%% Test prob
%%%% start with min 0.5*|| Ax - b ||_2^2

rng(20240929);
n = 25;
m = 15;
A = 15*randn([m n]);
x = 3*randn([n 1]);

b = A*x;
noise = 0.1 * randn([m 1]);
b = b + noise;

norm(A*x - b)

f = @(x_in) 0.5 * norm(A*x_in - b);
gradf = @(x_in) A'*(A*x_in - b);
proxf = @(x_in, t) inv(A'*A + (1/t)*eye(size(A'*A))) * (A'*b + (1/t)*x_in);

x0 = zeros(size(x));

xgrad = x0;
xprox = x0;
alpha = 2e-5;
for i = 1:5000
    xgrad = xgrad - alpha*gradf(xgrad);
    xprox = proxf(xprox, alpha);
end