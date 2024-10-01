function [xStar, objVals] = primal_dual_dr_test(x0,proxf,proxgconj,gamma,obj)

maxIter = 100;

xk = x0;
zk = zeros(size(xk));

objVals = zeros([maxIter 1]);
for i = 1:maxIter
  xkp1 = proxf(xk - zk, gamma);
  zkp1 = proxgconj(zk + 2*xkp1 - xk, gamma); 

  xk = xkp1;
  zk = zkp1;

  objVals(i, :) = obj(xk);

end

xStar = xk;
end