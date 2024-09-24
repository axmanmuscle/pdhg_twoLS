function out = proxF(x, b, eps)
  if norm(x - b, 2) < eps
    out = x;
  else
    xperp = (x - b)/norm(x-b,2) * eps;
    out = b + xperp;
  end
end