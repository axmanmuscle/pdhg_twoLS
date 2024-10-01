function out = tvNorm(x)
%tvNorm norm for total variation
%   input x is dim n * m * 2 
%   computes correct norm for TV regularization (isotropic)

if numel(size(x)) ~= 3
    disp('x should be dim n*m*2');
    return
end

n = size(x, 1);
m = size(x, 2);

out = 0;
for i = 1:n
    for j = 1:m
        o = [x(i, j, 1); x(i, j, 2)];
        out = out + norm(o, 2);
    end
end


end