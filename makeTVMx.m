function outMx = makeTVMx(sImg)
% outMx = makeTVMx
% makes an explicit matrix representation for a total variation gradient
% operator
% assume TV is working on an nxn matrix/image
% TV takes in an n*n matrix and outputs a n*n*2 tensor
% so let's make this a matrix from R^(n^2) -> R^(2n^2)

n = sImg(1);
m = sImg(2);
N = n*m;
outMx = zeros(2*N, N);

for i = 1:N
    ei = zeros([N, 1]);
    ei(i) = 1;
    ei = reshape(ei, sImg);
    o = computeGradient(ei);
    outMx(:, i) = o(:);
end
end
