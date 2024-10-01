function out = tvProx(in,t, lambda)
%tvProx prox operator of tvnorm

out = in - (1 / lambda) * proxConjL2L1(in / lambda, lambda, t);


end