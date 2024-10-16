function xOut = applyS_dr(xIn, pf, pg, gamma)
% we need to write a version of applyS that is much less confusing to do
% some error checking on
% THIS is simply the DR AOI operator.

a = pf(xIn, gamma); %% prox op/resolvent of f
b = 2*a - xIn; %% reflected resolvent
c = pg(b, gamma); %% prox op/resolvent of g
d = 2*c - b; %% reflected resolvent
xOut = d;


end