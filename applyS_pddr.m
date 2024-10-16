function xOut = applyS_pddr(xIn, pf, pgstar, gamma)
% we need to write a version of applyS that is much less confusing to do
% some error checking on
% THIS is simply the PDDR AOI operator.

a = pf(xIn, gamma); %% prox op/resolvent
b = 2*a - xIn; %% reflected resolvent
c = gamma * pgstar(b / gamma, 1 / gamma); %% resolvent of scaled inverse
d = 2*c - b; %% reflected resolvent
xOut = -1*d; %% negative for PDDR


end


