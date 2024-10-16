function xOut = applyS_pddr_resolve(xIn, Jf, Jg_inv, gamma)
% we need to write a version of applyS that is much less confusing to do
% some error checking on
% THIS is simply the PDDR AOI operator.
% this version uses a handle specifically to the resolvent, not the prox
% operator (only different in certain cases)

a = Jf(xIn, gamma); %% prox op/resolvent
b = 2*a - xIn; %% reflected resolvent
c = Jg_inv(b, gamma); %% resolvent of scaled inverse
d = 2*c - b; %% reflected resolvent
xOut = -1*d; %% negative for PDDR


end


