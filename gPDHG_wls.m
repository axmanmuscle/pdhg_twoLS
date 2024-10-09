function [xStar, objVals, alphasUsed] = gPDHG_wls(x0,proxf,proxgconj, f, g, maxIter, theta, A, B, gamma)
% implements our new generalized PDHG with line search

n = size(x0, 1);

if isnumeric(A)
    if ~isnumeric(B)
        disp('both A, B must be numeric or functions');
        return
    end
    applyA = @(x) A*x;
    applyAt = @(x) A'*x;
    applyB = @(x) B*x;
    applyBt = @(x) B'*x;
else
    applyA = @(x) A(x, 'notransp');
    applyAt = @(x) A(x, 'transp');
    applyB = @(x) B(x, 'notransp');
    applyBt = @(x) B(x, 'transp');

end
m = size(applyA(x0), 1);
function out = applyAB( in, op )
    if nargin < 2 || strcmp( op, 'notransp' )
      out = applyA( in(1:n) ) + applyB( in(n+1:end) );
    else
      out = zeros( n + nMask, 1 );
      out(1:n) = applyA( in, 'transp' );
      out(n+1:end) = applyB( in, 'transp' );
    end
end

doLineSearch = true;
doLineSearchTest = true;
if nargout > 1
    objVals = zeros( maxIter, 1 );
end

if nargout > 2
    alphasUsed = zeros( maxIter, 1 );
end

objFun = @(x) f(proxf(x, gamma)) + g(proxf(x, gamma));

%%% parameters
alpha_bar = 0.5; % alpha_bar
% gamma = 3; % gamma for prox operators
eps = 0.03; % eps for (1 - eps) || rbar_k || in linesearch
epsHat = 0.05; % for line search test
alpha0 = 50; % starting alpha
alpha_change = 1/1.4; % factor for change in alpha during linesearch

k = ceil( -log( alpha0 / alpha_bar ) / log( alpha_change ) );
alphas = alpha0 .* ( alpha_change.^(0:k) );
alphas(end) = alpha_bar;

Rf = @(phi) 2*proxf(phi, gamma) - phi;
% Rgconj = @(phi) 2*proxgconj(phi, gamma) - phi;
Rg = @(phi) 2*proxgconj(phi, 1/gamma) - phi;

% S = @(phi) Rgconj(Rf(phi));
S_old = @(phi) -gamma * Rg( (1/gamma) * Rf(phi) );

S = @(phi, tauk, thetak) applyS(phi, proxf, proxgconj, tauk, thetak, theta, A, B);
S2 = @(phi, tauk, thetak, yk) applyS2(phi, proxf, proxgconj, yk, tauk, thetak, theta, A, B);

tau0 = 1;
theta0 = 1;

xk = x0;
tauk = tau0;
thetak = theta0;
yk = zeros(size(A, 1), 1);

nAlphas = numel( alphas );
normRks = zeros( nAlphas, 1 );
xs = cell( 1, nAlphas );
rks = cell( 1, nAlphas );

[sxk, tauk, thetak] = S(xk, tauk, thetak);
rk = sxk - xk;

normRk = sqrt(real(dotP(rk, rk)));

for optIter = 1:maxIter
    lastRk = rk;

    if doLineSearch == true

        parfor alphaIndx = 1 : nAlphas
            alpha = alphas( alphaIndx );
            xAlpha = xk + alpha * rk;
            xs{alphaIndx} = xAlpha;
            [sxAlpha, ~, ~] = S(xAlpha, tauk, thetak);
            rkAlpha = sxAlpha - xAlpha;   %#ok<PFBNS>
            rks{alphaIndx} = rkAlpha;
            normRks( alphaIndx ) = sqrt( real( dotP( rkAlpha, rkAlpha ) ) );
        end

        normRk_bar = normRks(end);
        normRks(end) = 0;
        bestAlphaIndx = find( normRks <= ( 1-eps ) * normRk_bar, 1 );

        alphaUsed = alphas( bestAlphaIndx );
        xk = xs{ bestAlphaIndx };
        rk = rks{ bestAlphaIndx };

    else

        alphaUsed = alpha_bar;
        xk = xk + alpha_bar * rk;
        [sx, tauk, thetak] = S(xk, tauk, thetak);
        rk = sx - xk;
    end

    objValue = objFun(xk);

    if nargout > 1
        objVals( optIter ) = objValue;
    end
    if nargout > 2
        alphasUsed( optIter ) = alphaUsed;
    end

    if doLineSearchTest == true
        doLineSearch = false;
        lastNormRk = normRk;
        normRk = sqrt( real( dotP( lastRk, lastRk ) ) );
        if normRk * lastNormRk == 0, continue; end

        %if real( dotP( rk, lastRk ) ) / ( normRk * lastNormRk ) > ( 1 - epsHat )
        if alphaUsed ~= alpha_bar  ||  normRk / lastNormRk < 1 - epsHat  % My own heuristic
            % The values of S are changing quickly in this region
            doLineSearch = true;
        end
    end


    fprintf('iter: %d   objVal: %d   res: %d  alpha: %d\n', optIter, objVals(optIter), norm(rk(:)), alphaUsed );

end
xk = proxf(xk, gamma);
xStar = xk;
end