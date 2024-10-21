function [xStar, objVals, alphasUsed] = gPDHG_wls( z0, proxf, proxgconj, f, g, theta, A, B, gamma, varargin )
  % implements our new generalized PDHG with line search
  % make this take in proxf and proxgconj not tilde versions
  % just like pdhg

  p = inputParser;
  p.addParameter( 'f', [] );
  p.addParameter( 'maxIter', 100, @ispositive );
  p.addParameter( 'y0', [], @isnumeric );
  p.parse( varargin{:} );
  maxIter = p.Results.maxIter;
  y0 = p.Results.y0;

  if isnumeric(A) && isnumeric(B)
    m = size( A, 1 );
    n = size( A, 2 );
    applyA = @(x) A*x;
    applyAt = @(x) A'*x;
    applyB = @(x) B*x;
    applyBt = @(x) B'*x;
  
    %theta = diag( A*A' + B*B' );
    %theta = theta(1);

  elseif isa(A, 'function_handle') && isa(B, 'function_handle')
    applyA = @(x) A(x, 'notransp');
    applyAt = @(x) A(x, 'transp');
    applyB = @(x) B(x, 'notransp');
    applyBt = @(x) B(x, 'transp');

  else
      disp('both A, B must be numeric or functions');
      return
  end
  if numel( y0 ) > 0
      m = numel( y0 );
  else
      Az0 = applyA( z0 );
      m = numel( Az0 );
  end
  n = numel( z0 );

  if numel( y0 ) > 0
      x0 = [ z0(:); y0(:); ];
  else
    x0 = [ z0(:); Az0(:); ];
  end

  function out = applyAB( in, op )
      if nargin < 2 || strcmp( op, 'notransp' )
        out = applyA( in(1:n) ) + applyB( in(n+1:end) );
      else
        out = zeros( n + m, 1 );
        out(1:n) = applyAt( in );
        out(n+1:end) = applyBt( in );
      end
  end
  
  if nargout > 1
      objVals = zeros( maxIter, 1 );
  end
  
  if nargout > 2
      alphasUsed = zeros( maxIter, 1 );
  end

  
  objFun = @(x) f(proxf(x, gamma)) + g(applyA(proxf(x, gamma)));
  
  doLineSearch = true;
  doLineSearchTest = true;
  
  %%% parameters
  alpha_bar = 0.5; % alpha_bar
  eps = 0.03; % eps for (1 - eps) || rbar_k || in linesearch
  epsHat = 0.05; % for line search test
  alpha0 = 50; % starting alpha
  alpha_change = 1/1.4; % factor for change in alpha during linesearch
  
  k = ceil( -log( alpha0 / alpha_bar ) / log( alpha_change ) );
  alphas = alpha0 .* ( alpha_change.^(0:k) );
  alphas(end) = alpha_bar;
  
  S = @(xIn, zIn, tauk, thetak) applyS_pdhgwLS_op(xIn, zIn, proxf, proxgconj, tauk, thetak, theta, applyA, applyAt, @applyAB);
  
  tau0 = 1;
  theta0 = 1;
  
  xk = x0;
  tauk = tau0;
  thetak = theta0;
  if numel( y0 ) > 0
      zk = y0;
  else
    zk = Az0;
  end
  
  nAlphas = numel( alphas );
  normRks = zeros( nAlphas, 1 );
  xs = cell( 1, nAlphas );
  rks = cell( 1, nAlphas );
  tauks = cell( 1, nAlphas );
  thetaks = cell( 1, nAlphas );
  xks = cell( 1, nAlphas );
  zks = cell( 1, nAlphas );
  
  [sxk, xOut, zOut, tauk, thetak] = S(xk(1:n), zk, tauk, thetak);
  rk = sxk - xk;
  
  normRk = sqrt(real(dotP(rk, rk)));
  
  for optIter = 1:maxIter
      lastRk = rk;
  
      if doLineSearch == true
  
          parfor alphaIndx = 1 : nAlphas
              alpha = alphas( alphaIndx );
              xAlpha = xk + alpha * rk;
              xs{alphaIndx} = xAlpha;
              [sxAlpha, xOutAlpha, zOutAlpha, taukAlpha, thetakAlpha] = S(xOut, zOut, tauk, thetak);
              rkAlpha = sxAlpha - xAlpha;
              rks{alphaIndx} = rkAlpha;
              tauks{alphaIndx} = taukAlpha;
              thetaks{alphaIndx} = thetakAlpha;
              xks{alphaIndx} = xOutAlpha;
              zks{alphaIndx} = zOutAlpha;
              
              normRks( alphaIndx ) = sqrt( real( dotP( rkAlpha, rkAlpha ) ) );
          end
  
          normRk_bar = normRks(end);
          normRks(end) = 0;
          bestAlphaIndx = find( normRks <= ( 1-eps ) * normRk_bar, 1 );
  
          alphaUsed = alphas( bestAlphaIndx );
          xk = xs{ bestAlphaIndx };
          rk = rks{ bestAlphaIndx };
          tauk = tauks{ bestAlphaIndx };
          thetak = thetaks{ bestAlphaIndx };
          xOut = xks{ bestAlphaIndx };
          zOut = zks{ bestAlphaIndx };
  
      else
  
          alphaUsed = alpha_bar;
          xk = xk + alpha_bar * rk;
          [sx, xOut, zOut, tauk, thetak] = S(xOut, zOut, tauk, thetak);
          rk = sx - xk;
      end
  
      objValue = objFun(xk(1:n));
  
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
  xk = proxf(xk(1:n), gamma);
  xStar = xk;

end