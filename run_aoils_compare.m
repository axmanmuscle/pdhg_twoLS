% load('kData_knee.mat', 'kData');
load('brain.mat', 'd2');
kData = d2;
% load('ankle.mat')
% kData = d1;
rng(20240429);

kData = kData./max(abs(kData(:)));

pfSamples = kData;

sImg = size( kData, (1:2) );

vdSigFrac = 0.3;
sampleFraction = 0.3;

wavSplit = makeWavSplit( sImg );
% wavSplit = zeros(2);  wavSplit(1,1) = 1;
[ fsr, sFSR ] = mri_makeFullySampledCenterRegion( sImg, wavSplit );

nSamples = round( sampleFraction * prod( sImg ) );
vdSig = round( vdSigFrac * sImg );

sampleMask = mri_makeSampleMask( sImg, nSamples, vdSig );
wavMaskACR = mri_makeSampleMask( sImg, sum(sampleMask(:)), vdSig, 'startMask', fsr>0 );

fftSamples_wavACR = bsxfun( @times, kData, wavMaskACR );

coilIdx = 3;
coilData = fftSamples_wavACR(:,:,coilIdx);
pfData = kData(:, :, coilIdx);
fftSamples_wavACR_pf = coilData;

% fftSamples_wavACR( ceil( ( sImg(1) + 1 ) / 2 ) + round( sFSR(1) / 2 ) : end, :, : ) = 0;

fftSamples_wavACR_pf( ceil( sImg(1) / 2 ) + round( sFSR(1) / 2 ) : end, : ) = 0;
fftSamples_wavACR_pf( fsr > 0 ) = pfData( fsr > 0 );


[~,phaseImg] = mri_reconPartialFourier( fftSamples_wavACR_pf, sFSR );
phases = angle( phaseImg );

N = 5000;
gammas = 10.^(linspace(-6, 2, 30));
aoiObj = zeros([numel(gammas) N]);
aoiLSObj = zeros([numel(gammas) N]);
for gamma_idx = 1:numel(gammas)
    gamma = gammas(gamma_idx);
    [xStar, objVals] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        'alg', 'pdhg_bothls', 'gamma', gamma, 'N', N );
    % [xStar_ls, objVals_ls] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    %     'alg', 'primalDualDR_avgOp_wls', 'gamma', gamma, 'N', N );
    aoiObj(gamma_idx, :) = objVals;
end


% linesearchRecons = cell2mat( linesearchRecons ); linesearchRecon = mri_reconRoemer( linesearchRecons );
% figure; imshowscale(abs(linesearchRecon)); title('line search')
% 
