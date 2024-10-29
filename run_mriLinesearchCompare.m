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

coilIdx = 4;
coilData = fftSamples_wavACR(:,:,coilIdx);
pfData = kData(:, :, coilIdx);
fftSamples_wavACR_pf = coilData;

% fftSamples_wavACR( ceil( ( sImg(1) + 1 ) / 2 ) + round( sFSR(1) / 2 ) : end, :, : ) = 0;

fftSamples_wavACR_pf( ceil( sImg(1) / 2 ) + round( sFSR(1) / 2 ) : end, : ) = 0;
fftSamples_wavACR_pf( fsr > 0 ) = pfData( fsr > 0 );


[~,phaseImg] = mri_reconPartialFourier( fftSamples_wavACR_pf, sFSR );
phases = angle( phaseImg );

N = 5000;
gammas = 10.^(-12:2);
taus = 10.^(-6:0.5:1);

% drObj = zeros([numel(gammas) N]);
% drOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);
% 
% drAOIObj = zeros([numel(gammas) N]);
% drAOIOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);
% 
% pddrObj = zeros([numel(gammas) N]);
% pddrOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);
% 
% pddrAOIObj = zeros([numel(gammas) N]);
% pddrAOIOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);

pdhgObj = zeros([numel(gammas) N]);
pdhgOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);

% aoiLSObj = zeros([numel(gammas) N]);
% aoiLSOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);

gpdhgObj = zeros([numel(gammas) N]);
gpdhgOut = zeros([numel(gammas) size(fftSamples_wavACR_pf(:), 1)]);

for gamma_idx = 1:numel(gammas)
    disp(gamma_idx)
    gamma = gammas(gamma_idx);
    tau0 = taus(gamma_idx);
    % [xStar_dr, objVals_dr] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    %     'alg', 'douglasRachford', 'gamma', gamma, 'N', N );
    % [xStar_drAOI, objVals_drAOI] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    %     'alg', 'douglasRachford_avgOp', 'gamma', gamma, 'N', N );
    % [xStar_pddr, objVals_pddr] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    %     'alg', 'primalDualDR', 'gamma', gamma, 'N', N );
    % [xStar_pddrAOI, objVals_pddrAOI] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    %     'alg', 'primalDualDR_avgOp', 'gamma', gamma, 'N', N );
    [xStar_pdhg, objVals_pdhg] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        'alg', 'pdhg', 'gamma', gamma, 'N', N );
    [xStar_gpdhg, objVals_gpdhg] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        'alg', 'gpdhg', 'N', N, 'tau0', tau0 );
    % [xStar_ls, objVals_ls] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        % 'alg', 'primalDualDR_avgOp_wls', 'gamma', gamma, 'N', N );
    
    
    % drObj(gamma_idx, :) = objVals_dr;
    % drOut(gamma_idx, :) = xStar_dr(:);
    % 
    % drAOIObj(gamma_idx, :) = objVals_drAOI;
    % drAOIOut(gamma_idx, :) = xStar_drAOI(:);
    % 
    % pddrObj(gamma_idx, :) = objVals_pddr;
    % pddrOut(gamma_idx, :) = xStar_pddr(:);
    % 
    % pddrAOIObj(gamma_idx, :) = objVals_pddrAOI;
    % pddrAOIOut(gamma_idx, :) = xStar_pddrAOI(:);

    pdhgObj(gamma_idx, :) = objVals_pdhg;
    pdhgOut(gamma_idx, :) = xStar_pdhg(:);
    % 
    gpdhgObj(gamma_idx, :) = objVals_gpdhg;
    gpdhgOut(gamma_idx, :) = xStar_gpdhg(:);
    % 
    % aoiLSObj(gamma_idx, :) = objVals_ls;
    % aoiLSOut(gamma_idx, :) = xStar_ls(:);
end


% linesearchRecons = cell2mat( linesearchRecons ); linesearchRecon = mri_reconRoemer( linesearchRecons );
% figure; imshowscale(abs(linesearchRecon)); title('line search')
% 
