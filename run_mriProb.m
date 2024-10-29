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

gpdhgRecons = cell(1,1,8);
pdhgRecons = cell(1,1,8);

for coilIdx = 1:8
    disp(coilIdx);
    coilData = fftSamples_wavACR(:,:,coilIdx);
    pfData = kData(:, :, coilIdx);
    fftSamples_wavACR_pf = coilData;

    % fftSamples_wavACR( ceil( ( sImg(1) + 1 ) / 2 ) + round( sFSR(1) / 2 ) : end, :, : ) = 0;

    fftSamples_wavACR_pf( ceil( sImg(1) / 2 ) + round( sFSR(1) / 2 ) : end, : ) = 0;
    fftSamples_wavACR_pf( fsr > 0 ) = pfData( fsr > 0 );


    [~,phaseImg] = mri_reconPartialFourier( fftSamples_wavACR_pf, sFSR );
    phases = angle( phaseImg );

    N = 100;
    gamma = 10.^(-3);
    tau0 = 1;

    [xStar_pdhg, objVals_pdhg] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        'alg', 'pdhg', 'gamma', gamma, 'N', N );
    [xStar_gpdhg, objVals_gpdhg] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
        'alg', 'gpdhg', 'N', N, 'tau0', tau0 );
    % [xStar_ls, objVals_ls] = mri_reconCSPFHomodyne( fftSamples_wavACR_pf, sFSR, 'wavSplit', wavSplit, ...
    % 'alg', 'primalDualDR_avgOp_wls', 'gamma', gamma, 'N', N );

    gpdhgRecons{1, 1, coilIdx} = xStar_gpdhg;
    pdhgRecons{1,1,coilIdx} = xStar_pdhg;


end


linesearchRecons = cell2mat( gpdhgRecons ); linesearchRecon = mri_reconRoemer( linesearchRecons );
figure; imshowscale(abs(linesearchRecon)); title('gpdhg recon')

pdhgRecons = cell2mat( pdhgRecons ); pdhgRecon = mri_reconRoemer( pdhgRecons );
figure; imshowscale(abs(pdhgRecon)); title('pdhg recon')
% 
