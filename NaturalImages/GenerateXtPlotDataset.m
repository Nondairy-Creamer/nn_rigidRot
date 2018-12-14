function GenerateXtPlots
    % get natural images
    inputImages = 'natImageCombinedFilteredContrast';
%     inputImages = 'natImageCombinedFiltered';
    folderPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(folderPath,inputImages);
    natImage = load(dataPath);
    xyPlot = natImage.xyPlot;
    clear natImage;
    
    rng('shuffle');
    
    % randomy shuffle phase of natural images
    for ss = 1:size(xyPlot,3)
        xyPlot(:,:,ss) = circshift(xyPlot(:,:,ss),[0 floor(rand*size(xyPlot,2)) 0]);
    end
    
    numScenes = 20;
    
    % crop dataset
    xyPlot = xyPlot(:,:,1:numScenes);
    
    xyRes = 360/size(xyPlot,2);
    yLim = xyRes*size(xyPlot,1);
    sampleFreq = 100; % Hz
    totalTime = 2; % s
    numTraces = 5;
    numTime = sampleFreq*totalTime+1;
    
    %% choose phi
    phi = (0:5:355)';
    
    %% get xy points
    xSamplePoints = (0:360:359)';
    ySamplePoints = (1:5:yLim)';
    
    [xMat,yMat] = meshgrid(xSamplePoints,ySamplePoints);
    
    startX = xMat(:);
    startY = yMat(:);
    
    %% get velocities
    % this is the halflife of the autocorrelation of turning I measured
    % from the data in my 2018 paper
    halfLife = 0.2;
    tau = halfLife/log(2);
    filterBufferTime = 10*halfLife;
    filterBufferNum = filterBufferTime*sampleFreq;
    
    % assume velocity is normally distributed with a std of 100 in a
    % natural setting. This is based on fly turning rates
    velStd = 100;
    velNoFilter = velStd*randn(filterBufferNum+numTime,length(startX),numTraces);

    filtT = linspace(0,totalTime+filterBufferTime,filterBufferNum+numTime)';
    filter = exp(-filtT/tau)*sqrt(1-exp(-2/(tau*sampleFreq)));
    vel = ifft(fft(filter).*fft(velNoFilter));
    vel = vel(1:numTime,:,:);
    
    position = cumsum(vel)/sampleFreq;
    position = position-position(1,:,:);
    
    % filter with an exponential filter so that the velocity is updated
    % according to 

    
    %% generate xt plots
    xtPlot = MakeXtPlotShiftVel(xyPlot,startX,startY,position,phi);
    clear xyPlot;
    
    %% randomize dataset and seperate into train/dev/test
    % convert to machine learning convention of having m examples in the
    % first dimension
    xtPlot = permute(xtPlot,[4 5 3 1 2]);
    xtPlot = [xtPlot flip(xtPlot,5)];
    
    % reshape velocity vector to be [sample*traces time]
    % duplicate left/right
    vel = cat(3,vel,-vel);
    
    numSamples = size(xtPlot,1);
    numTraces = size(xtPlot,2);
    numScenes = size(xtPlot,3);
    sizeT = size(xtPlot,4);
    sizeX = size(xtPlot,5);
    
    % fraction of data set to save for dev/test
    devFrac = 0.05;
    devNum = round(numScenes*devFrac);
    testNum = devNum;
    trainNum = round(numScenes-2*devNum);

    % get train/dev/train inds
    newInd = randperm(numScenes)';
    
    trainInd = newInd(1:end-2*devNum);
    devInd = newInd(end-2*devNum+1:end-devNum);
    testInd = newInd(end-devNum+1:end);
    
    % define the test and train x/y pairs
    trainX = xtPlot(:,:,trainInd,:,:);
    trainX = reshape(trainX,[trainNum*numSamples*numTraces sizeT sizeX]);
    
    devX = xtPlot(:,:,devInd,:,:);
    devX = reshape(devX,[devNum*numSamples*numTraces sizeT sizeX]);
    
    testX = xtPlot(:,:,testInd,:,:);
    testX = reshape(testX,[testNum*numSamples*numTraces sizeT sizeX]);
    
    clear xtPlot;
    
    velShaped = reshape(vel,[numTime numSamples*numTraces])';
    
    trainY = repmat(velShaped,[trainNum 1]);
    devY = repmat(velShaped,[devNum 1]);
    testY = repmat(velShaped,[testNum 1]);
    
    clear velShaped;
    
    % matlab transposes for some reason
    trainX = permute(trainX,[3 2 1]);
    devX = permute(devX,[3 2 1]);
    testX = permute(testX,[3 2 1]);
    
    trainY = trainY';
    devY = devY';
    testY = testY';
    
    totalTimeStr = num2str(totalTime);
    totalTimeStr = totalTimeStr(totalTimeStr~='.');
    
    devFracStr = num2str(devFrac);
    devFracStr = devFracStr(devFracStr~='.');
    
    %% save xtPlots
    savePath = fullfile(folderPath,['xtPlot_' inputImages '_' num2str(numScenes) 'scenes_' totalTimeStr 's_' num2str(numTraces) 'traces_' num2str(max(phi)) 'phi_' num2str(sampleFreq) 'Hz_' devFracStr 'devFrac' '.mat']);
    save(savePath,'trainX','trainY','devX','devY','testX','testY','-v7.3');
%     save(fullfile(folderPath,'xtPlot_test'),'trainX','trainY','devX','devY','testX','testY','-v7.3');
end