function GenerateXtPlotDataset
    %% changeable parameters
    % total number of scenes in the database
    % can't be less than 3 so that there is one natural scene saved for the
    % dev and test sets.
    folderPath = 'G:\My Drive\data_sets\nn_RigidRot\natural_images';
    
    numScenes = 20;

    % where in space to start each plot
    xEnd = 360; % degrees
    xStep = 360; % degrees
    
    yEnd = -1; % use -1 for max
    if yEnd == -1
        yEst = 100;
    else
        yEst = yEnd;
    end
    yStep = 5; % degrees
    
    % how often and how far to sample in space for each plot
    phaseEnd = 360; % degrees
    phaseStep = 5; % degrees
    
    % how often and how far to sample in time
    sampleFreq = 100; % Hz
    totalTime = 1; % s
    
    % number of velocity traces to generate for each point in space
    numTraces = 1;
    
    % standard devation of gaussian noise (in contrast) to be added to each
    % xt plot
    noiseStd = 0;
    
    % velocity parameters
    % this is the halflife of the autocorrelation of turning I measured
    % from the data in my 2018 paper
    halfLife = 0.2; % s
    velStd = 100; % degrees/s

    % fraction of dataset to keep for dev and test sets
    devFrac = 0.05;

    % make a string to tag the saved file with
    saveStr = [ 'xtPlot' ...
                '_ns' num2str(numScenes) ...
                '_xe' num2str(xEnd) ...
                '_xs' num2str(xStep) ...
                '_ye' num2str(yEst) ...
                '_ys' num2str(yStep) ...
                '_pe' num2str(phaseEnd) ...
                '_ps' num2str(phaseStep) ...
                '_sf' num2str(sampleFreq) ...
                '_tt' num2str(totalTime) ...
                '_nt' num2str(numTraces*2) ...
                '_hl' num2str(halfLife) ...
                '_vs' num2str(velStd) ...
                '_df' num2str(devFrac) ...
                '_no' num2str(noiseStd) ...
                ];
    saveStr(saveStr=='.') = '-';
    
    %% calculate how big this bad boy is going to be
    scenesMult = numScenes;
    stepMultX = xEnd/xStep;
    stepMultY = yEst/yStep-1;
    phaseMult = phaseEnd/phaseStep;
    timeMult = sampleFreq*totalTime+1;
    tracesMult = numTraces*2;
    
    imagesMult = scenesMult*stepMultX*stepMultY*phaseMult*timeMult*tracesMult;
    velMult = timeMult*numScenes*stepMultX*stepMultY*tracesMult;
    
    predSize = (imagesMult+velMult)*64/8/2^30;
    disp(['expected size is ' num2str(predSize) ' gb']);
    
    %% get natural images
    inputImages = 'natImageCombinedFilteredContrast';
    dataPath = fullfile(folderPath,'xy',inputImages);
    natImage = load(dataPath);
    xyPlot = natImage.xyPlot;
    clear natImage;
    
    rng('shuffle');
    
    % randomy shuffle phase of natural images
    for ss = 1:size(xyPlot,3)
        xyPlot(:,:,ss) = circshift(xyPlot(:,:,ss),[0 floor(rand*size(xyPlot,2)) 0]);
    end
        
    % radnomly shuffle scene order
    newInd = randperm(numScenes)';
    
    % crop dataset
    xyPlot = xyPlot(:,:,newInd(1:numScenes));
    
    xyRes = 360/size(xyPlot,2);
    yLim = xyRes*size(xyPlot,1);
    numTime = sampleFreq*totalTime+1;
    
    if yEnd == -1
        yEnd = yLim;
    end
    
    %% choose phi
    phi = (0:phaseStep:phaseEnd-phaseStep)';
    
    %% get xy points
    xSamplePoints = (0:xStep:xEnd-xStep)';
    ySamplePoints = (1:yStep:yEnd-yStep)';
    
    [xMat,yMat] = meshgrid(xSamplePoints,ySamplePoints);
    
    startX = xMat(:);
    startY = yMat(:);
    
    %% get velocities
    tau = halfLife/log(2);
    filterBufferTime = 10*halfLife;
    filterBufferNum = filterBufferTime*sampleFreq;
    
    % assume velocity is normally distributed with a std of 100 in a
    % natural setting. This is based on fly turning rates
    velNoFilter = velStd*randn(filterBufferNum+numTime,length(startX),numTraces);

    % filter the trace to give it an autocorrelation
    filtT = linspace(0,totalTime+filterBufferTime,filterBufferNum+numTime)';
    % multiply the filter by a corrective constant so that the final trace
    % has the correct std of velocity across traces (but not within a
    % trace). If a trace is particularly short it will look like low
    % variance due to the autocorrelation
    filter = exp(-filtT/tau)*sqrt(1-exp(-2/(tau*sampleFreq)));
    vel = ifft(fft(filter).*fft(velNoFilter));
    vel = vel(1:numTime,:,:);
    
    position = cumsum(vel)/sampleFreq;
    position = position-position(1,:,:);

    
    %% generate xt plots
    xtPlot = MakeXtPlotShiftVel(xyPlot,startX,startY,position,phi);
    clear xyPlot;
    
    % add noise
    noise = randn(size(xtPlot))*noiseStd;
    xtPlot = xtPlot + noise;
    
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
    devNum = ceil(numScenes*devFrac);
    testNum = devNum;
    trainNum = numScenes-devNum-testNum;

    % get train/dev/train inds
    trainInd = 1:numScenes-2*devNum;

    devInd = numScenes-2*devNum+1:numScenes-devNum;
    testInd = numScenes-devNum+1:numScenes;
    
    % define the test and train x/y pairs
    train_in = xtPlot(:,:,trainInd,:,:);
    train_in = reshape(train_in,[trainNum*numSamples*numTraces sizeT sizeX]);
    
    dev_in = xtPlot(:,:,devInd,:,:);
    dev_in = reshape(dev_in,[devNum*numSamples*numTraces sizeT sizeX]);
    
    test_in = xtPlot(:,:,testInd,:,:);
    test_in = reshape(test_in,[testNum*numSamples*numTraces sizeT sizeX]);
    
    clear xtPlot;
    
    velShaped = reshape(vel,[numTime numSamples*numTraces])';
    
    train_out = repmat(velShaped,[trainNum 1]);
    dev_out = repmat(velShaped,[devNum 1]);
    test_out = repmat(velShaped,[testNum 1]);
    
    clear velShaped;
    
    % matlab transposes for some reason
    train_in = permute(train_in,[3 2 1]);
    dev_in = permute(dev_in,[3 2 1]);
    test_in = permute(test_in,[3 2 1]);
    
    train_out = train_out';
    dev_out = dev_out';
    test_out = test_out';
    
    %% save xtPlots
    savePath = fullfile(folderPath,'xt',saveStr);
    save(savePath,'train_in','train_out','dev_in','dev_out','test_in','test_out','sampleFreq','phaseStep','-v7.3');
end