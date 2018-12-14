function TestKerasModel
    %% load weights and images
    w = load('C:\CDocuments\python\NaturalImages\weights.mat');
    images = load('C:\CDocuments\python\NaturalImages\xtPlot_natImageCombinedFilteredContrast_20scenes_2s_10traces_355phi_100Hz_005devFrac.mat');

    leftFilt = w.weight3;
    leftBias = w.biases3;

    rightFilt = w.weight4;
    rightBias = w.biases4;

    finalMult = w.weight8;
    finalBias = w.biases8;

    %% rearrange dimensions of python data
    trainX = permute(images.trainX,[3 2 1]);
    devX = permute(images.devX,[3 2 1]);
    testX = permute(images.testX,[3 2 1]);

    trainY = permute(images.trainY,[2 1]);
    devY = permute(images.devY,[2 1]);
    testY = permute(images.testY,[2 1]);

    %% get predictions
    m = size(devX,1);
    pred = zeros(m,181);

    for mInd = 1:m
        pred(mInd,:) = HRC(squeeze(devX(mInd,:,:)),leftFilt,rightFilt,finalMult,leftBias,rightBias,finalBias);
    end
    
    devY_trim = devY(:,1:size(pred,2));
    
    %% calculate R2
    R2 = zeros(m,1);
    for mInd = 1:m
        thisPred = pred(m,:)';
        thisY = devY_trim(m,:)';
        
        R2(mInd) = 1 - sum((thisPred-thisY).^2)./sum((thisY-mean(thisY)).^2);
    end
    
    MakeFigure;
    plot(R2);
    
    globalY = devY_trim(:);
    globalPred = pred(:);
    
    globalR2 = 1-sum((globalY-globalPred).^2)./sum((globalY-mean(globalY)).^2);
    
    disp(globalR2)
    
    MakeFigure;
    scatter(devY_trim(:),pred(:));
end

function pred = HRC(img,h1,h2,hf,b1,b2,bf)
    arm1 = conv2(img,h1,'valid')+b1;
    arm2 = conv2(img,h2,'valid')+b2;
    
    arm3 = conv2(img,fliplr(h1),'valid')+b1;
    arm4 = conv2(img,fliplr(h2),'valid')+b2;
    
    pred = sum((arm1.*arm2-arm3.*arm4)*hf+bf,2);
end