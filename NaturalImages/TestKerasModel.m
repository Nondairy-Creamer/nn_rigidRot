function TestKerasModel
    %% load weights and images
    w = load('C:\CDocuments\python\nn_RigidRot\NaturalImages\weightsHrc.mat');
    images = load('C:\CDocuments\python\nn_RigidRot\NaturalImages\xtPlot_natImageCombinedFilteredContrast_20scenes_2s_10traces_355phi_100Hz_005devFrac.mat');

    weightNames = fieldnames(w);
    weightCell = cell(length(weightNames),1);
    
    for wInd = 1:length(weightNames)
        weightCell{wInd} = double(w.(weightNames{wInd}));
    end
    
    h = weightCell(1:2:end);
    b = weightCell(2:2:end);
    
    h{1} = randn(21,10)/10;
    h{2} = randn(21,10)/10;
    
    %% rearrange dimensions of python data
    trainX = permute(images.trainX,[3 2 1]);
    devX = permute(images.devX,[3 2 1]);
    testX = permute(images.testX,[3 2 1]);

    trainY = permute(images.trainY,[2 1]);
    devY = permute(images.devY,[2 1]);
    testY = permute(images.testY,[2 1]);
    
    
    %% get predictions
    m = size(devX,1);
    pred = zeros(m,201);
    
    for mInd = 1:m
        natScene = squeeze(devX(mInd,:,:));
        natScene_norm = natScene/std(natScene(:));
%         pred(mInd,:) = LnModel(natScene_norm,h,b);
        pred(mInd,:) = HrcModel(natScene_norm,h,b);
    end
    
    devY_trim = devY(:,1:size(pred,2));
    
    %% calculate R2
    R2 = zeros(m,1);
    for mInd = 1:m
        thisPred = pred(mInd,:)';
        thisY = devY_trim(mInd,:)';
        
        R2(mInd) = 1 - sum((thisPred-thisY).^2)./sum((thisY-mean(thisY)).^2);
    end
    
    MakeFigure;
    plot(R2);
    
    globalY = devY_trim(:,1);
    globalPred = pred(:,1);
    
    for rr = -100:100
        globalR2(rr+101) = 1-sum((globalY-rr*globalPred).^2)./sum((globalY-mean(globalY)).^2);
    end
    max(globalR2)
    
    MakeFigure;
    scatter(globalY,globalPred);
    
    for hInd = 1:length(h)
        H{hInd} = fft2(h{hInd});
    end
    
    RH = real(H{1}.*conj(H{2})-fliplr(H{1}.*conj(H{2})));
    
    imagesc(fftshift(RH));
end

function pred = LnModel(img,h,b)
    arm1 = imfilter(img,h{1},'same')+b{1};
    arm1(arm1<0) = 0;
    
    arm2 = imfilter(img,fliplr(h{1}),'same')+b{1};
    arm2(arm2<0) = 0;
    
    pred = sum((arm1-arm2)*h{2}+b{2},2);
end

function pred = HrcModel(img,h,b)
    arm1 = imfilter(img,h{1},'same')+b{1};
    arm2 = imfilter(img,h{2},'same')+b{2};
    
    arm3 = imfilter(img,fliplr(h{1}),'same')+b{1};
    arm4 = imfilter(img,fliplr(h{2}),'same')+b{2};
    
    pred = sum((arm1.*arm2-arm3.*arm4)*h{3}+b{3},2);
end