function TestKerasModel
    %% load weights and images
%     funcPath = fileparts(mfilename('fullpath'));
    imageFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters';
    imageName = 'xtPlot_natImageCombinedFilteredContrast_40scenes_1s_2traces_355phi_1000Hz_005devFrac.mat';
%     imageName = 'xtPlot_natImageCombinedFilteredContrast_20scenes_2s_10traces_355phi_100Hz_005devFrac.mat.mat';
    imagePath = fullfile(imageFolder,imageName);
    images = load(imagePath);


    
    % boolean to decide whether to sum over space
    sumOverSpace = false;
    useOneTime = false;
    %% choose filter type to use
    filterType = 'gradientDescentSep';
    
    filtSizeIn = [21 11];
    
    b{1} = 0;
    b{2} = 0;
    b{3} = 0;
    
    switch filterType
        case 'gradientDescentSep'
            filterPath = fullfile(filterFolder,'weightsSep.mat');
%             filterPath = fullfile(filterFolder,'weightsSep_NoSum.mat');
%             filterPath = fullfile(filterFolder,'weightsSep_highRes_noSum.mat');
%             filterPath = fullfile(filterFolder,'weights_highRes_longRun_sep.mat');

            w = load(filterPath);
            
            weightNames = fieldnames(w);
            weightCell = cell(length(weightNames),1);

            for wInd = 1:length(weightNames)
                weightCell{wInd} = double(w.(weightNames{wInd}));
            end
            
            weightCell{5}(15:end) = 0; 
            
            h{1} = weightCell{1}*weightCell{3};
            h{2} = weightCell{5}*weightCell{7};
            h{3} = 1;
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            h = cellfun(@(x)[x zeros(size(x,1),1)],h,'UniformOutput',false);
            
            h{3} = h{3}(1);
            
        case 'gradientDescent'
%             filterPath = fullfile(filterFolder,'weightsHrc.mat');
            filterPath = fullfile(filterFolder,'weightsPretty.mat');

            w = load(filterPath);

            weightNames = fieldnames(w);
            weightCell = cell(length(weightNames),1);

            for wInd = 1:length(weightNames)
                weightCell{wInd} = double(w.(weightNames{wInd}));
            end
            
            h = weightCell(1:2:end);
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            h = cellfun(@(x)[x zeros(size(x,1),1)],h,'UniformOutput',false);
            h{3} = h{3}(1);
            h{2}(1,1) = 0;
            
            thresh = 0.11;
            h{1}(abs(h{1})<thresh) = 0;
            h{2}(abs(h{1})<thresh) = 0;
            
        case 'newRandom'
            h{1} = randn(filtSizeIn)/10;
            h{2} = randn(filtSizeIn)/10;
            h{3} = 1;
%             save('randomH','h');
        case 'savedRandom'
            filterPath = fullfile(filterFolder,'randomH.mat');

            w = load(filterPath);
            
            h = w.h;
            
        case 'gaussian'
            h{1} = zeros(filtSizeIn);
            h{2} = zeros(filtSizeIn);
            h{3} = 1;
            
        case 'deltas'
            h{1} = zeros(filtSizeIn);
            h{2} = zeros(filtSizeIn);
            h{3} = 1;
            
            fillX1 = [1 1]';
            fillY1 = [1 2]';
            fillVal1 = [2 -1]';
            
            fillX2 = [(1:10)+1 (1:10)+1]';
            fillY2 = [ones(1,10) 1+ones(1,10)]';
            fillVal2 = [ones(1,10) -ones(1,10)]';
            
            for ff = 1:length(fillX1)
                h{1}(fillY1(ff),fillX1(ff)) = fillVal1(ff);
            end
            
            for ff = 1:length(fillX2)
                h{2}(fillY2(ff),fillX2(ff)) = fillVal2(ff);
            end
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            
        case 'deltas2'
            h{1} = zeros(filtSizeIn);
            h{2} = zeros(filtSizeIn);
            h{3} = 1;
            
            fillX1 = [1:10 1:10]';
            fillY1 = [ones(1,10) ones(1,10)+1]';
            fillVal1 = [ones(1,10) -2*ones(1,10)]';
            
            fillX2 = [1 1]';
            fillY2 = [1 2]';
            fillVal2 = [1 -2]';
            
            for ff = 1:length(fillX1)
                h{1}(fillY1(ff),fillX1(ff)) = fillVal1(ff);
            end
            
            for ff = 1:length(fillX2)
                h{2}(fillY2(ff),fillX2(ff)) = fillVal2(ff);
            end
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            
        case 'deltas3'
            h{1} = zeros(filtSizeIn);
            h{2} = zeros(filtSizeIn);
            h{3} = 1;
            
            fillX1 = [1 2 3 1 2 3 4:9 4:9 1]';
            fillY1 = [1 1 1 2 2 2 1 1 1 1 1 1 2 2 2 2 2 2 3]';
            fillVal1 = [-2 -1 -1 1 1 2 -.5 -.5 -.5 -.5 -.5 -.5 .5 .5 .5 .5 .5 .5 .5]';
            
            fillX2 = [2 2 5 6 5 6 1 1 3]';
            fillY2 = [1 2 1 1 2 2 1 3 1]';
            fillVal2 = [-1 2 -.5 -.5 .5 .5 -.5 -.5 -.5]';
            
            for ff = 1:length(fillX1)
                h{1}(fillY1(ff),fillX1(ff)) = fillVal1(ff);
            end
            
            for ff = 1:length(fillX2)
                h{2}(fillY2(ff),fillX2(ff)) = fillVal2(ff);
            end
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            
        case 'deltaSquare'
            h{1} = zeros(filtSizeIn);
            h{2} = zeros(filtSizeIn);
            h{3} = 1;
            
%             fillX1 = (1:4)';
%             fillY1 = (1:10)';
%             
%             fillX2 = fillX1+3;
%             fillY2 = fillY1+10;

            fillX1 = (1)';
            fillY1 = (1)';
            
            fillX2 = fillX1+1;
            fillY2 = fillY1+1;
            
            h{1}(fillY1,fillX1) = 1/10;
            h{2}(fillY2,fillX2) = 1/10;
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
    end
    
    
    filtSize = size(h{1});
    
    padX = (filtSize(2)-1)/2;
    padT = (filtSize(1)-1)/2;
    
    %% rearrange dimensions of python data
    trainImage = permute(images.trainX,[3 2 1]);
    devImage = permute(images.devX,[3 2 1]);
    testImage = permute(images.testX,[3 2 1]);

    trainVel = permute(images.trainY,[2 1]);
    devVel = permute(images.devY,[2 1]);
    testVel = permute(images.testY,[2 1]);
    
    sizeImage = size(devImage);
    sizeVelIn = size(devVel);
    
    %% truncate velocity vectors to account for valid convolutions
    sizePred = [sizeVelIn(1) sizeVelIn(2)-2*padT sizeImage(3)-2*padX];
    
    devVelTrunc = repmat(devVel(1:sizePred(1),1:sizePred(2)),[1 1 sizePred(3)]);
    
    %% get predictions
    m = size(devImage,1);
    
    pred = zeros(size(devVelTrunc));
    
    for mInd = 1:m
        natScene = squeeze(devImage(mInd,:,:));
        natScene_norm = natScene/std(natScene(:));
%         pred(mInd,:) = LnModel(natScene_norm,h,b);
        pred(mInd,:,:) = HrcModel(natScene_norm,h,b);
    end
    
    if sumOverSpace
        pred = sum(pred,3);
        devVelTrunc = devVelTrunc(:,:,1);
    end
    
    if useOneTime
        pred = pred(:,1,:);
        devVelTrunc = devVelTrunc(:,1,:);
    end
    
    %% displace the steadystate response to sine waves
    H = cell(length(h),1);
    for hInd = 1:length(h)
        H{hInd} = fftshift(fft2(h{hInd}));
    end
    
    RH = real(H{1}.*conj(H{2})-fliplr(H{1}.*conj(H{2})));
    
    %% calculate R2 across all examples and find the best R2 by scalling global pred
    globalY = devVelTrunc(:);
    globalPred = pred(:);
    
    mult = lsqlin(globalPred,globalY);
    
    bestR2 = 1-sum((globalY-mult*globalPred).^2)./sum((globalY-mean(globalY)).^2);

    finalPred = pred*mult;
    globalPred = finalPred(:);
    
    %% calculate R2 for each example
    R2 = zeros(m,1);
    for mInd = 1:m
        thisPred = finalPred(mInd,:,:);

        thisVel = devVelTrunc(mInd,:,:);
        
        thisPred = thisPred(:);
        thisVel = thisVel(:);
        
        R2(mInd) = 1 - sum((thisPred-thisVel).^2)./sum((thisVel-mean(thisVel)).^2);
    end
    
    %% plot figs
    disp(bestR2);
        
    MakeFigure;
    max_abs_h = max(abs([h{1}(:); h{2}(:)]));
    subplot(2,2,1);
    imagesc(h{1});
    caxis([-max_abs_h max_abs_h]);
    
    subplot(2,2,2);
    imagesc(h{2})
    caxis([-max_abs_h max_abs_h]);

    subplot(2,2,3);
    imagesc(RH);
    
    colormap(flipud(cbrewer('div','RdBu',100)));
    
    MakeFigure;
    plot(R2);
    xlabel('example');
    ylabel('r2');
    
    MakeFigure;
    scatter(globalY,globalPred);
    xlabel('scene velocity');
    ylabel('model prediction');
    
end

function pred = LnModel(img,h,b)
    arm1 = conv2(img,h{1},'valid')+b{1};
    arm1(arm1<0) = 0;
    
    arm2 = imfilter(img,fliplr(h{1}),'valid')+b{1};
    arm2(arm2<0) = 0;
    
    pred = (arm1-arm2)*h{2}+b{2};
end

function pred = HrcModel(img,h,b)
    arm1 = conv2(img,h{1},'valid')+b{1};
    arm2 = conv2(img,h{2},'valid')+b{2};
    
    arm3 = conv2(img,fliplr(h{1}),'valid')+b{1};
    arm4 = conv2(img,fliplr(h{2}),'valid')+b{2};
    
    pred = (arm1.*arm2-arm3.*arm4)*h{3}+b{3};
end