function TestKerasModel
    %% load weights and images
%     funcPath = fileparts(mfilename('fullpath'));
    imageFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters';
%     imageFolder = 'G:\My Drive\data_sets\nn_Odometry\composite_dataset';
%     filterFolder = 'G:\My Drive\data_sets\nn_Odometry\saved_parameters';
    imageName = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-05.mat';
%     imageName = 'image_dataset_filtered_rows_temp.mat';
%     imageName = 'xtPlot_natImageCombinedFilteredContrast_20scenes_2s_10traces_355phi_100Hz_005devFrac.mat.mat';
    imagePath = fullfile(imageFolder,imageName);
    images = load(imagePath);

    % boolean to decide whether to sum over space
    sumOverSpace = true;
    useOneTime = false;
    
    %% choose filter type to use
    filterType = 'gradientDescentLn';
    
    filtSizeIn = [21 11];
    
    b{1} = 0;
    b{2} = 0;
    b{3} = 0;
    
    switch filterType
        case 'gradientDescentSep'
            filterPath = fullfile(filterFolder,'hrc_model_sep_0-2filterTime_100filterSpace_100sampleFreq_5phaseStep.mat');
            filterPath = fullfile(filterFolder,'hrc_model_sep_0-2filterTime_20filterSpace_100sampleFreq_5phaseStep.mat');

            w = load(filterPath);
            
            weightNames = fieldnames(w);
            weightCell = cell(length(weightNames),1);

            for wInd = 1:length(weightNames)
                weightCell{wInd} = double(w.(weightNames{wInd}));
            end
                        
            h{1} = weightCell{1}*weightCell{5};
            h{2} = weightCell{3}*weightCell{7};
            h{3} = weightCell{9};
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
                        
        case 'gradientDescent'
            filterPath = fullfile(filterFolder,'hrc_model_1inputFilters_FalseSumSpace0-4filterTime_20filterSpace_20sampleFreq_5phaseStep.mat');

            w = load(filterPath);

            weightNames = fieldnames(w);
            weightCell = cell(length(weightNames),1);

            for wInd = 1:length(weightNames)
                weightCell{wInd} = double(w.(weightNames{wInd}));
            end
            
            h = weightCell(1:2:end);
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);
            
        case 'gradientDescentLn'
            filterPath = fullfile(filterFolder,'ln_model_2numFilt_TrueSumSpace_0-2filterTime_80filterSpace_100sampleFreq_5phaseStep.mat');

            w = load(filterPath);

            h{1} = w.weight2(:,:,1,1);
            h{2} = w.weight2(:,:,1,2);
            h{3} = 1;
            
            h = cellfun(@(x)rot90(x,2),h,'UniformOutput',false);

        case 'newRandom'
            h{1} = randn(filtSizeIn)/10;
            h{2} = randn(filtSizeIn)/10;
            h{3} = 1;
            save('randomH','h');
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
    
    padX = (filtSize(2)-1)/2+(size(h{3},2)-1)/2;
    padT = (filtSize(1)-1)/2;
    
    %% rearrange dimensions of python data
%     trainImage = permute(images.train_in,[3 2 1]);
    devImage = permute(images.dev_in,[3 2 1]);
%     testImage = permute(images.test_in,[3 2 1]);

%     trainVel = permute(images.train_out,[2 1]);
    devVel = permute(images.dev_out,[2 1]);
%     testVel = permute(images.test_out,[2 1]);
    
    sizeImage = size(devImage);
    sizeVelIn = size(devVel);
    
    %% truncate velocity vectors to account for valid convolutions
    sizePred = [sizeVelIn(1) sizeVelIn(2)-2*padT sizeImage(3)-size(h{1},2)+1-size(h{3},2)+1];
    
    devVelTrunc = repmat(devVel(1:sizePred(1),1:sizePred(2)),[1 1 sizePred(3)]);
    
    %% get predictions
    m = size(devImage,1);
    
    pred = zeros(size(devVelTrunc));
    
        
    devVelTrunc = devVelTrunc(:,:,1:size(pred,3));
    
    for mInd = 1:m
        natScene = squeeze(devImage(mInd,:,:));
        natScene_norm = natScene/std(natScene(:));
%         pred(mInd,:) = LnModel(natScene_norm,h,b);
        pred(mInd,:,:) = HrcModel(natScene_norm,h,b);
    end

    if useOneTime
        pred = pred(:,1,:);
        devVelTrunc = devVelTrunc(:,1,:);
    end
    
    %% displace the steadystate response to sine waves
    H = cell(length(h),1);
    for hInd = 1:length(h)
        imageSize = size(devImage);
        filterSize = size(h{hInd});
        pad = imageSize(2:3)-filterSize;
        h_pad = [h{hInd} zeros(filterSize(1),pad(2))];
        h_pad = [h_pad; zeros(pad(1),imageSize(3))];
        H{hInd} = fftshift(fft2(h_pad));
    end
    
    RH = real(H{1}.*conj(H{2}));
    
    % assume axis for now
    k = -1/10:1/360:1/10;
    k = k(2:end);
    w = -50:1/1:50;
    
    velToDisp = [100 200 300];
    velToPlot = [k'*velToDisp];
    
    %% calculate R2 across all examples and find the best R2 by scalling global pred
    globalY = devVelTrunc(:);
    globalPred = pred(:);
        
    R2 = 1-sum((globalY-globalPred).^2)./sum((globalY-mean(globalY)).^2);
    
    %% calculate R2 for each example
    R2_each = zeros(m,1);
    for mInd = 1:m
        thisPred = pred(mInd,:,:);

        thisVel = devVelTrunc(mInd,:,:);
        
        thisPred = thisPred(:);
        thisVel = thisVel(:);
        
        R2_each(mInd) = 1 - sum((thisPred-thisVel).^2)./sum((thisVel-mean(thisVel)).^2);
    end
    
    %% plot figs
    disp(['R2 using every data point = ' num2str(R2)]);
    disp(['median R2 across images = ' num2str(median(R2_each))]);
        
    MakeFigure;
    max_abs_h = max(abs([h{1}(:); h{2}(:)]));
    subplot(2,2,1);
    imagesc(rot90(h{1},2));
    caxis([-max_abs_h max_abs_h]);
    
    subplot(2,2,2);
    imagesc(rot90(h{2},2))
    caxis([-max_abs_h max_abs_h]);
    
    subplot(2,2,3);
    imagesc(k,w,RH);
    caxis([-max(abs(RH(:))) max(abs(RH(:)))]);
    hold on;
    plot(k,velToPlot,'k');
    hold off;
    set(gca,'yDir','normal');
    colormap(flipud(cbrewer('div','RdBu',100)));
    
    MakeFigure;
    plot(R2_each);
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
    
    arm2 = conv2(img,h{2},'valid')+b{1};
    arm1(arm1<0) = 0;
    
    pred = (arm1-arm2)*h{2}+b{2};
end

function pred = HrcModel(img,h,b)
    arm1 = conv2(img,h{1},'valid')+b{1};
    arm2 = conv2(img,h{2},'valid')+b{2};
    
    pred = conv2(arm1.*arm2,h{3},'valid')+b{3};
end