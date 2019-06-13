function TestKerasModel
    %% load weights and images
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters';
    xtPlotName = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf500_tt1_nt2_hl0-2_vs100_df0-05.mat';
    xtPlotPath = fullfile(xtPlotFolder,xtPlotName);
    images = load(xtPlotPath);

    % boolean to decide whether to use only a single time point of response
    % from each xt plot
    useOneTime = false;
    
    %% choose filter type to use
    filterType = 'gradientDescentLnFlip';
    
    filtSizeIn = [21 11];
    
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
            fileName = '2019-05-31_18-11-07.972329.mat';
            filterPath = fullfile(filterFolder,fileName);

            w = load(filterPath);
            w = w.param_array{1}.weight_dict;
            wNames = fieldnames(w);
            h = cell(0,1);
            b = cell(0,1);
            
            hInd = 0;
            bInd = 0;
            
            for nn = 1:length(wNames)
                theseParams = double(w.(wNames{nn}));
                if size(theseParams,3) == 1
                    theseParams = squeeze(theseParams);
                end
                
                if isequal(wNames{nn}(1:6),'weight')
                    hInd = hInd + 1;
                    for ff = 1:size(theseParams,3)
                        h{hInd}{ff} = theseParams(:,:,ff);
                    end
                else
                    bInd = bInd + 1;
                    
                    for ff = 1:size(theseParams,2)
                        b{bInd}{ff} = theseParams(ff);
                    end
                end
            end
            
%             h{1} = cellfun(@(x)rot90(x,2),h{1},'UniformOutput',false);
%             h{2} = cellfun(@(x)rot90(x,2),h{2},'UniformOutput',false);
        
        case 'gradientDescentLnFlip'
            fileName = '2019-05-31_18-11-07.972329.mat';
            filterPath = fullfile(filterFolder,fileName);

            w = load(filterPath);
            w = w.param_array{1}.weight_dict;
            wNames = fieldnames(w);
            h = cell(0,1);
            b = cell(0,1);
            
            hInd = 0;
            bInd = 0;
            
            for nn = 1:length(wNames)
                theseParams = double(w.(wNames{nn}));
                if size(theseParams,3) == 1
                    theseParams = squeeze(theseParams);
                end
                
                if isequal(wNames{nn}(1:6),'weight')
                    hInd = hInd + 1;
                    for ff = 1:size(theseParams,3)
                        h{hInd}{ff} = theseParams(:,:,ff);
                    end
                else
                    bInd = bInd + 1;
                    
                    for ff = 1:size(theseParams,2)
                        b{bInd}{ff} = theseParams(ff);
                    end
                end
            end
            
            h{1} = [h{1} cellfun(@fliplr,h{1},'UniformOutput',false)];
            h{2} = [h{2} cellfun(@(x)-x,h{2},'UniformOutput',false)];
            
            b{1} = [b{1} b{1}];
            
%             h{1} = cellfun(@(x)rot90(x,2),h{1},'UniformOutput',false);
%             h{2} = cellfun(@(x)rot90(x,2),h{2},'UniformOutput',false);

        case 'newRandom'
            h{1} = randn(filtSizeIn)/10;
            h{2} = randn(filtSizeIn)/10;
            h{3} = 1;
            save('randomH','h');
        case 'savedRandom'
            filterPath = fullfile(filterFolder,'randomH.mat');

            w = load(filterPath);
            
            h = w.h;
            
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
    end
    
    
    filtSize = size(h{1}{1});
    
    padX = filtSize(2)-1;
    padT = filtSize(1)-1;
    
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
    sizePred = [sizeVelIn(1) sizeVelIn(2)-padT sizeImage(3)-size(h{1}{1},2)+1-size(h{2}{1},2)+1];
    
    devVelTrunc = repmat(devVel(1:sizePred(1),1:sizePred(2)),[1 1 sizePred(3)]);
    
    %% get predictions
    m = size(devImage,1);
    
    pred = zeros(size(devVelTrunc));
        
    for mInd = 1:m
        natScene = squeeze(devImage(mInd,:,:));
        natScene_norm = natScene/std(natScene(:));
        [pred(mInd,:,:), respCorr(:,:,mInd)] = LnModel(natScene_norm,h,b);
%         pred(mInd,:,:) = HrcModel(natScene_norm,h,b);
    end
    
    disp(median(respCorr,3));
    
    if useOneTime
        pred = pred(:,1,:);
        devVelTrunc = devVelTrunc(:,1,:);
    end
    
    %% calculate R2 across all examples and find the best R2 by scalling global pred
    globalY = devVelTrunc(:);
    globalPred = pred(:);
    
    optMult = lsqlin(globalPred,globalY);
    globalPred = globalPred*optMult;
    pred = pred*optMult;
    
    
    R2 = 1-sum((globalY-globalPred).^2)./sum((globalY-mean(globalY)).^2);
    
    %% calculate R2 for each example
    R2_each = zeros(m,1);
    for mInd = 1:m
        thisPred = pred(mInd,:,:);

        thisVel = devVelTrunc(mInd,:,:);
        
        thisPred = thisPred(:);
        thisVel = thisVel(:);
        
        R2_each(mInd) = 1 - sum((thisPred-thisVel).^2)./sum((thisVel-0*mean(thisVel)).^2);
    end
    
    %% plot figs
    disp(['R2 using every data point = ' num2str(R2)]);
    disp(['median R2 across images = ' num2str(median(R2_each))]);
    
    plotY = round(sqrt(length(h{1})));
    plotX = ceil(length(h{1})/plotY);
    
    MakeFigure;
    max_abs_h = max(cellfun(@(x)max(max(abs(x))),h{1}));
    for hh = 1:length(h{1})
        subplot(plotY,plotX,hh);
        imagesc(rot90(h{1}{hh},2));
        caxis([-max_abs_h max_abs_h]);
    end
    colormap(flipud(cbrewer('div','RdBu',100)));

    MakeFigure;
    plot(R2_each);
    xlabel('example');
    ylabel('r2');
    
    MakeFigure;
    hold on
    for mm = 1:size(devVelTrunc,1)
        scatter(Columnize(devVelTrunc(mm,:,:)),Columnize(pred(mm,:,:)));
    end
    hold off;
    xlabel('scene velocity');
    ylabel('model prediction');
 
end


function [pred, respCorr] = LnModel(img,h,b)
    for ff = 1:length(h{1})
        layer1{ff} = conv2(img,h{1}{ff},'valid')+b{1}{ff};
        layer1{ff}(layer1{ff}<0) = 0;
        
        layer2{ff} = conv2(layer1{ff},h{2}{ff},'valid');
    end
    
    pred = layer2{1};
    for aa = 2:length(layer2)
        pred = pred + layer2{aa};
    end
    
    for ii = 1:length(layer1)
        for jj = 1:length(layer1)
            corrMat = corrcoef(layer1{ii}(:),layer1{jj}(:));
            respCorr(ii,jj) = corrMat(2,1);
            respCorr(ii,jj) = mean(layer1{ii}(:).*layer1{jj}(:))/std(layer1{ii}(:))/std(layer1{jj}(:));
            respCorr(ii,jj) = mean(layer1{ii}(:).*layer1{jj}(:))/sqrt(mean(layer1{ii}(:).^2))/sqrt(mean(layer1{jj}(:).^2));
        end
    end
    
    pred = pred + b{2}{1};
end


function pred = HrcModel(img,h,b)
    arm1 = conv2(img,h{1},'valid')+b{1};
    arm2 = conv2(img,h{2},'valid')+b{2};
    
    pred = conv2(arm1.*arm2,h{3},'valid')+b{3};
end