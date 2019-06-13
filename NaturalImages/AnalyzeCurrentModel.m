function AnalyzeCurrentModel
    %% set images and params folder locations
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters\';
    
    %% get the fit outputs. Default to the last one generated
    param_list = dir([filterFolder '*.mat']);
    params = load(fullfile(filterFolder,param_list(end).name));
%     params = load(fullfile(filterFolder,'2019-05-31_18-11-07.972329.mat'));
    params = params.param_dict;
    
    %% get the xt plot used to fit the list
    [~, xtPlotName] = fileparts(params{1}.data_set_path);
    xtPlotPath = fullfile(xtPlotFolder,xtPlotName);
    images = load(xtPlotPath);

    %% extract the weights and R2 value for the runs
    num_runs = length(params);
    r2 = zeros(size(params{1}.r2, 2), num_runs);
    val_r2 = zeros(size(params{1}.val_r2, 2), num_runs);
    % filter weights
    h = cell(size(params));
    % offset values
    b = cell(size(params));
    % weight for final averaging
    m = cell(size(params));
    
    for pp = 1:num_runs
        r2(:, pp) = params{pp}.r2;
        val_r2(:, pp) = params{pp}.val_r2;
        
        switch params{pp}.name
            case 'ln_model'
                h{pp} = params{pp}.weights{1};
                m{pp} = params{pp}.weights{end};
                b{pp} = params{pp}.biases{1}(:);
            case 'ln_model_flip'
                h{pp} = params{pp}.weights{1};
                sm{pp} = params{pp}.weights{end};
                b{pp} = params{pp}.biases{1}(:);
            case 'current_model'
                for ff = 1:params{pp}.num_filt
                    h{pp}(:,:,ff) = [params{pp}.weights{1}(:,:,:,ff) params{pp}.weights{2}(:,:,:,ff) params{pp}.weights{3}(:,:,:,ff)];
                end
                
                b{pp} = zeros(size(h{pp},3),1);
            case 'current_model_flip'
                for ff = 1:params{pp}.num_filt/2
                    h{pp}(:,:,ff) = [params{pp}.weights{1}(:,:,:,ff) params{pp}.weights{2}(:,:,:,ff) params{pp}.weights{3}(:,:,:,ff)];
                end
                
                b{pp} = zeros(size(h{pp},3),1);
        end
        
        h{pp} = rot90(h{pp},2);
    end
    
    
    MakeFigure;
    plot(val_r2);
    ylim([0, 1]);
    ConfAxis('labelX','epochs','labelY','R2');
    
    %% analyze filters
    filt = 1;
    
    h = h{filt};
    m = double(squish(params{filt}.weights{4}));
    t_sample_rate = double(params{filt}.sample_freq);
    x_step = double(params{filt}.phase_step);

%     if size(h,3) == 2
        h = cat(3, h, fliplr(h));
%         b = cat(1, b, b);
        m = cat(1, m, -m);        
%     end
    
    dev_set = permute(images.dev_in, [2 1 3]);
    
    if params{filt}.normalize_std
        for dd = 1:size(dev_set,3)
            scene = dev_set(:,:,dd);
            dev_set(:,:,dd) = scene/std(scene(:));
        end
    end
    
    cMax = max(abs(h(:)));
    MakeFigure;
    for pp = 1:size(h,3)
        subplot(2,2,pp);
        imagesc(h(:,:,pp));
        caxis([-cMax cMax]);
    end
    colormap(flipud(cbrewer('div','RdBu',100)));
    
    out = cell(size(h,3),1);

    % calculate output of the filters
    for h_ind = 1:size(h,3)
        for ff = 1:size(dev_set,3)
            out{h_ind}(:,:,ff) = conv2(dev_set(:,:,ff), h(:,:,h_ind), 'valid') + b(h_ind);
        end
        out{h_ind}(out{h_ind}<0) = 0;
        out{h_ind} = m(h_ind)*out{h_ind};
    end
    
    
    for aa = 1:length(out)
        for bb = 1:length(out)            
            rms_a = rms(reshape(out{aa}, [size(out{aa},1)*size(out{aa},2) 1 size(out{aa},3)]));
            rms_b = rms(reshape(out{bb}, [size(out{bb},1)*size(out{bb},2) 1 size(out{bb},3)]));
            
            co_act(aa,bb) = mean(mean(mean(out{aa}.*out{bb}./rms_a./rms_b)));
        end
    end
    
    disp(co_act)
    
    t = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
    
    corr_disp = co_act;
    
    mapLim = max(abs(h(:)));
    actLim = max(abs(corr_disp(:)));
    MakeFigure;
    for pp = 1:length(out)
        subplot(5,5,pp+1);
        imagesc(x, t, h(:,:,pp));
        caxis([-mapLim mapLim]);
        set(gca,'XAxisLocation','top');
        if pp==1
            xlabel('space (deg)');
            ylabel('time (ms)');
        end
        
        subplot(5,5,5*pp+1);
        imagesc(x, t, h(:,:,pp));
        caxis([-mapLim mapLim]);
        set(gca,'XAxisLocation','top');
    end
    
    subplot(5,5,[7:10 12:15 17:20 22:25]);
    imagesc(abs(corr_disp))
    caxis([-actLim actLim]);

    colormap(flipud(cbrewer('div','RdBu',100)));
    
    
    
    
    
    %% plot prediction vs answer
    
    dev_ans = repmat(permute(images.dev_out, [1 3 2]), [1 size(out{1}, 2) 1]);
    dev_ans = dev_ans(end-size(out{1},1)+1:end, :, :);
%     dev_ans(dev_ans<0) = 0;
    
    pred_half = out{1} + out{2};
    pred_full = out{1} + out{2} + out{3} + out{4};
    
    pred = pred_full;
    
    MakeFigure;
    scatter(dev_ans(:),pred(:));
    
    num = sum(sum((dev_ans-pred).^2));
    denom = sum(sum((dev_ans-mean(mean(dev_ans))).^2));
    
    num = sum((dev_ans(:)-pred(:)).^2);
    denom = sum((dev_ans(:)-mean(dev_ans(:))).^2);
    
    r2 = 1-mean(num./denom);
    disp(r2);
    
    
    
    
    keyboard;
    
    
    
    
    
    
    
    
    