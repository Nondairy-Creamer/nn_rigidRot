function AnalyzeRigidRotOutput
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
        
        % get the parameters out based on what model was run
        switch params{pp}.model_name
            case 'ln_model'
                h{pp} = squeeze(params{pp}.weights{1});
                m{pp} = params{pp}.weights{end}(:);
                b{pp} = params{pp}.biases{1}(:);
                    
                model_structure = @LnModel;
                
            case 'ln_model_flip'
                h{pp} = squeeze(params{pp}.weights{1});
                m{pp} = params{pp}.weights{end}(:);
                b{pp} = params{pp}.biases{1}(:);
                
                [h{pp},b{pp},m{pp}] = FlipFilters(h{pp},b{pp},m{pp});
                
                model_structure = @LnModel;

            case 'conductance_model'
                % load in backwards so that when we rotate 180 degrees its
                % in the correct orientation
                for ff = 1:params{pp}.num_filt
                    h{pp}(:,:,ff) = [params{pp}.weights{3}(:,:,:,ff) params{pp}.weights{2}(:,:,:,ff) params{pp}.weights{1}(:,:,:,ff)];
                end
                
                m{pp} = params{pp}.weights{end};
                
                for bb = 1:length(params{pp}.biases)-1
                    b{pp}(:, bb) = params{pp}.biases{bb}';
                end
                
                model_structure = @ConductanceModel;

            case 'conductance_model_flip'
                % load in backwards so that when we rotate 180 degrees its
                % in the correct orientation
                for ff = 1:params{pp}.num_filt/2
                    h{pp}(:,:,ff) = [params{pp}.weights{3}(:,:,:,ff) params{pp}.weights{2}(:,:,:,ff) params{pp}.weights{1}(:,:,:,ff)];
                end
                
                m{pp} = params{pp}.weights{end}(:);
                
                for bb = 1:length(params{pp}.biases)-1
                    b{pp}(:, bb) = params{pp}.biases{bb}';
                end
                
                [h{pp},b{pp},m{pp}] = FlipFilters(h{pp},b{pp},m{pp});

                model_structure = @ConductanceModel;
                
            otherwise
                error('counldnt detect model type');

        end
        
        h{pp} = rot90(h{pp},2);
    end
    
    % plot R2
    MakeFigure;
    plot(val_r2);
    ylim([0, 1]);
    ConfAxis('labelX','epochs','labelY','R2');
    
    %% analyze filters
    % example filter to show
    [chosen_model, filt_ind] = GetModel(params,'max');
    chosen_model = chosen_model{1};
    t_sample_rate = double(chosen_model.sample_freq);
    x_step = double(chosen_model.phase_step);
    
    % set up dev_set
    dev_set = permute(images.dev_in, [2 1 3]);
    
    if chosen_model.normalize_std
        for dd = 1:size(dev_set,3)
            scene = dev_set(:,:,dd);
            dev_set(:,:,dd) = scene/std(scene(:));
        end
    end

    % get model output
    [model_output, component_output] = model_structure(h{filt_ind},b{filt_ind},m{filt_ind},dev_set);

    num_filt = length(component_output);
    
    % set up dev_ans
    dev_ans = repmat(permute(images.dev_out, [1 3 2]), [1 size(component_output{1}, 2) 1]);
    dev_ans = dev_ans(end-size(component_output{1},1)+1:end, :, :);
    
    if chosen_model.no_opponency
        dev_ans(dev_ans<0) = 0;
    end
    
    % calculate coactivation of the filters
    co_act = zeros(num_filt);
    for aa = 1:num_filt
        for bb = 1:num_filt
            rms_a = rms(component_output{aa}, [1 2]);
            rms_b = rms(component_output{bb}, [1 2]);
            
            co_act(aa,bb) = mean(component_output{aa}.*component_output{bb}./rms_a./rms_b, [1, 2, 3]);
        end
    end
    
    %% plot example filter
    t = linspace(0,(size(h{filt_ind},1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h{filt_ind},2)-1)*x_step,size(h{filt_ind},2));
    
    cMax = max(abs(h{filt_ind}(:)));
    MakeFigure;
    for pp = 1:size(h{filt_ind},3)
        subplot(2,2,pp);
        imagesc(x, t, h{filt_ind}(:,:,pp));
        caxis([-cMax cMax]);
        
        if pp==1
            xlabel('space (deg)');
            ylabel('time (ms)');
        end
    end
    colormap(flipud(cbrewer('div','RdBu',100)));
   
    
    % plot filters vs their coactivation values    
    mapLim = max(abs(h{filt_ind}(:)));
    actLim = max(abs(co_act(:)));
    MakeFigure;
    for pp = 1:length(component_output)
        % plot the filters along the top
        subplot(num_filt+1,num_filt+1,pp+1);
        imagesc(x, t, h{filt_ind}(:,:,pp));
        caxis([-mapLim mapLim]);
        set(gca,'XAxisLocation','top');
        
        if pp==1
            xlabel('space (deg)');
            ylabel('time (ms)');
        end
        
        % plot the filters along the side
        subplot(num_filt+1,num_filt+1,(num_filt+1)*pp+1);
        imagesc(x, t, h{filt_ind}(:,:,pp));
        caxis([-mapLim mapLim]);
        set(gca,'XAxisLocation','top');
    end
    
    % find the indicies of the center
    [x_ind, y_ind] = meshgrid(2:num_filt+1, 2:num_filt+1);
    linear_ind = sub2ind([num_filt+1 num_filt+1], x_ind(:), y_ind(:));
    subplot(num_filt+1,num_filt+1,linear_ind);
    % plot the co activations
    imagesc(abs(co_act))
    caxis([-actLim actLim]);

    colormap(flipud(cbrewer('div','RdBu',100)));
    
    % scatter prediction vs answer
    MakeFigure;
    scatter(dev_ans(:),model_output(:));
    
    num = sum((dev_ans(:)-model_output(:)).^2);
    denom = sum((dev_ans(:)-mean(dev_ans(:))).^2);
    
    ConfAxis();
    
    r2 = 1-mean(num./denom);
    disp(r2);
end


function [model_output, component_output] = LnModel(h,b,m,dev_set)
    component_output = cell(size(h,3),1);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(dev_set,3)
            component_output{h_ind}(:,:,ff) = conv2(dev_set(:,:,ff), h(:,:,h_ind), 'valid') + b(h_ind);
        end
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = m(h_ind)*component_output{h_ind};
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
end

    
function [model_output, component_output] = ConductanceModel(h,b,m,dev_set)
    % calculates the output of the conductance model and all its
    % intermediate steps. h, b, m should be matricies from a single model
    % not cell arrays of multiple models

    component_output = cell(size(h,3),1);
    v_exc = 60;
    v_inh = -30;
    g_leak = 1;
    channels = zeros(size(dev_set,1)-size(h,1)+1,size(dev_set,2)-2,3);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(dev_set,3)
            
            for ii = 1:3
                channels(:,:,ii) = conv2(dev_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b(h_ind, ii);
            end
            
            % rectify the channels
            channels(channels<0) = 0;
            
            numerator = v_inh*channels(:,:,1) + v_exc*channels(:,:,2) + v_inh*channels(:,:,3);
            denomenator = g_leak + sum(channels,3);
            component_output{h_ind}(:,:,ff) = numerator./denomenator;
        end
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = m(h_ind)*component_output{h_ind};
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
end
    
    
function [h,b,m] = FlipFilters(h,b,m)
    h = cat(3, h, fliplr(h));
    b = cat(1, b, b);
    m = cat(1, m, -m);
end
    
    
    
    
    
    