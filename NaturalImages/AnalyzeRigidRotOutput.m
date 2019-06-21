function AnalyzeRigidRotOutput
    %% set images and params folder locations
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters\';
    
    save_model = false;
    
    %% get the fit outputs. Default to the last one generated
    param_list = dir([filterFolder '*.mat']);
    params = load(fullfile(filterFolder,param_list(end).name));
%     params = load(fullfile(filterFolder,'saved\cond_4filt_vNoise_vNorm_big.mat'));
%     params = load(fullfile(filterFolder,'saved\ln_4filt_vNoise_vNorm_big.mat'));
    params = params.param_dict;
    
    %% get the model you want to analyze
%     [chosen_model, model_ind] = GetModel(params,'max','normalize_std',false,'noise_std',0.1);
    [chosen_model, model_ind] = GetModel(params,'max');
    chosen_model = chosen_model{1};
    
    %% get the xt plot used to fit the list
    [~, xtPlotName] = fileparts(params{1}.data_set_path);
    xtPlotPath = fullfile(xtPlotFolder,xtPlotName);
    images = load(xtPlotPath);
    
    % switch dev to test set if you want
    data_set = images.dev_in;
    data_ans = images.dev_out;

    %% decide how to handle models fit without negative velocities
    if chosen_model.no_opponency
        chosen_model.model_name = [chosen_model.model_name '_flip'];
        chosen_model.no_opponency = false;
    end
        
    %% extract R2 value for the runs
    num_runs = length(params);
    r2 = zeros(size(params{1}.r2, 2), num_runs);
    val_r2 = zeros(size(params{1}.val_r2, 2), num_runs);
    
    for pp = 1:num_runs
        r2(:, pp) = params{pp}.r2;
        val_r2(:, pp) = params{pp}.val_r2;
    end
    
    %% extract the weights from the runs
    % get the parameters out based on what model was run
    switch chosen_model.model_name
        case 'ln_model'
            h = squeeze(chosen_model.weights{1});
            m = chosen_model.weights{end}(:);
            b = chosen_model.biases{1}(:)';

            h = rot90(h,2);

            model_structure = @LnModel;

        case 'ln_model_flip'
            h = squeeze(chosen_model.weights{1});
            m = chosen_model.weights{end}(:);
            b = chosen_model.biases{1}(:)';

            h = rot90(h,2);

            [h,b,m] = FlipFilters(h,b,m);

            model_structure = @LnModel;

        case 'conductance_model'
            % load in backwards so that when we rotate 180 degrees its
            % in the correct orientation
            for ff = 1:chosen_model.num_filt
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m = chosen_model.weights{end}(:);

            for bb = 1:3
                b(bb, :) = chosen_model.biases{bb};
            end

            model_structure = @ConductanceModel;

        case 'conductance_model_flip'
            % load in backwards so that when we rotate 180 degrees its
            % in the correct orientation
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m = chosen_model.weights{end}(:);

            for bb = 1:3
                b(bb, :) = chosen_model.biases{bb}';
            end
            
            b(4, :) = chosen_model.weights{4}(:)';
            
            [h,b,m] = FlipFilters(h,b,m);

            model_structure = @ConductanceModel;
            
        case 'LNLN_flip'
            % load in backwards so that when we rotate 180 degrees its
            % in the correct orientation
            % this was written so hacky with m2 i'm ashamed.
            filt_in = size(chosen_model.weights{1},4);
            for ff = 1:filt_in
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m = chosen_model.weights{end}(:);

            for bb = 1:3
                b(bb, :) = chosen_model.biases{bb}';
            end
            
            b(4, :) = repmat(chosen_model.biases{4}(:), [2 1])';
            m2_in = chosen_model.weights{4}(:);
            
            for ff = 1:filt_in
                m2{ff} = m2_in;
            end
            
            for ff = 1:filt_in
                m2{filt_in+ff} = flipud(m2_in);
            end
            
            [h,b,m] = FlipFilters(h,b,m);

            model_structure = @(d,x,y,z)LNLN_model(d,x,y,z,m2);

        otherwise
            error('counldnt detect model type');

    end
    
    if save_model
        save('standalone_model\model_params.mat','h','b','m');
    end
    
    % plot R2
    MakeFigure;
    plot(val_r2);
    ylim([0, 1]);
    ConfAxis('labelX','epochs','labelY','R2');
    
    %% analyze filters
    % example filter to show
    t_sample_rate = double(chosen_model.sample_freq);
    x_step = double(chosen_model.phase_step);
    
    % set up data_set
    data_set = permute(data_set, [2 1 3]);
    
    if chosen_model.normalize_std
        data_set = data_set./std(data_set,[],[1 2]);
    end

    data_set = data_set + randn(size(data_set))*double(chosen_model.noise_std);
    
    
    % get model output
    [model_output, component_output] = model_structure(data_set,h,b,m);

    fraction_zero = zeros(size(component_output));
    for cc = 1:length(component_output)
        fraction_zero(cc) = mean(component_output{cc}(:)==0);
    end
    
    num_filt = length(component_output);
    
    % set up data_ans
    data_ans = repmat(permute(data_ans, [1 3 2]), [1 size(component_output{1}, 2) 1]);
    data_ans = data_ans(end-size(component_output{1},1)+1:end, :, :);
    
    if chosen_model.no_opponency
        data_ans(data_ans<0) = 0;
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
    
    %% plot example model
    t = (0:size(h,1)-1)/t_sample_rate*1000;
    t_label = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
    
    c_max = max(abs(h(:)));
    plot_max = max(abs(h(:)));
    
    b_max = max([b(:); 0]);
    b_min = min([b(:); 0]);
    
    mult_max = max(abs(m(:)));
    
    MakeFigure;
    for pp = 1:size(h,3)
        subplot(4,num_filt,pp);
        plot(t,h(:,:,pp));
        hold on;
        PlotConstLine(0);
        hold off;
        ylim([-plot_max plot_max]);
        ConfAxis();
        
        if pp == 1
            title({'fraction zeros = ', fraction_zero(pp)});
            xlabel('time (ms)');
        else
            title(fraction_zero(pp));
        end
        
        subplot(4,num_filt,num_filt+pp);
        imagesc(x, t_label, h(:,:,pp));
        caxis([-c_max c_max]);
%         set(gca,'XAxisLocation','top');
        ConfAxis();
        if pp==1
            xlabel(['space (' char(186) ')']);
            ylabel('time (ms)');
        end
        
        
        subplot(4,num_filt,2*num_filt+pp);
        bar(b(:,pp));
        ylim([b_min b_max]);
        ConfAxis()
        if pp==1
            ylabel('filter offset');
        end
        
        subplot(4,num_filt,3*num_filt+pp);
        imagesc(m(pp));
        caxis([-mult_max mult_max]);
        axis off;
        ConfAxis();
        
        sgtitle(['R2 = ' num2str(chosen_model.val_r2(end))]);
    end
    colormap(flipud(cbrewer('div','RdBu',100)));
   
    
    % plot filters vs their coactivation values    
    mapLim = max(abs(h(:)));
    actLim = max(abs(co_act(:)));
    MakeFigure;
    for pp = 1:length(component_output)
        % plot the filters along the top
        subplot(num_filt+1,num_filt+1,pp+1);
        imagesc(x, t_label, h(:,:,pp));
        caxis([-mapLim mapLim]);
        set(gca,'XAxisLocation','top');
        
        if pp==1
            xlabel(['space (' char(186) ')']);
            ylabel('time (ms)');
        end
        
        % plot the filters along the side
        subplot(num_filt+1,num_filt+1,(num_filt+1)*pp+1);
        imagesc(x, t_label, h(:,:,pp));
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
    
%     scatter prediction vs answer
    MakeFigure;
    scatter(data_ans(:),model_output(:));
    
    num = sum((data_ans(:)-model_output(:)).^2);
    denom = sum((data_ans(:)-mean(data_ans(:))).^2);
    
    xlabel(['true velocity (' char(186) '/s)']);
    ylabel(['predicted velocity (' char(186) '/s)']);
    axis equal;
    ConfAxis();
    
    r2 = 1-mean(num./denom);
    title({['matlab R2 = ' num2str(r2)]; ['python R2 = ' num2str(chosen_model.val_r2(end))]});
end


function [model_output, component_output] = LnModel(data_set,h,b,m)
    component_output = cell(size(h,3),1);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            component_output{h_ind}(:,:,ff) = conv2(data_set(:,:,ff), h(:,:,h_ind), 'valid') + b(h_ind);
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

    
function [model_output, component_output] = ConductanceModel(data_set,h,b,m)
    % calculates the output of the conductance model and all its
    % intermediate steps. h, b, m should be matricies from a single model
    % not cell arrays of multiple models

    component_output = cell(size(h,3),1);
    v_exc = 60;
    v_inh = -30;
    g_leak = 1;
    channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,3);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            
            for ii = 1:3
                channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b(ii, h_ind);
            end
            
            % rectify the channels
            channels(channels<0) = 0;
            
            numerator = v_inh*channels(:,:,1) + v_exc*channels(:,:,2) + v_inh*channels(:,:,3);
            denomenator = g_leak + sum(channels,3);
            component_output{h_ind}(:,:,ff) = numerator./denomenator;
        end
        
        component_output{h_ind} = component_output{h_ind} + b(4,h_ind);
        
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


function [model_output, component_output] = LNLN_model(data_set,h,b,m,m2)
    % calculates the output of the LNLN model and all its
    % intermediate steps.

    component_output = cell(size(h,3),1);
    channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,3);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            
            for ii = 1:3
                channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b(ii, h_ind);
            end
            
            % rectify the channels
            channels(channels<0) = 0;
            
            component_output{h_ind}(:,:,ff) = m2{h_ind}(1)*channels(:,:,1) + m2{h_ind}(2)*channels(:,:,2) + m2{h_ind}(3)*channels(:,:,3);
        end
        
        component_output{h_ind} = component_output{h_ind} + b(4,h_ind);
        
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
    b = cat(2, b, b);
    if size(b,1)>1
        b(1:3,3:end) = flipud(b(1:3,3:end));
    end
    m = cat(1, m, -m);
end
    
    
    
    
    
    