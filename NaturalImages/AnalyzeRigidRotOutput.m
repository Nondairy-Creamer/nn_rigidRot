function AnalyzeRigidRotOutput
    % this function will load in models that were fit to predict velocity
    % from rotating natural scenes.

    %% set images and params folder locations
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters\';
    
    save_model = false;
    
    %% get the fit outputs. Default to the last one generated
    param_list = dir([filterFolder '*.mat']);
%     params = load(fullfile(filterFolder,param_list(end).name));
    params = load(fullfile(filterFolder,'saved\ln_lnln_cond_vNoise_vNorm.mat'));
    params = params.param_dict;
    
    %% get the model you want to analyze
%     [chosen_model, model_ind] = GetModel(params,'max','model_name','ln_model_flip');
%     [chosen_model, model_ind] = GetModel(params,'max','model_name','lnln_model_flip');
%     [chosen_model, model_ind] = GetModel(params,'max','model_name','conductance_model_flip');
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
            % h are filters
            % b are offsets
            % m are scalar weights
            h = squeeze(chosen_model.weights{1});
            b1 = zeros(size(h,2), size(h,3));
            m1 = ones(size(h,2), size(h,3));
            
            m2 = chosen_model.weights{end}(:);
            b2 = chosen_model.biases{1}(:)';

            h = rot90(h,2);

            model_structure = @ln_model;

        case 'ln_model_flip'
            h = squeeze(chosen_model.weights{1});
            b1 = zeros(size(h,2), size(h,3));
            m1 = ones(size(h,2), size(h,3));
            
            m2 = chosen_model.weights{end}(:);
            b2 = chosen_model.biases{1}(:);

            h = rot90(h,2);

            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @ln_model;

        case 'conductance_model'
            for ff = 1:chosen_model.num_filt
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb};
            end
            
            m2 = chosen_model.weights{end}(:);

            model_structure = @conductance_model;

        case 'conductance_model_flip'
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            if chosen_model.fit_reversal
                m1 = repmat(chosen_model.weights{4}(:), [1, size(h,3)]);

                b2 = chosen_model.weights{5}(:);
            else
                m1 = [-30 60 -30];
                
                b2(4, :) = chosen_model.weights{4}(:);
            end
            
            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb}';
            end
                
            m2 = chosen_model.weights{end}(:);
                
            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @conductance_model;
            
        case 'lnln_model_flip'
            for ff = 1:size(chosen_model.weights{1},4)
                h(:,:,ff) = [chosen_model.weights{1}(:,:,:,ff) chosen_model.weights{2}(:,:,:,ff) chosen_model.weights{3}(:,:,:,ff)];
            end

            h = flipud(h);

            m1 = repmat(chosen_model.weights{4}(:), [1, size(h,3)]);
            
            for bb = 1:3
                b1(bb, :) = chosen_model.biases{bb}';
            end
            
            b2 = chosen_model.weights{5}(:);

            m2 = chosen_model.weights{end}(:);
                
            [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2);

            model_structure = @lnln_model;

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
    [model_output, component_output] = model_structure(data_set,h,b1,m1,b2,m2);

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
    
    b_max = max([b1(:); b2(:); 0]);
    b_min = min([b1(:); b2(:); 0]);
    
    mult_max = max(abs([m1(:); m2(:)]));
    
    MakeFigure;
    for pp = 1:size(h,3)
        subplot(6,num_filt,[pp num_filt+pp]);
        imagesc(x, t_label, h(:,:,pp));
        caxis([-c_max c_max]);
%         set(gca,'XAxisLocation','top');
        ConfAxis();
        
        if pp==1
%             xlabel(['space (' char(186) ')']);
            ylabel('time (ms)');
            title({'fraction zeros = ', fraction_zero(pp)});
        else
            xlabel(' ');
            title(fraction_zero(pp));
        end
        
        subplot(6,num_filt,2*num_filt+pp);
        bar(b1(:,pp));
        ylim([b_min b_max]);
        ConfAxis()
        if pp==1
            ylabel('filter offset');
        end
        
        subplot(6,num_filt,3*num_filt+pp);
        imagesc(m1(:,pp)');
        caxis([-mult_max mult_max]);
        axis off;
        ConfAxis();
        if pp==1
            ylabel('weights');
        end
        
        subplot(6,num_filt,4*num_filt+pp);
        bar(b2(pp));
        ylim([b_min b_max]);
        ConfAxis()
        if pp==1
            ylabel('filter offset');
        end
        
        subplot(6,num_filt,5*num_filt+pp);
        imagesc(m2(pp));
        caxis([-mult_max mult_max]);
        axis off;
        ConfAxis();
        if pp==1
            ylabel('weights');
        end
        
        display_name = chosen_model.model_name;
        display_name = strrep(display_name,'_','\_');
        sgtitle([display_name ', R2 = ' num2str(chosen_model.val_r2(end))]);
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


function [model_output, component_output] = ln_model(data_set,h,b1,m1,b2,m2)
    component_output = cell(size(h,3),1);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            component_output{h_ind}(:,:,ff) = conv2(data_set(:,:,ff), h(:,:,h_ind), 'valid') + b2(h_ind);
        end
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = m2(h_ind)*component_output{h_ind};
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
end


function [model_output, component_output] = lnln_model(data_set,h,b1,m1,b2,m2)
    % calculates the output of the LNLN model and all its
    % intermediate steps.

    component_output = cell(size(h,3),1);
    channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,3);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            
            for ii = 1:3
                channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b1(ii, h_ind);
            end
            
            % rectify the channels
            channels(channels<0) = 0;
            
            component_output{h_ind}(:,:,ff) = m1(1,h_ind)*channels(:,:,1) + m1(2,h_ind)*channels(:,:,2) + m1(3,h_ind)*channels(:,:,3);
        end
        
        component_output{h_ind} = component_output{h_ind} + b2(h_ind);
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = m2(h_ind)*component_output{h_ind};
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
end


function [model_output, component_output] = conductance_model(data_set,h,b1,m1,b2,m2)
    component_output = cell(size(h,3),1);
    g_leak = 1;
    channels = zeros(size(data_set,1)-size(h,1)+1,size(data_set,2)-2,3);

    for h_ind = 1:size(h,3)
        % convolve each filter in the model with each image in the dev set
        for ff = 1:size(data_set,3)
            
            for ii = 1:3
                channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b1(ii, h_ind);
            end
            
            % rectify the channels
            channels(channels<0) = 0;
            
            numerator = m1(1,h_ind)*channels(:,:,1) + m1(2,h_ind)*channels(:,:,2) + m1(3,h_ind)*channels(:,:,3);
            denomenator = g_leak + sum(channels,3);
            component_output{h_ind}(:,:,ff) = numerator./denomenator;
        end
        
        component_output{h_ind} = component_output{h_ind} + b2(h_ind);
        
        % rectify
        component_output{h_ind}(component_output{h_ind}<0) = 0;
        
        % multiply before summing
        component_output{h_ind} = m2(h_ind)*component_output{h_ind};
    end
    
    model_output = component_output{1};
    
    for cc = 2:length(component_output)
        model_output = model_output + component_output{cc};
    end
end




function [h,b1,m1,b2,m2] = FlipFilters(h,b1,m1,b2,m2)
    h = cat(3, h, fliplr(h));
    
    b1 = cat(2, b1, flipud(b1));
    m1 = cat(2, m1, flipud(m1));
    
    b2 = cat(1, b2, b2);
    m2 = cat(1, m2, -m2);
end
    
    
    
    
    
    