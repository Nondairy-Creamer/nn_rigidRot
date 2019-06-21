function model_output = OptimalModelPredict(data_set,param_path)
    load(param_path);
    
    model_output = ConductanceModel(data_set,h,b,m);
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
                channels(:,:,ii) = conv2(data_set(:,ii:end-3+ii,ff), h(:,ii,h_ind), 'valid') + b(h_ind, ii);
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