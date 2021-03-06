function [models_out, model_index] = GetModel(models, r2_style, varargin)
    var_names = cell(0,1);
    var_values = cell(0,1);
    for ii = 1:2:length(varargin)
        var_names{(ii+1)/2,1} = varargin{ii};
        var_values{(ii+1)/2,1} = varargin{ii+1};
    end
    
    models_out = cell(0,1);
    index = zeros(0,1);
    model_ind = 1;
    
    for pp = 1:length(models)
        keep = true;
        
        for vv = 1:length(var_names)
            if ischar(models{pp}.(var_names{vv}))
                if ~strcmp(models{pp}.(var_names{vv}),var_values{vv})
                    keep = false;
                end
            else
                if models{pp}.(var_names{vv}) ~= var_values{vv}
                    keep = false;
                end
            end
        end
        
        if keep
            models_out{model_ind,1} = models{pp};
            index(model_ind,1) = pp;
            model_ind = model_ind + 1;
        end
    end
    
    if isempty(models_out)
        error('no models have all the requested parameters');
    end
    
    switch r2_style
        case 'all'
            
        case 'max'
            max_model = models_out{1};
            model_index = 1;
            
            for ii = 2:length(models_out)
                if models_out{ii}.val_r2(end) > max_model.val_r2(end)
                    max_model = models_out{ii};
                    model_index = index(ii);
                end
            end
            
            models_out = {max_model};
    end
end