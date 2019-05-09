function AnalyzeRigidRotOutput
    %% load weights and images
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters';
    xtPlotName = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-05.mat';
    xtPlotPath = fullfile(xtPlotFolder,xtPlotName);
    images = load(xtPlotPath);
    
    param_list = dir(filterFolder);
    params = load(fullfile(filterFolder,param_list(end).name));
    params = params.param_array;
    
    R2 = zeros(size(params{1}.param_dict.R2,2),size(params,1),size(params,2),size(params,3),size(params,4),size(params,5),size(params,6),size(params,7));
    val_R2 = zeros(size(params{1}.param_dict.val_R2,2),size(params,1),size(params,2),size(params,3),size(params,4),size(params,5),size(params,6),size(params,7));
    
    size(params)
    
    weights = cell(size(params));
    time = zeros(size(params));
    
    for i1 = 1:size(params,1)
        for i2 = 1:size(params,2)
            for i3 = 1:size(params,3)
                for i4 = 1:size(params,4)
                    for i5 = 1:size(params,5)
                        for i6 = 1:size(params,6)
                            for i7 = 1:size(params,7)
                                R2(:,i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.R2;
                                val_R2(:,i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.val_R2;
%                                 weights{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.weight2;
%                                 biases{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.biases2;
%                                 endMult{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.weight3;
                                time(i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.time;
                                
                                
%                                 MakeFigure;
%                                 numFilt = size(weights{i1,i2,i3,i4,i5,i6,i7},4);
%                                 for pp = 1:numFilt
%                                     colorMax = max(abs(weights{i1,i2,i3,i4,i5,i6,i7}(:)));
%                                     subplot(3,numFilt,pp);
%                                     imagesc(weights{i1,i2,i3,i4,i5,i6,i7}(:,:,pp));
%                                     caxis([-colorMax colorMax]);
%                                     colormap(flipud(cbrewer('div','RdBu',100)));
%                                 end
%                                 
%                                 title(num2str(val_R2(end,i1,i2,i3,i4,i5,i6,i7)));
%                                 
%                                 subplot(3,numFilt,numFilt+1:numFilt*2);
%                                 plot(biases{i1,i2,i3,i4,i5,i6,i7});
%                                 hold on;
%                                 PlotConstLine(0,1);
%                                 hold off;
%                                 ConfAxis;
%                                 
%                                 for pp = 1:numFilt
%                                     colorMax = max(abs(endMult{i1,i2,i3,i4,i5,i6,i7}(:)));
%                                     subplot(3,numFilt,pp+numFilt*2);
%                                     imagesc(endMult{i1,i2,i3,i4,i5,i6,i7}(:,:,pp));
%                                     caxis([-colorMax colorMax]);
%                                     colormap(flipud(cbrewer('div','RdBu',100)));
%                                 end
                            end
                        end 
                    end
                end
            end
        end
    end
    
    MakeFigure;
    plot(val_R2(:,:,1,1,1,1,1,1,1,1,1,1,1))
    ConfAxis('labelX','epochs','labelY','R2')