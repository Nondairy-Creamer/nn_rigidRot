function AnalyzeRigidRotOutput
    %% load weights and images
    xtPlotFolder = 'G:\My Drive\data_sets\nn_RigidRot\natural_images\xt';
    filterFolder = 'G:\My Drive\data_sets\nn_RigidRot\saved_parameters';
    xtPlotName = 'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt2_hl0-2_vs100_df0-05.mat';
    xtPlotPath = fullfile(xtPlotFolder,xtPlotName);
    images = load(xtPlotPath);
    
    param_list = dir(filterFolder);
    params = load(fullfile(filterFolder,param_list(end-1).name));
    params = params.param_array;
    
    R2 = zeros(size(params{1}.param_dict.R2,2),size(params,1),size(params,2),size(params,3),size(params,4),size(params,5),size(params,6),size(params,7));
    val_R2 = zeros(size(params{1}.param_dict.val_R2,2),size(params,1),size(params,2),size(params,3),size(params,4),size(params,5),size(params,6),size(params,7));
        
    weights = cell(size(params));
    time = zeros(size(params));
    
    plot_fig = true;
    
    for i1 = 1:size(params,1)
        for i2 = 1:size(params,2)
            for i3 = 1:size(params,3)
                for i4 = 1:size(params,4)
                    for i5 = 1:size(params,5)
                        for i6 = 1:size(params,6)
                            for i7 = 1:size(params,7)
                                R2(:,i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.R2;
                                val_R2(:,i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.val_R2;
                                weights{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.weight3(1:25,:,:,:);
                                biases{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.biases3;
                                endMult{i1,i2,i3,i4,i5,i6,i7} = params{i1,i2,i3,i4,i5,i6,i7}.weight_dict.weight5;
                                time(i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.time;
                                sample_rate(i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.sample_freq;
                                phase_step(i1,i2,i3,i4,i5,i6,i7) = params{i1,i2,i3,i4,i5,i6,i7}.param_dict.phase_step;
                                
                                if plot_fig
                                    MakeFigure;
                                    numFilt = size(weights{i1,i2,i3,i4,i5,i6,i7},4);
                                    for pp = 1:numFilt
                                        colorMax = max(abs(weights{i1,i2,i3,i4,i5,i6,i7}(:)));
                                        subplot(3,numFilt,pp);
                                        imagesc(weights{i1,i2,i3,i4,i5,i6,i7}(:,:,pp));
                                        caxis([-colorMax colorMax]);
                                        colormap(flipud(cbrewer('div','RdBu',100)));
                                    end

                                    title(num2str(val_R2(end,i1,i2,i3,i4,i5,i6,i7)));

                                    subplot(3,numFilt,numFilt+1:numFilt*2);
                                    plot(biases{i1,i2,i3,i4,i5,i6,i7});
                                    hold on;
                                    PlotConstLine(0,1);
                                    hold off;
                                    ConfAxis;

                                    for pp = 1:numFilt
                                        colorMax = max(abs(endMult{i1,i2,i3,i4,i5,i6,i7}(:)));
                                        subplot(3,numFilt,pp+numFilt*2);
                                        imagesc(endMult{i1,i2,i3,i4,i5,i6,i7}(:,:,pp));
                                        caxis([-colorMax colorMax]);
                                        colormap(flipud(cbrewer('div','RdBu',100)));
                                    end
                                    
%                                     plot_fig = false;
                                end
                            end
                        end 
                    end
                end
            end
        end
    end
     
    MakeFigure;
    plot(val_R2(:,:,1,1,1,1,1,1,1,1,1,1,1));
    ConfAxis('labelX','epochs','labelY','R2');
    
    %% analyze filters
    h = double(squeeze(weights{1,1,1,1,1,1,1}));
    h = cat(3, h, fliplr(h));
    b = double(squeeze(biases{1,1,1,1,1,1,1}));
    b = cat(2, b, b);
    m = double(squeeze(endMult{1,1,1,1,1,1,1}));
    m = cat(2, m, -m);
    t_sample_rate = double(squeeze(sample_rate(1,1,1,1,1,1,1)));
    x_step = double(squeeze(phase_step(1,1,1,1,1,1,1)));

    % calculate output of the filters
    for h_ind = 1:size(h,3)
        out{h_ind} = imfilter(images.dev_in, h(:,:,h_ind)) + b(h_ind);
        out{h_ind}(out{h_ind}<0) = 0;
        out{h_ind} = m(h_ind)*out{h_ind};
    end
    
    
    for aa = 1:length(out)
        for bb = 1:length(out)
            co_act(aa,bb) = mean(mean(mean(out{aa}.*out{bb})));
            
            std_a = std(reshape(out{aa}, [size(out{aa},1)*size(out{aa},2) 1 size(out{aa},3)]));
            std_b = std(reshape(out{bb}, [size(out{bb},1)*size(out{bb},2) 1 size(out{bb},3)]));
            
            h_co(aa,bb) = mean(mean(mean(out{aa}.*out{bb}./std_a./std_b)));
            out_a_sub = out{aa} - mean(mean(out{aa}));
            out_b_sub = out{bb} - mean(mean(out{bb}));
            h_corr(aa,bb) = mean(mean(mean(out_a_sub.*out_b_sub./std_a./std_b)));
        end
    end
    
    disp(co_act)
    disp(h_co)
    disp(h_corr)
    
    t = linspace(0,(size(h,1)-1)/t_sample_rate*1000,4);
    x = linspace(0,(size(h,2)-1)*x_step,size(h,2));
    
    mapLim = max(abs(h(:)));
    actLim = max(abs(h_co(:)));
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
    imagesc(abs(h_co))
    caxis([-actLim actLim]);

    colormap(flipud(cbrewer('div','RdBu',100)));
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    