function ConfAxis(varargin)
    tickX = [];
    tickY = [];
    tickLabelX = [];
    tickLabelY = [];
    fTitle = [];
    figLeg = cell(0,1);
    labelX = [];
    labelY = [];
    rotateLabels = 0;
    axisFontSize = 12;
    labelFontSize = 14;
    titleFontSize = 18;
    useLatex = true;
    
    for ii = 1:2:length(varargin)
        eval([varargin{ii} '= varargin{' num2str(ii+1) '};']);
    end
    
    if useLatex
        interpreterStr = 'latex';
    else
        interpreterStr = 'none';
    end
    
    if ~isempty(tickX)
        set(gca,'XTick',tickX)
    end
    
    if ~isempty(tickY)
        set(gca,'YTick',tickY)
    end
    
    % if length(XTickLabel) > length(XTick) matlab will ignore the extra
    % entries in XTickLabel
    if ~isempty(tickLabelX)
        
%         if ~iscell(tickLabelX)
%             tickLabelX = num2cell(tickLabelX);
%             tickLabelX = cellfun(@num2str,tickLabelX,'UniformOutput',0);
%         end
        
        set(gca,'XTickLabel',tickLabelX)
        
        if iscell(tickLabelX)
            maxLength = 0;
            for ii = 1:length(tickLabelX)
                thisLength = length(tickLabelX{ii});
                if thisLength > maxLength
                    maxLength = thisLength;
                end
            end

            if maxLength > 20
                InterleaveTickLabels(gca);
            else
%                 RotateTickLabels(gca,45);
            end
        end
        
        if rotateLabels
            RotateTickLabels(gca,45);
        end
        
        
        if isempty(tickX)
            set(gca,'XTick',1:length(tickLabelX))
        end
    end

    if ~isempty(tickLabelY)
        set(gca,'YTickLabel',tickLabelY);
    end
    
    if ~isempty(fTitle)
        title(fTitle,'fontSize',titleFontSize);
    end
    
    if ~isempty(labelX)
        xlabel(labelX);
    end
    
    if ~isempty(labelY)
        ylabel(labelY);
    end
    
    if ~isempty(figLeg)
        if ~iscell(figLeg)
            figLeg = {figLeg};
        end
        legend(figLeg);
        legend('boxoff');
    end
    
    
    set(gca,'FontSize',axisFontSize,'box','off','FontName','Arial');
    set(gca,'XColor',[0 0 0],'YColor',[0 0 0],'ZColor',[0 0 0]);
    set(findall(gca, 'Type', 'Line'),'LineWidth',2,'MarkerSize',8);
    set(findall(gca, 'Type', 'ErrorBar'),'LineWidth',2,'MarkerSize',8);
%     set(gca,'TickLabelInterpreter',interpreterStr);
    ax = gca;
    ax.YLabel.FontSize = labelFontSize;
    ax.XLabel.FontSize = labelFontSize;
    ax.LineWidth = 2;
%     yAxisLimits = ylim;
%     set(gca,'LooseInset',get(gca,'TightInset'));
%     axis('tight');
%     ylim(yAxisLimits);
end