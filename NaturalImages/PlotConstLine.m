function PlotConstLine(value,constDim,referenceLineColor)
        if nargin < 2 || isempty(constDim)
            constDim = 1;
        end
        
        if nargin < 3 || isempty(constDim)
            referenceLineColor = [0.5 0.5 0.5];
        end
        
        limitsX = xlim';
        limitsY = ylim';
        
%         totalDistX = limitsX(2)-limitsX(1);
%         totalDistY = limitsY(2)-limitsY(1);
%         
%         fudgeFrac = 0.1;
%         fudgeAmountX = fudgeFrac*totalDistX;
%         fudgeAmountY = fudgeFrac*totalDistY;
%         
%         limitsX = limitsX + [fudgeAmountX -fudgeAmountX]';
%         limitsY = limitsY + [fudgeAmountY -fudgeAmountY]';
        
        switch constDim
            case 1 % y value doesn't change
                h=plot(limitsX,[value; value],'--','Color',referenceLineColor);
                h.HandleVisibility = 'off';
            case 2 % x value doesn't change
                h=plot([value; value],limitsY,'--','Color',referenceLineColor);
                h.HandleVisibility = 'off';
        end
end