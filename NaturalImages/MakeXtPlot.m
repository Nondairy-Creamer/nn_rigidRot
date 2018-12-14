function xtPlot = MakeXtPlot(xyPlot,xPoints,yPoints,vel,sampleFreq,tMin,phi)
    % xyPlot is a matrix of natural scenes that is y by x by scenes
    % xPoints is a vector in deg
    % yPoints is a vector in deg
    % vel is a vector in deg/s
    % sample freq is in hz
    % tMin is the amount of time to rotate
    % phi is a vector of the offset of your visual inputs
    
    % shift the image over time at velocity vel to make an xt plot
    numSamples = size(xPoints,1);
    numScenes = size(xyPlot,3);

    tRes = 1/sampleFreq;

    xyRes = 360/size(xyPlot,2); 
    yPointsInd = ceil(yPoints/xyRes);

    % add one extra value so you can interpolate circularly
    xyPlot = [xyPlot xyPlot(:,1,:)];
    x = (0:xyRes:360)';
    
    numPhi = length(phi);
    numT = length(0:tRes:tMin-tRes);
    xtPlot = zeros(numT,numPhi,numScenes,numSamples);
    
    parfor ss = 1:numSamples
%         samTimeStart = tic;

        for sc = 1:numScenes
            for pp = 1:numPhi
                baseSample = (0:tRes:tMin-tRes)'*-vel(ss);
                
                xSam = round(mod(baseSample+xPoints(ss)+phi(pp),360));

                xtPlot(:,pp,sc,ss) = interp1(x,xyPlot(yPointsInd(ss),:,sc),xSam);
            end
        end
        
%         samTimeEnd = toc(samTimeStart);
%         
%         if mod(ss,round(numSamples/10))==0
%             disp([num2str(ss) '/' num2str(numSamples) ' xt plots made and took ' num2str(samTimeEnd) ' seconds per plot']);
%             disp([num2str(samTimeEnd*(numSamples-ss)) ' seconds remaining']);
%         end
    end
end