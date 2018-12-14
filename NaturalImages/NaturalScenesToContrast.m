function NaturalScenesToContrast
    sysConfig = GetSystemConfiguration;

    %% load images
    folderPath = fileparts(mfilename('fullpath'));
    dataPath = fullfile(folderPath,'natImageCombined.mat');
    data = load(dataPath);
    
    scenes = data.xyPlot;
    xRes = 360/size(scenes,2);
    
    x = (0:xRes:360-xRes)';
    y = (0:xRes:xRes*size(scenes,1)-xRes)';
    
    
    % individual photo receptors
    filtStd = 5/(2*sqrt(2*log(2)));

    xFilt = ifftshift(normpdf(x,180,filtStd));
    yFilt = ifftshift(normpdf(y,(y(end)+xRes)/2,filtStd));
    
    xyFiltMat = yFilt*xFilt';
    
    xyFiltTensor = repmat(xyFiltMat,[1 1 size(scenes,3)]);
    
    % for mean estimation
    filtStdContrast = 20;

    xFiltContrast = ifftshift(normpdf(x,180,filtStdContrast));
    yFiltContrast = ifftshift(normpdf(y,(y(end)+xRes)/2,filtStdContrast));
    
    xyFiltMatContrast = yFiltContrast*xFiltContrast';
    
    xyFiltTensorContrast = repmat(xyFiltMatContrast,[1 1 size(scenes,3)]);
    
    % perform the convolution
    fftScenes = fft2(scenes);
    fftFilt = fft2(xyFiltTensor);
    fftFiltContrast = fft2(xyFiltTensorContrast);
    
    filteredScenes = real(ifft2(fftScenes.*fftFilt));
    filteredScenesContrast = real(ifft2(fftScenes.*fftFiltContrast));
    
    xyPlot = (filteredScenes-filteredScenesContrast)./filteredScenesContrast;
    
    MakeFigure;
    subplot(1,2,1);
    imagesc(scenes(:,:,105))
    subplot(1,2,2);
    imagesc(xyPlot(:,:,105));
    colormap(gray);
    
    save(fullfile(folderPath,'combinedFiltered2D.mat'),'xyPlot');
end