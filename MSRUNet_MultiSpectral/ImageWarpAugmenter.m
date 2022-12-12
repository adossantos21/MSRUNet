function out = ImageWarpAugmenter(data)
% Unpack original data.
I = data{1};
C = data{2};

% Apply random color jitter.
% I = jitterColorHSV(I,"Brightness",0.3,"Contrast",0.4,"Saturation",0.2);

% Define random affine transform.
tform = randomAffine2d("Scale",[0.8 1.5],"XReflection",true,"YReflection",true,'Rotation',[-30 30], 'XTranslation',[-10 10], 'YTranslation',[-10 10]);
rout = affineOutputView(size(I),tform);

% Transform image and bounding box labels.
augmentedImage = imwarp(I,tform,"OutputView",rout);
augmentedLabel = imwarp(C,tform,"OutputView",rout);

% Return augmented data.
out = {augmentedImage,augmentedLabel};
end