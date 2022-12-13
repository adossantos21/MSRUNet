%Download unet data
downloadFolder = tempdir;
url = 'https://home.cis.rit.edu/~cnspci/other/data/rit18_data.mat';
filename = fullfile(downloadFolder,'rit18_data.mat');

if ~exist(filename,'file')
    fprintf("Downloading Hamlin Beach data set (3 GB)... ");
    websave(filename,url);
    fprintf("Done.\n")
end

%Inspect training data, alter structure of training, validation, and test
%data to have the image channels in the third dimension
whos train_data val_data test_data
train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);
whos train_data val_data test_data

%Save the training data as a MAT file and the training labels as a PNG file
save("train_data.mat","train_data");
imwrite(train_labels,"train_labels.png");

%Display classes then assign names
disp(classes)

classNames = [
    "RoadMarkings"
    "Tree"
    "Building"
    "Vehicle"
    "Person"
    "LifeguardChair"
    "PicnicTable"
    "BlackWoodPanel"
    "WhiteWoodPanel"
    "OrangeLandingPad"
    "Buoy"
    "Rocks"
    "LowLevelVegetation"
    "Grass_Lawn"
    "Sand_Beach"
    "Water_Lake"
    "Water_Pond"
    "Asphalt" 
    ];

%Create Random Patch Extraction Datastore for Training
% Use a random patch extraction datastore to feed the training data to the network. 
% This datastore extracts multiple corresponding random patches from an image 
% datastore and pixel label datastore that contain ground truth images and pixel 
% label data. Patching is a common technique to prevent running out of memory 
% for large images and to effectively increase the amount of available training 
% data.
% 
% Begin by storing the training images from "train_data.mat" in an  
% imageDatastore. Because the MAT file format is a nonstandard image format, 
% you must use a MAT file reader to enable reading the image data. You can use 
% the helper MAT file reader, matRead6Channels, that extracts the first six 
% channels from the training data and omits the last channel containing the mask. 
% This function is attached to the example as a supporting file.

imds = imageDatastore("train_data.mat",FileExtensions=".mat",ReadFcn=@matRead6Channels);

% Create a pixelLabelDatastore to store the label patches containing the 18
% labeled regions
pixelLabelIds = 1:18;
pxds = pixelLabelDatastore("train_labels.png",classNames,pixelLabelIds);

%Create a colormap for later segmentation
cmap = HamlinBeachColormap;

%Augment data
data1 = read(imds);
I = data1{1};
data2 = read(pxds);
C = data2{1};
imdsTrain = transform(imds,@ImageOnlyWarpAugmenter);
pxdsTrain = transform(pxds,@LabelOnlyWarpAugmenter);


%Create a randomPatchExtractionDatastore from the image and pixel label
%datastore. Each mini-batch contains 8 patches of size 256x256 pixels.
%One thousand mini-batches are extracted at each iteration of the epoch.
dsTrain = randomPatchExtractionDatastore(imdsTrain,pxdsTrain,[256,256],PatchesPerImage=4000);

%Create U-Net Network Layers
inputTileSize = [256,256,6];
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers)

%Select Training Options
options = trainingOptions("adam", ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.3, ...
    'Epsilon', 1e-8, ...
    'InitialLearnRate', 0.05, ...
    'L2Regularization', 0.0001, ...
    'ValidationData', dsVal, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 4, ...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 20, ...
    'Plots', 'training-progress', ...
    'ValidationPatience', 4, ...
    'OutputNetwork', 'best-validation-loss', ...
    'ExecutionEnvironment','auto', ...
    'CheckpointPath','C:\Users\Alessandro\Desktop\IR_1_ResNet18_DeepLabV3_SGDM_8\CityScapes RGB Network\Checkpoints', ...
    'CheckpointFrequency',2, ...
    'CheckpointFrequencyUnit','epoch', ...
    'GradientThresholdMethod','l2norm', ...
    GradientThreshold=0.05);

%Train the network
doTraining = true; 
if doTraining
    net = trainNetwork(dsTrain,lgraph,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    save(fullfile(dataDir,"multispectralUnet-"+modelDateTime+".mat"),"net");

else 
    trainedUnet_url = "https://www.mathworks.com/supportfiles/vision/data/multispectralUnet.mat";
    downloadTrainedNetwork(trainedUnet_url,dataDir);
    load(fullfile(dataDir,"multispectralUnet.mat"));
end
%%
%Segment image to see results. Multiply by the segmented image by the mask
%channel of the validation data to extract only the valid portion of the
%segmentation
predictPatchSize = [1024 1024];
segmentedImage = segmentMultispectralImage(val_data,net,predictPatchSize);
segmentedImage = uint8(val_data(:,:,7)~=0) .* segmentedImage;

figure
imshow(segmentedImage,[])
title("Segmented Image")

%Apply a median filter to remove salt and pepper noise from the
%segmentation
segmentedImage = medfilt2(segmentedImage,[7,7]);
imshow(segmentedImage,[]);
title("Segmented Image with Noise Removed")

% Overlay the segmented image on the histogram-equalized RGB validation image.
B = labeloverlay(histeq(val_data(:,:,[3 2 1])),segmentedImage,Transparency=0.8,Colormap=cmap);

figure
imshow(B)
title("Labeled Segmented Image")
colorbar(TickLabels=cellstr(classNames),Ticks=ticks,TickLength=0,TickLabelInterpreter="none");
colormap(cmap)

% Save the segmented image and ground truth labels as PNG files. They will
% be used to calculate the Intersection over Union (IoU)
imwrite(segmentedImage,"results.png");
imwrite(val_labels,"gtruth.png");
IoU = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,Metrics="iou");
weightedIoU = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,Metrics="weighted-iou");


%Partition Datasets (split image and label datasets into training,
%validation, and test datasets weighted 60/20/20 respectively.
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = partitionCityKITTIDataset(imds, pxds);

%Create the Network
%Specify image input size, the number of classes, and create the DeepLab
%V3+ Network, respectively
imageSize = [320 640 3];
numClasses = numel(classes);
lgraph = lgraph;

%Replace layers and create new layer graph
%newlgraph = replaceLayer(lgraph,'imageinput',imageInputLayer([320 640 4],'Name','input'));
%newlgraph = replaceLayer(newlgraph,'conv',convolution2dLayer([7 7],64,'stride',[2 2],'padding',[3 3 3 3],'Name','conv1'));

%Print statistics for class weights and pixel frequency for each class
tbl = countEachLabel(pxdsTrain)
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq

%Replace the last layer in the neural network with a pixel classification
%layer to allow for semantic segmentation
% pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
% newlgraph = replaceLayer(lgraph, "RegressionLayer_output1", pxLayer);

%Define validation data
dsVal = combine(imdsVal, pxdsVal);



%Augment Data
dsTrain = combine(imdsTrain, pxdsTrain);
data = read(dsTrain);
I = data{1};
C = data{2};
dsTrain = transform(dsTrain,@ImageWarpAugmenter);
data = readall(dsTrain);

%Start stopwatch for training time
tic

%Start Training
[Resnet50_deeplabv3plus_320x640x4_trained_cityscapes] = trainNetwork(dsTrain,lgraph,options);

%Stop stopwatch for training time
toc

% %Save the newly trained network
save Resnet50_deeplabv3plus_320x640x4_trained_cityscapes


%exportONNXNetwork(OnnxSODACOCOIR_1,'OnnxSODACOCOIR_1');


%%
%Test the Network on our data from UND
whiteImage = 127.5 * ones(1024, 2048, 3, 'uint8');
imshow(whiteImage);

% Read and segment recently collected images to test network performance
TestingImages = 'G:\Infrared Test Images for Neural Nets';
LabelResults = 'G:\IR Neural Networks\IR_1_ResNet18_DeepLabV3_SGDM_8\Label Results 2';
OverlayResults = 'G:\IR Neural Networks\IR_1_ResNet18_DeepLabV3_SGDM_8\Overlay Results 2';
filePattern = fullfile(TestingImages, '*.png'); % Change to whatever pattern you need.
theFiles = dir(filePattern);

tic
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
    % Now do whatever you want with this file name,
    % such as reading it in as an image array with imread()
    I = imread(fullFileName);
    C = semanticseg(I, IR_1_ResNet18_DeepLabV3_SGDM_8_100422);
    label = labeloverlay(whiteImage,C,'Colormap',cmap);
    imwrite(label,fullfile(LabelResults,baseFileName));
    overlay = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.7);
    imwrite(overlay,fullfile(OverlayResults,baseFileName));
    pixelLabelColorbar(cmap, classes);
end
toc