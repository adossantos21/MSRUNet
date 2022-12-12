%clear all; close all; clc
%Download unet data
downloadFolder = tempdir;
url = 'https://home.cis.rit.edu/~cnspci/other/data/rit18_data.mat';
filename = fullfile(downloadFolder,'rit18_data.mat');

if ~exist(filename,'file')
    fprintf("Downloading Hamlin Beach data set (3 GB)... ");
    websave(filename,url);
    fprintf("Done.\n")
end
load rit18_data.mat
%Inspect training data, alter structure of training, validation, and test
%data to have the image channels in the third dimension
whos train_data val_data test_data
train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);
whos train_data val_data test_data

%Save the training and validation data as a MAT file and the training
%labels as a PNG file
save("train_data.mat","train_data");
save("val_data.mat","val_data")
imwrite(train_labels,"train_labels.png");
imwrite(val_labels,"val_labels.png");

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
imdsVal = imageDatastore("val_data.mat",FileExtensions=".mat",ReadFcn=@matRead6Channels);
% Create a pixelLabelDatastore to store the label patches containing the 18
% labeled regions
pixelLabelIds = 1:18;
pxds = pixelLabelDatastore("train_labels.png",classNames,pixelLabelIds);
pxdsVal = pixelLabelDatastore("val_labels.png",classNames,pixelLabelIds);

%Create a colormap for later segmentation
% cmap = HamlinBeachColormap;

%Create a randomPatchExtractionDatastore from the image and pixel label
%datastore. Each mini-batch contains 8 patches of size 256x256 pixels.
%One thousand mini-batches are extracted at each iteration of the epoch.
dsTrain = randomPatchExtractionDatastore(imds,pxds,[256,256],PatchesPerImage=8000);
dsVal = randomPatchExtractionDatastore(imdsVal,pxdsVal,[256,256],PatchesPerImage=8000);
%Create U-Net Network Layers
inputTileSize = [256,256,6];
lgraph = lgraph;
disp(lgraph.Layers)

%Select Training Options
options = trainingOptions("sgdm", ...
    'ValidationData',dsVal,...
    'Momentum', 0.9,...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.3, ...
    'InitialLearnRate', 0.05, ...
    'L2Regularization', 0.0001, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 8, ...
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
    'GradientThreshold',0.05);

%Train the network, change doTraining boolean to load example Unet this
%network was based on (it is slightly different)
doTraining = true; 
if doTraining
    ResNetUnet_6channels = trainNetwork(dsTrain,lgraph,options);
    modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
    dataDir = 'C:\Users\Alessandro\Desktop\IR_1_ResNet18_DeepLabV3_SGDM_8\CityScapes RGB Network\ResNet18_Unet';
    save(fullfile(dataDir,"multispectralResNetUnet_6channels_-"+modelDateTime+".mat"),"ResNetUnet_6channels");

else 
    fprintf("Model has not been trained");
end
%%
%IoU = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,Metrics="iou");
%weightedIoU = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,Metrics="weighted-iou");
evalMetrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth,"Metrics","all");
datasetMetrics = evalMetrics.DataSetMetrics
classMetrics = evalMetrics.ClassMetrics
confMatrix = evalMetrics.ConfusionMatrix
Results = 'C:\Users\Alessandro\Desktop\IR_1_ResNet18_DeepLabV3_SGDM_8\CityScapes RGB Network\ResNet18_Unet\6ChannelHamlinBeach';
%%
T = array2table(datasetMetrics);
TString = evalc('disp(T)');
% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');
% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');
% Launch a figure that is the size of the user's screen, then display the
% table as a figure
figure('units','normalized','outerposition',[0 0 1 1]);
fig = annotation(gcf,'Textbox','String',TString,'Interpreter','Tex',...
    'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);
saveas(fig, fullfile(Results,'dataSetMetrics.png'));
writetable(T,fullfile(Results,'datasetMetrics.csv'));
%%
T = array2table(classMetrics);
TString = evalc('disp(T)');
% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');
% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');
% Launch a figure that is the size of the user's screen, then display the
% table as a figure
figure('units','normalized','outerposition',[0 0 1 1]);
fig = annotation(gcf,'Textbox','String',TString,'Interpreter','Tex',...
    'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);
saveas(fig, fullfile(Results,'classMetrics.png'));
writetable(T,fullfile(Results,'classMetrics.csv'));
%%
T = array2table(confMatrix);
TString = evalc('disp(T)');
% Use TeX Markup for bold formatting and underscores.
TString = strrep(TString,'<strong>','\bf');
TString = strrep(TString,'</strong>','\rm');
TString = strrep(TString,'_','\_');
% Get a fixed-width font.
FixedWidth = get(0,'FixedWidthFontName');
% Launch a figure that is the size of the user's screen, then display the
% table as a figure
figure('units','normalized','outerposition',[0 0 1 1]);
fig = annotation(gcf,'Textbox','String',TString,'Interpreter','Tex',...
    'FontName',FixedWidth,'Units','Normalized','Position',[0 0 1 1]);
saveas(fig, fullfile(Results,'confMatrix.png'));
writetable(T,fullfile(Results,'confMatrix.csv'));