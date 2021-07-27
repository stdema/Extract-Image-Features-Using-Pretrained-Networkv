unzip('MerchData.zip');
imds = imageDatastore('MerchData', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages, 16); %랜덤으로 이미지 16장 불러오기 위해 인덱스 저장
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain, idx(i));
    imshow(I)
end

net = resnet18
inputSize = net.Layers(1).InputSize;
analyzeNetwork(net)

augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain);
augimdsTest = augmentedImageDatastore(inputSize(1:2), imdsTest);

layer = 'pool5';
featuresTrain = activations(net, augimdsTrain, layer, 'OutputAs', 'rows');
featuresTest = activations(net, augimdsTest, layer, 'OutputAs', 'rows');

whos featuresTrain %whos 작업공간의 변수 크기, 유형 나열

YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;

classifier = fitcecoc(featuresTrain, YTrain);

YPred = predict(classifier, featuresTest);

idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest, idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

accuracy = mean(YPred == YTest) %예측과 실제가 같으면 1

layer = 'res3b_relu';
featuresTrain = activations(net, augimdsTrain, layer);
featuresTest = activations(net, augimdsTest, layer);
whos featuresTrain

featuresTrain = squeeze(mean(featuresTrain, [1 2]))';
featuresTest = squeeze(mean(featuresTest, [1 2]))';
whos featuresTrain

classifier = fitcecoc(featuresTrain, YTrain);
YPred = predict(classifier, featuresTest);
accuracy = mean(YPred == YTest)