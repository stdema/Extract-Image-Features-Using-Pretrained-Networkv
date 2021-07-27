# 사전 훈련된 신경망을 사용하여 영상 특징 추출하기
이 예제에서는 사전 훈련된 컨벌루션 신경망에서 학습된 영상 특징을 추출한 다음 추출한 특징을 사용하여 영상 분류기를 훈련시키는 방법을 보여줍니다. 특징 추출은 사전 훈련된 심층 신경망의 강력한 표현 기능을 가장 쉽고 빠르게 사용하는 방법입니다. 예를 들어, 추출된 특징에 대해 fitcecoc(Statistics and Machine Learning Toolbox™)를 사용하여 서포트 벡터 머신(SVM)을 훈련시킬 수 있습니다. 특징 추출은 데이터를 한 번만 통과하면 되기 때문에 신경망 훈련을 가속할 GPU가 없을 때 시도할 수 있는 좋은 시작점이 됩니다.
# 데이터 불러오기
샘플 영상의 압축을 풀고 영상 데이터저장소로서 불러옵니다. imageDatastore는 폴더 이름을 기준으로 영상에 자동으로 레이블을 지정하고 데이터를 ImageDatastore 객체로 저장합니다. 영상 데이터저장소를 사용하면 메모리에 담을 수 없는 데이터를 포함하여 다량의 영상 데이터를 저장할 수 있습니다. 데이터를 훈련 데이터 70%와 테스트 데이터 30%로 분할합니다.
```c
unzip('MerchData.zip');
imds = imageDatastore('MerchData','IncludeSubfolders',true,'LabelSource','foldernames');
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized'); %데이터를 훈련 데이터 70%(55개)와 테스트 데이터 30%(20개)로 분할
```
이 매우 작은 데이터 세트에는 이제 55개의 훈련 영상과 20개의 검증 영상이 있습니다. 샘플 영상 몇 개를 표시합니다.
```c
numTrainImages = numel(imdsTrain.Labels); %55개의 훈련 영상
idx = randperm(numTrainImages,16); %55개의 훈련 영상 중 16개 랜덤으로 선택
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
```
![16](https://user-images.githubusercontent.com/86040099/127119636-6382c878-bd03-4ed9-b834-96fb3e75cf44.jpg)
# 사전 훈련된 신경망 불러오기
사전 훈련된 ResNet-18 신경망을 불러옵니다. Deep Learning Toolbox Model for ResNet-18 Network 지원 패키지가 설치되어 있지 않으면 이를 다운로드할 수 있는 링크가 제공됩니다. 1백만 개가 넘는 영상에 대해 훈련된 ResNet-18은 영상을 키보드, 마우스, 연필, 각종 동물 등 1,000가지 사물 범주로 분류할 수 있습니다. 그 결과 이 모델은 다양한 영상을 대표하는 다양한 특징을 학습했습니다.
```c
net = resnet18
```
![image](https://user-images.githubusercontent.com/86040099/127120004-efbc02be-baa2-4643-8f4f-05d763018431.png)
```c
inputSize = net.Layers(1).InputSize; %딥러닝 신경망 분석기 사진 아래 부가 설명
analyzeNetwork(net)
```
![image](https://user-images.githubusercontent.com/86040099/127120078-4e4f520a-2af4-49a5-b0c9-6bc1abcc81ea.png)
딥러닝 신경망 분석기의 분석결과에서 첫 번째 계층이 'data'임을 알 수 있다. 'data'의 활성화 칸에 224x224x3라고 기재되어 있으며 이것은 픽셀의 크기가 224x224임을 나타내며 후미의 3은 RGB를 나타낸다.
# 영상 특징 추출하기
이 신경망의 입력 영상은 크기가 224x224x3이어야 하는데 영상 데이터저장소의 영상은 이와 크기가 다릅니다. 신경망에 입력하기 전에 훈련 영상과 테스트 영상의 크기를 자동으로 조정하려면 증대 영상 데이터저장소를 만들고 원하는 영상 크기를 지정한 다음 이러한 데이터저장소를 activations에 대한 입력 인수로 사용하십시오.

*activations이란 훈련된 딥러닝 신경망을 사용하여 특징을 추출할 수 있도록 딥러닝 신경망 계층 활성화 계산을 말한다.*
```c
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain); %imdaTrain의 영상을 사용하여 증대 영상데이터저장소를 만든다. 출력크기는 inputSize(1:2)속성, 즉 224x224로 설정한다.
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest); %imdaTest의 영상을 사용하여 증대 영상데이터저장소를 만든다. 출력크기는 inputSize(1:2)속성, 즉 224x224로 설정한다.
```
신경망은 입력 영상에 대한 계층 표현을 생성합니다. 보다 심층의 계층에는 앞쪽 계층의 하위 수준 특징을 사용하여 생성한 상위 수준의 특징이 포함됩니다. 훈련 영상과 테스트 영상의 특징 표현을 가져오려면 신경망의 끝부분에 있는 전역 풀링 계층 'pool5', 에 대해 activations를 사용하십시오. 전역 풀링 계층은 모든 공간 위치에 대해 입력 특징을 풀링하여 총 512개의 특징을 제공합니다.
```c
layer = 'pool5'; % 전역 풀링 계층으로 1x1x512의 크기를 가진다.
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows'); %'OutputAs','rows'는 출력을 행의 형태로 나타내겠음을 의미한다.
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows'); %'OutputAs','rows'는 출력을 행의 형태로 나타내겠음을 의미한다.

whos featuresTrain
```
![image](https://user-images.githubusercontent.com/86040099/127121837-8c3d3909-8224-49dc-b390-570ce4388cf5.png)

activations를 통해 size가 55x512가 되었음을 알 수 있다.

훈련 데이터와 테스트 데이터로부터 클래스 레이블을 추출합니다.
```c
YTrain = imdsTrain.Labels;
YTest = imdsTest.Labels;
```
![image](https://user-images.githubusercontent.com/86040099/127122537-f7b71447-3a8c-4775-a178-1b6ec7ac6238.png)

위 그림의 Labels에 해당하는 55개 20개의 카테고리를 각각 YTrain과 YTest에 할당한다.

# 영상 분류기 피팅하기
훈련 영상으로부터 추출한 특징을 예측 변수로 사용하고 fitcecoc(Statistics and Machine Learning Toolbox)를 사용하여 다중클래스 서포트 벡터 머신(SVM)을 피팅합니다.
```c
classifier = fitcecoc(featuresTrain,YTrain);
```
# 테스트 영상 분류하기
훈련된 SVM 모델과 테스트 영상으로부터 추출한 특징을 사용하여 테스트 영상을 분류합니다.
```c
YPred = predict(classifier,featuresTest);
```
4개의 샘플 테스트 영상을 예측된 레이블과 함께 표시합니다.
```c
idx = [1 5 10 15];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end
```
![4](https://user-images.githubusercontent.com/86040099/127123159-f9f8fefe-f8d8-45d7-a625-934e5b176c8a.jpg)

테스트 세트에 대한 분류 정확도를 계산합니다. 정확도는 신경망이 올바르게 예측하는 레이블의 비율입니다.
```c
accuracy = mean(YPred == YTest)
```
accuracy = 1
