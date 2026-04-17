# Reliable and Trustworthy AI - Assignment 2

## 프로젝트 설명
이 프로젝트는 CIFAR-10 데이터셋에서 학습한 두 개의 ResNet50 모델을 비교하여, 같은 입력에 대해 서로 다른 예측을 내리는 사례를 찾기 위한 실험이다.  
과제의 핵심 목표는 differential testing 관점에서 모델 간 disagreement를 확인하고, 그 결과를 시각화하는 것이다.

현재 구현된 내용은 다음과 같다.

- CIFAR-10용 ResNet50 모델 2개 준비
- 두 모델의 테스트셋 성능 평가
- 두 모델의 예측이 서로 다른 샘플 개수 계산
- disagreement 사례 5개를 이미지로 저장

## 파일 구성
- `train_resnet50_cifar10.py` : 첫 번째 ResNet50 모델 학습 코드
- `train_resnet50_cifar10_model2.py` : 두 번째 ResNet50 모델 학습 코드
- `test.py` : 두 모델을 불러와 성능 평가, disagreement 계산, 시각화 저장
- `requirements.txt` : 실행에 필요한 Python 패키지 목록
- `models/` : 학습된 모델 파일 저장 폴더
- `results/disagreements/` : disagreement 이미지 저장 폴더

## 실행 환경
- Python 3.6
- TensorFlow 1.15.0
- Keras 2.2.4

## 실행 방법
1. conda 환경 활성화  
`conda activate dx36`

2. 테스트 실행  
`python test.py`

## 실행 결과
`test.py`를 실행하면 아래 정보가 출력된다.

- model 1의 test loss / test accuracy
- model 2의 test loss / test accuracy
- 두 모델의 disagreement 개수
- 각 모델의 정답 개수

또한 disagreement가 발생한 샘플 중 앞의 5개를 아래 폴더에 저장한다.

`results/disagreements/`

## 비고
- 두 모델은 서로 다른 조건으로 학습하여 예측 차이가 발생하도록 구성
- 저장된 이미지 제목에는 정답 라벨과 두 모델의 예측 라벨이 함께 표시
- DeepXplore 원본 코드는 legacy 환경에서 확인했고, MNIST 예제로 기본 동작을 검증