
# Hand Bone Image Segmentation 대회 
- 2024.11.13 - 2024.11.28 
- Hand Bone x-ray 객체가 담긴 이미지에 대한 Segmentation task
- Naver Connect & Upstage 대회 

## Authors

-  [이준학](https://github.com/danlee0113)  [김민환](https://github.com/alsghks1066)  [김준원](https://github.com/KimJunWon98)  [전인석](https://github.com/inDseok)  [신석준](https://github.com/SeokjunShin)



## 대회 소개
![image](https://github.com/user-attachments/assets/fe33f559-b68f-4db3-9aa9-228652154972)

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적이다. 

Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있다. 

이번 프로젝트에서는 Hand Bone Segmentation을 진행하며, 손가락, 손등, 팔 뼈로 구성된 29개의 class에 대한 예측을 하는 것이 주요 task이다. 


## 평가 방법

Test set의 Dice coefficient로 평가

Semantic Segmentation에서 사용되는 대표적인 성능 측정 방법
![image](https://github.com/user-attachments/assets/4870dd87-34a0-492e-8032-43436783800f)



## 리더보드
![image](https://github.com/user-attachments/assets/458b3b61-a1ed-4171-adf0-b51b38bb1cb0)
![image](https://github.com/user-attachments/assets/5fd45d4a-1594-4a2f-bc90-78170688adb9)


## Dataset
- Train: 800, Label O  
- Test: 288, Label X  

- 이미지 크기: 2048 x 2048  
- 손가락 / 손등 / 팔로 구성되며 29개의 뼈 종류가 존재


## 개발 환경

- GPU : v100


## 협업 환경 
- Notion으로 실험 관리와 정보 공유를 진행함.

![image](https://github.com/user-attachments/assets/a6d68c64-da0e-4958-8cca-34da4cdec579)

- Wandb로 실험 결과를 공유하고 모델의 학습 상황을 시각화함.

![image](https://github.com/user-attachments/assets/6cf37168-f8fe-46ac-9b91-d248162b9580)

## Models
|Model|Features|Library|
|------|---|---|
|U-Net++|다중 skip connection을 사용해 다양한 스케일에서 특징을 통합, 세분화 성능을 개선함|SMP|
|DeepLabV3+|Atrous Spatial Pyramid Pooling (ASPP)와 디코더 모듈을 결합해 다중 스케일 맥락 정보와 경계 복원을 강화함|SMP|

## 데이터 증강 기법

### 데이터 증강 기법 설명

| 기법                     | 설명                                                                 |
|--------------------------|---------------------------------------------------------------------|
| **flip_lr (좌우 반전)**    | 이미지를 좌우로 뒤집어 대칭적인 데이터를 생성                             |
| **flip_ud (상하 반전)**    | 이미지를 상하로 뒤집어 다양한 시각적 변형 데이터를 제공                       |
| **rotate_90 (90도 회전)**  | 이미지를 시계 방향으로 90도 회전하여 방향성을 추가                      |
| **rotate_270 (270도 회전)**| 이미지를 시계 방향으로 270도 회전하여 데이터 다양성을 극대화                 |

### Albumentations 변환

| 기법                        | 설명                                                                                     | 적용 확률 (p) |
|-----------------------------|------------------------------------------------------------------------------------------|---------------|
| **Resize (리사이즈)**        | 이미지를 1536 x 1536 크기로 변환                                   | -             |
| **RandomBrightnessContrast**| 이미지의 밝기와 대비를 무작위로 조정                                               | 0.5           |
| **CLAHE**                   | 제한 대비 적응 히스토그램 균일화(CLAHE)를 적용하며, 클립 한계는 2.0, 타일 그리드 크기는 (8, 8)로 설정 | 0.5           |
| **Sharpen (샤프닝)**         | 이미지를 선명하게 하기 위해 알파 값(0.2–0.5)과 밝기(0.5–1.0)를 조정                  | 0.5           |
| **GaussianBlur (가우시안 블러)**| 커널 크기 1에서 3 사이로 가우시안 블러를 적용                                        | 0.5           |

---


## Conclusion
- 데이터 증강을 통한 성능 향상
- 이전 프로젝트보다 활발한 협업 툴 활용으로 효율적인 실험을 진행함
- 시각화를 통한 모델의 장단점을 파악하는 것의 중요성을 알게 됨
