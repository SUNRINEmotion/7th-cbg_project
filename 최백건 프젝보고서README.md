가위바위보 사진 분류

프로젝트 동기

처음 봤을 때 너무 복잡하거나 어렵지 않고 누구나 쉽게 흥미를 느낄 수 있는 인공지능 프로젝트를 만들어보고 싶었습니다. 그래서 일상에서 우리가 흔히 접할 수 있는 가위바위보 게임을 이용했습니다. 손모양을 보고 가위인지 바위인지 보인지 인식할 수 있는 인공지능 모델을 만들어 처음 본 사람도 인공지능과 머신러닝에 관심을 가지게끔 하고 싶었습니다. 더불어 인공지능 이미지 인식 기술을 살펴보고 이해하는 데 도움을 주고, 인공지능에 대한 막연한 두려움을 없애 누구나 쉽게 활용할 수 있는 미래 과학기술이라는 것을 깨닫도록 합니다.

프로젝트 설명
```python
!wget --no-check-certificate \
  https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip \
  -O /tmp/rockpaperscissors.zip
```
wget 명령어를 사용해 깃허브 레포지토리에서 가위바위보 사진 데이터셋을 다운로드했습니다. --no-check-certificate으로 서버의 인증서를 확인하지 않고 가져옵니다.

```python
train_rock_dir, val_rock_dir = train_test_split(os.listdir(rock_dir), test_size = 0.4)
train_paper_dir, val_paper_dir = train_test_split(os.listdir(paper_dir), test_size = 0.4)
train_scissors_dir, val_scissors_dir = train_test_split(os.listdir(scissors_dir), test_size = 0.4)
```
가져온 가위바위보 데이터들을 학습(train)셋과 검증(val)셋으로 분할하고, 각 클래스(rock, paper, scissors)의 이미지를 해당 디렉터리에 복사합니다. 전체 데이터들 중 40%를 검증용으로 사용합니다.

```python
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True, shear_range = 0.2, validation_split = 0.4, fill_mode = 'nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
```
이미지 데이터를 전처리하는 객체를 생성합니다.
rescale=1./255로 이미지의 픽셀 값을 0에서 1 사이로 정규화(normalize)한 뒤 
20도까지 랜덤하게 회전시키고 뒤집습니다.

```python
IMG_HEIGHT = 150
IMG_WIDTH = 150

model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
```
이미지들의 크기를 가로세로 150으로 설정해 입력 이미지의 모양을 통일시키고, 이미지를 스캔하는 필터의 크기를 3x3으로 합니다. 이미지의 각 특성 맵에서 가장 큰 값을 추출하여 크기를 줄이는 최대 풀링(MaxPooling) 층을 추가합니다. 2x2 크기를 사용하여 특성 맵을 반으로 줄입니다. 또한 입력되는 이미지들 중 25%를 삭제해 과적합을 방지합니다.

```python
history = model.fit(train_generator, steps_per_epoch=25, epochs=50, validation_data = validation_generator, verbose=0, callbacks=[checkpointer])
```
모델을 50번 훈련시켜 손실과 정확도를 기록합니다.

```python
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Loss')
plt.xlabel('epoch')
plt.show()
```
손실과 정확도의 그래프를 그립니다.

```python
import cv2
import numpy as np
img_path = '20240211_133925.jpg'
img = cv2.imread(img_path)
img = cv2.resize(img, (150, 150))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)
predictions = model.predict(img)
predicted_class = np.argmax(predictions)
print("Predicted class:", predicted_class)
```
직접 찍은 사진을 모델에 넣어 모델이 잘 작동하는지 확인합니다.

실행결과

![image](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/7dd80496-2776-4100-a402-1bc90973061a)

모델을 50번 훈련시킵니다.
![image](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/3de4cc37-f478-4afc-b22a-e10dcf59f528)
![image](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/62006c1c-43f7-4954-b9a7-4b7ed6a59b00)

정확도와 손실의 그래프입니다. 정확도는 올라가고, 손실은 줄어듭니다.
![20240211_133925](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/30ce6f80-dc63-4553-bd7a-686154f568b9)

보자기를 찍어 모델에게 입력했습니다.

![image](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/29c4b23e-48dc-41de-8e48-cdf356ea5c30)

보자기는 0번 클래스입니다.
![image](https://github.com/SUNRINEmotion/7th-cbg_project/assets/112744687/7007adb9-fe28-470c-925b-df9d80ba3ed2)

0번으로 출력됩니다.

개선 방안
가위, 바위, 보 중 하나를 찍은 사진을 인공지능에게 보내면 인공지능이 랜덤으로 가위, 바위, 보 중 하나를 골라
인공지능과 대결해 가위바위보의 승패를 알려주는 기능도 만들어보고 싶습니다.
