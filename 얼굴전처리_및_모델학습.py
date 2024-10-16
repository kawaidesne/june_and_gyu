import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import shuffle

# 데이터 로드 및 전처리 함수
def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    class_names = os.listdir(data_dir)
    class_names.sort()
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        class_images = []
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            # 이미지를 넘파이 배열로 변환하여 전처리
            image = img_to_array(image)
            image = preprocess_input(image)
            class_images.append(image)
            labels.append(label)
        images.extend(class_images)
    
    # 이미지와 레이블을 동시에 섞음
    images, labels = shuffle(images, labels, random_state=42)
    
    images = np.array(images, dtype="float32")
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(class_names))
    
    return images, labels, class_names

data_dir = 'c:/Users/ygyi/Desktop/facef'
images, labels, class_names = load_and_preprocess_data(data_dir)

# 모델 구성
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# 사전 훈련된 모델의 가중치를 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(images, labels, validation_split=0.2, epochs=10, batch_size=16, callbacks=[early_stopping])

# 모델 저장
model.save('model_with_shuffle.keras')
