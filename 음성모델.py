import pickle
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Conv1D, MaxPooling1D, Flatten, Activation
from sklearn.model_selection import StratifiedKFold
import numpy as np
# 변수 로드
with open('variables.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# 로드된 변수 사용
X = loaded_data['X']
y = loaded_data['y']


# k-fold cross-validation 설정
k = 5  # 폴드 개수 설정
skf = StratifiedKFold(n_splits=k)

fold_accuracies = []

# 각 폴드에서 모델을 훈련하고 평가
for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
    print(f"{fold_idx + 1}번째 폴드")

    X_train_fold, X_val_fold = X[train_indices], X[val_indices]
    y_train_fold, y_val_fold = y[train_indices], y[val_indices]
    
    # 모델 생성
    model = Sequential()
    model.add(Conv1D(512, kernel_size=3,  input_shape=(X.shape[1],X.shape[2]))) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(256, kernel_size=3))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(256, kernel_size=3))  
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=3)) 
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(128, kernel_size=3))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    model.add(Dense(20, activation='softmax'))

    # 모델 컴파일
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 모델 훈련
    model.fit(X_train_fold, y_train_fold, epochs=20, validation_data=(X_val_fold, y_val_fold))
    
    # 모델 평가
    scores = model.evaluate(X_val_fold, y_val_fold)
    fold_accuracy = scores[1]
    fold_accuracies.append(fold_accuracy)
    print(f"검증 정확도: {fold_accuracy}")
    
    # 향후 예측을 위해 모델 저장
    model.save(f'rmodel_fold_{fold_idx + 1}.keras')

# 폴드 간 평균 검증 정확도
avg_accuracy = np.mean(fold_accuracies)
print(f"평균 검증 정확도: {avg_accuracy}")

# 가장 높은 정확도를 가진 폴드 출력
best_fold_idx = np.argmax(fold_accuracies)  # 가장 높은 정확도를 가진 폴드 인덱스 찾기
print(f"가장 높은 정확도를 가진 폴드: {best_fold_idx + 1}")