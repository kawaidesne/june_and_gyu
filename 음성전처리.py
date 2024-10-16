import librosa
import numpy as np
import os
import pickle

def preprocess_audio(audio_file, sample_rate=16000, n_mfcc=20, hop_length=256, n_fft=512, pad_len=None):
    # 오디오 파일 로드
    audio_data, sr = librosa.load(audio_file, sr=sample_rate)
    
    # 음성 활성화 감지
    voice_activity_indices = librosa.effects.split(y=audio_data, top_db=20)
    
    # 첫 번째 음성 활성화 영역 선택
    if len(voice_activity_indices) > 0:
        start_index, end_index = voice_activity_indices[0]
        audio_data_selected = audio_data[start_index:end_index]
    else:
        # 음성 활성화가 감지되지 않은 경우, 전체 오디오 데이터 사용
        audio_data_selected = audio_data
    
    # 목표 길이 계산 (2.32초에 해당하는 샘플 수)
    target_length = int(sr * 2.32)

    # 현재 길이와 목표 길이의 차이 계산
    padding_length = max(0, target_length - len(audio_data_selected))

    # 패딩 적용
    if padding_length > 0:
        # 뒤쪽으로만 패딩을 적용합니다.
        pad_width = (0, padding_length)  # 앞쪽은 0으로 패딩하지 않음, 뒤쪽으로 padding_length만큼 패딩
        audio_data_selected = np.pad(audio_data_selected, pad_width, mode='constant')

    # 스펙트로그램 계산
    spectrogram = np.abs(librosa.stft(audio_data_selected, hop_length=hop_length, n_fft=n_fft))
    
    # MFCC 특징 추출
    mfccs = librosa.feature.mfcc(S=spectrogram, n_mfcc=n_mfcc)
    
    mfccs_normalized_scaled = (mfccs - np.mean(mfccs)) / np.std(mfccs)
    
    # CNN 모델에 입력할 형태로 데이터 변환
    mfccs_scaled = np.expand_dims(mfccs_normalized_scaled, axis=-1)  # 차원 확장: (시간 프레임 수, 특징 수, 1)
    
    return mfccs_scaled

def load_and_preprocess_data(main_folder, pad_len=None):
    processed_data = []
    labels = []
    
    # 상위 폴더 내의 하위 폴더들 탐색
    subfolders = [os.path.join(main_folder, subfolder) for subfolder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, subfolder))]
    
    for i, subfolder in enumerate(subfolders):
        # 하위 폴더 내의 WAV 파일 목록 가져오기
        files = os.listdir(subfolder)
        for file in files:
            if file.endswith(".wav"):
                # WAV 파일의 전체 경로 생성
                file_path = os.path.join(subfolder, file)
                
                # 오디오 데이터 전처리
                processed_audio = preprocess_audio(file_path, pad_len=pad_len)
                
                # 전처리된 데이터를 리스트에 추가
                processed_data.append(processed_audio)
                labels.append(i)
    
    return processed_data, labels

main_folder = "C:/Users/ygyi/Desktop/speech"
processed_data, labels = load_and_preprocess_data(main_folder)

# 데이터의 최대 길이 찾기
max_len = max(data.shape[1] for data in processed_data)

# 모든 데이터를 최대 길이에 맞추어 배열로 변환
processed_data_padded = np.array([np.pad(data, ((0, 0), (0, max_len - data.shape[1]), (0, 0)), mode='constant') for data in processed_data])

X = processed_data_padded
X = np.transpose(X, (0, 2, 1, 3))  # 데이터의 차원 순서를 (batch_size, features, time_steps)로 변경
y = np.array(labels)
print(X.shape)
data = {
    'X': X,
    'y': y
}

# 변수 저장
with open('variables.pkl', 'wb') as f:
    pickle.dump(data, f)
