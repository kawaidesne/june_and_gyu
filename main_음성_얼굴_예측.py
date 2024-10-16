from keras.models import load_model
import cv2
import dlib
import numpy as np
import time as tm
import pyaudio
import librosa
import sqlite3
from datetime import datetime
from collections import deque
import RPi.GPIO as GPIO
from keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input

# 모델 불러오기
model_face = load_model('model_face.keras')
model_speech = load_model('model_speech.keras')

# 얼굴 클래스
face_class_names = {
    0: "ahs",
    1: "jhs",
    2: "jsj",
    3: "kms",
    4: "ksh",
    5: "kwonms",
    6: "nch",
    7: "nsh",
    8: "sjg",
    9: "yhs"
}
# 음성 클래스
speech_class_names = {
    0: "ahs 출근",
    1: "ahs 퇴근",
    2: "jhs 출근",
    3: "jhs 퇴근",
    4: "jsj 출근",
    5: "jsj 퇴근",
    6: "kms 출근",
    7: "kms 퇴근",
    8: "ksh 출근",
    9: "ksh 퇴근",
    10: "kwonms 출근",
    11: "kwonms 퇴근",
    12: "nch 출근",
    13: "nch 퇴근",
    14: "nsh 출근",
    15: "nsh 퇴근",
    16: "sjg 출근",
    17: "sjg 퇴근",
    18: "yhs 출근",
    19: "yhs 퇴근",

}
# 얼굴 음성 클래스 mapping ex(얼굴 : [음성])
face_to_speech_mapping = {
    0 : [0, 1],
    1 : [2, 3],
    2 : [4, 5],
    3 : [6, 7],
    4 : [8, 9],
    5 : [10, 11],
    6 : [12, 13],
    7 : [14, 15],
    8 : [16, 17],
    9 : [18, 19]
    
}


# 필요 파라미터 설정
detector = dlib.get_frontal_face_detector()
table_name = "work_time_log_" + datetime.now().strftime('%Y%m%d')
TRIG = 23
ECHO = 24
GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)


#데이터 베이스 연결
def get_db_connection():
    conn = sqlite3.connect('checking.db')
    return conn

# 출근 시간 업데이트 함수
def update_check_in_time(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%H:%M:%S')
    update_sql = '''
    UPDATE {table_name}
    SET check_in_time = ?
    WHERE id = ?
    '''
    cursor.execute(update_sql.format(table_name=table_name), (current_time, employee_id))
    conn.commit()
    conn.close()

# 퇴근 시간 업데이트 함수
def update_check_out_time(employee_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.now().strftime('%H:%M:%S')
    update_sql = '''
    UPDATE {table_name}
    SET check_out_time = ?
    WHERE id = ?
    '''
    cursor.execute(update_sql.format(table_name=table_name), (current_time, employee_id))
    conn.commit()
    conn.close()

#현재시각 데이터 베이스에 기록
def update_time(number):
    room_number = (number // 2) + 1
    if number % 2 == 0:
        update_check_in_time(room_number)
    else:
        update_check_out_time(room_number)

# 출퇴근 기록 확인
def check_work_time_log(predicted_class_speech):
    employee_id = (predicted_class_speech // 2) + 1
    conn = get_db_connection()
    cursor = conn.cursor()
    check_in_sql = '''
    SELECT check_in_time, check_out_time
    FROM {table_name}
    WHERE id = ?
    '''
    cursor.execute(check_in_sql.format(table_name=table_name), (employee_id,))
    row = cursor.fetchone()
    check_in_time, check_out_time = row
    
    if predicted_class_speech % 2 == 0:
        # 출근 시
        if check_in_time is None:
                update_time(predicted_class_speech)
                print("출근 시간이 기록되었습니다") 
        elif check_out_time is None:
                print("출근 시간이 이미 기록되어있습니다.")


        else:        
                check_in_time1 = datetime.strptime(check_in_time, "%H:%M:%S")
                check_out_time2 = datetime.strptime(check_out_time, "%H:%M:%S")
                now_time = datetime.now()
                check_in_time11 = now_time.replace(hour=check_in_time1.hour, minute=check_in_time1.minute, second=check_in_time1.second, microsecond=0)
                check_out_time22 = now_time.replace(hour=check_out_time2.hour, minute=check_out_time2.minute, second=check_out_time2.second, microsecond=0)
               
                diff1 = abs(now_time - check_in_time11)
                diff2 = abs(now_time - check_out_time22)

                if diff1 < diff2:
                   print("출근 시간이 이미 기록되어있습니다.")
  
                else:
                     update_time(predicted_class_speech)
                     print("출근 시간이 기록되었습니다") 

            
        
    else:
        # 퇴근 시
        if check_out_time is None:
            if check_in_time is None:
                # 출근 o 퇴근 o
                print("출근 기록이 존재하지 않습니다.")
            else:
                update_time(predicted_class_speech)
                print("퇴근 시간이 기록되었습니다")

        else:
            check_in_time1 = datetime.strptime(check_in_time, "%H:%M:%S")
            check_out_time2 = datetime.strptime(check_out_time, "%H:%M:%S")
            now_time = datetime.now()
            check_in_time11 = now_time.replace(hour=check_in_time1.hour, minute=check_in_time1.minute, second=check_in_time1.second, microsecond=0)
            check_out_time22 = now_time.replace(hour=check_out_time2.hour, minute=check_out_time2.minute, second=check_out_time2.second, microsecond=0)
               
            diff1 = abs(now_time - check_in_time11)
            diff2 = abs(now_time - check_out_time22)

            if diff1 < diff2: 
                   update_time(predicted_class_speech)
                   print("퇴근 시간이 기록되었습니다")
            else:
                   print("퇴근 시간이 이미 존재합니다.")


# 거리 측정 함수
def get_distance():
    GPIO.output(TRIG, False)
    tm.sleep(0.1)
    GPIO.output(TRIG, True)
    tm.sleep(0.00001)
    GPIO.output(TRIG, False)

    # 에코 핀으로부터 신호 수신
    while GPIO.input(ECHO) == 0:
        pulse_start = tm.time()
    while GPIO.input(ECHO) == 1:
        pulse_end = tm.time()

    # 거리 계산
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    return distance

# 음성데이터 전처리 함수
def preprocess_audio(audio_data, sr=16000, n_mfcc=20, hop_length=256, n_fft=512, pad_len = None):
    voice_activity_indices = librosa.effects.split(y=audio_data, top_db=20)
    
    if len(voice_activity_indices) > 0:
        start_index, end_index = voice_activity_indices[0]
        audio_data_selected = audio_data[start_index:end_index]
    else:
        audio_data_selected = audio_data

    target_length = int(sr * 2.32)

    padding_length = max(0, target_length - len(audio_data_selected))

    if padding_length > 0:
        pad_width = (0, padding_length) 
        audio_data_selected = np.pad(audio_data_selected, pad_width, mode='constant')
    
    spectrogram = np.abs(librosa.stft(audio_data_selected, hop_length=hop_length, n_fft=n_fft))
    
    mfccs = librosa.feature.mfcc(S=spectrogram, n_mfcc=n_mfcc)
    mfccs_normalized_scaled = (mfccs-np.mean(mfccs))/np.std(mfccs)
    
    mfccs_scaled = np.expand_dims(mfccs_normalized_scaled, axis=-1)
    audio_mfcc = []
    audio_mfcc.append(mfccs_scaled)
    audio_mfcc_trans = np.transpose(audio_mfcc, (0, 2, 1, 3))
    
    return audio_mfcc_trans

# 음성 데이터 녹음
def record_audio(duration=5):
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    audio_data = np.hstack(frames)
    return audio_data

#얼굴과 음성 예측결과 비교
def is_same_person(predicted_class_face, predicted_class_speech):
    if predicted_class_speech in face_to_speech_mapping[predicted_class_face]:
        return True
    return False


def main():
    
    cap = cv2.VideoCapture(0)
    frame_count = 0
    frame_skip = 1 
    prediction_history = deque(maxlen=5)
    stable_prediction = None
    last_change_frame = 0
    stability_duration = 20
    
    while cap.isOpened():
        ret , frame = cap.read()
        distance = get_distance()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        if distance <= 50:
            frame_count += 1
            if frame_count % frame_skip == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_img = frame[y:y+h, x:x+w]
                    if face_img.size == 0:
                        continue
                    face_img = cv2.resize(face_img, (224, 224))
                    face_img = img_to_array(face_img)
                    face_img = preprocess_input(face_img)
                    face_img = np.expand_dims(face_img, axis=0)
                    prediction_face = model_face.predict(face_img)
                    predicted_class_face = np.argmax(prediction_face)
                    label = face_class_names[predicted_class_face]
                    # 화면에 얼굴 및 예측 결과 표시
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, "Predicted Face Class: {}".format(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # 예측 값을 큐에 추가
                    prediction_history.append(predicted_class_face)

                # 큐에 저장된 예측 값들의 최빈값을 계산
                if len(prediction_history) == prediction_history.maxlen:
                    most_common_prediction = max(set(prediction_history), key=prediction_history.count)

                    # 현재 예측 값이 안정된 예측 값과 다른 경우, 안정된 예측 값을 갱신
                    if most_common_prediction != stable_prediction:
                        stable_prediction = most_common_prediction
                        last_change_frame = frame_count
                    elif frame_count - last_change_frame >= stability_duration:
                        # 안정된 예측 값이 일정 시간 동안 유지된 경우 음성 인식 수행
                        while True:
                            audio_data = record_audio()
                            if len(audio_data) == 0:
                                print("음성이 입력되지 않았습니다. 다시 녹음하세요.")
                                continue
                            if audio_data is not None:
                                # 음성 활성화 감지를 위한 최소 에너지 값 설정
                                min_energy = np.mean(np.square(audio_data)) * 1e4
                                if min_energy < 0.1:  
                                    print("음성이 감지되지 않았습니다. 다시 녹음하세요.")
                                    continue
                                # 음성 데이터 전처리
                                input_data_speech = preprocess_audio(audio_data)
                                # 음성 데이터 예측
                                prediction_speech = model_speech.predict(input_data_speech)
                                predicted_class_speech = np.argmax(prediction_speech)
                                pre_speech = int(predicted_class_speech)
                                speech_name = speech_class_names[predicted_class_speech]
                                print(speech_name)
                                if is_same_person(predicted_class_face, predicted_class_speech):
                                    check_work_time_log(pre_speech)
                                    last_change_frame = frame_count                               
                                    break
                                else:
                                    print("음성인식 결과와 얼굴인식 결과가 동일하지 않습니다.")
                                    last_change_frame = frame_count
                                    break
                          
        cv2.imshow('CAMERA', frame)
        
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()