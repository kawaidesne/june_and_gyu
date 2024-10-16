from flask import Flask, render_template
import sqlite3
from flask_socketio import SocketIO, emit
from datetime import datetime

table_name = "work_time_log_" + datetime.now().strftime('%Y%m%d')
# 직원 이름
employee_names = [
    "안해성",
    "전호성",
    "전석준",
    "김민식",
    "김수한",
    "권민수",
    "남철환",
    "나상호",
    "신진규",
    "양현승"
]


#데이터 베이스 연결
def get_db_connection():
    conn = sqlite3.connect('checking.db')
    return conn

# 테이블 생성 함수
def create_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    create_table_sql = f'''
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        check_in_time TEXT,
        check_out_time TEXT
    )
    '''
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

# 데이터 삽입 함수
def insert_employee(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    insert_sql = f'''
    INSERT OR IGNORE INTO {table_name} (name)
    VALUES (?)
    '''
    cursor.execute(insert_sql, (name,))
    conn.commit()
    conn.close()

# 직원 이름 추가 함수
def add_employees(names):
    for name in names:
        insert_employee(name)

app = Flask("webproject")
socketio = SocketIO(app)

def get_data():
    conn = sqlite3.connect('checking.db')
    cursor = conn.cursor()
    query = f'SELECT * FROM {table_name}'
    
    data = cursor.execute(query).fetchall()
    conn.close()
    return data

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    data = get_data()
    emit('update_data', {'data': data})

@socketio.on('request_update')
def handle_request_update():
    data = get_data()
    emit('update_data', {'data': data})    

@app.route('/')
def index():
    current_time = datetime.now().strftime("%Y-%m-%d")
    data = get_data()
    print(data)  # 데이터 확인
    return render_template('index.html', data=data, current_time = current_time)

create_table()
add_employees(employee_names)
socketio.run(app, host="0.0.0.0", port=5001, debug=True)
