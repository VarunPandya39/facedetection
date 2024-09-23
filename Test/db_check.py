import mysql.connector
from datetime import datetime
import io
import cv2
import base64
import numpy as np
from PIL import Image

def dbconnection():
    conn = mysql.connector.connect(host='localhost', username='root', password = '', database='employee_details')
    cursor = conn.cursor()
    print(f"Connected db")
    return conn, cursor

def show(base64_string, ctime):
    x = len(base64_string) % 4
    extra = "=" * x
    base64_string = base64_string + extra
    imgdata = base64.b64decode(base64_string)
    nparr = np.frombuffer(imgdata, np.uint8)
    imgdata = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    while True:
        cv2.imshow(f"{ctime}", imgdata)
        if cv2.waitKey(0) == ord('z'):
            break

# Define id and capture_time
employee_name = f"Varun Pandya_1"  # example employee_id
capture_time = datetime.now().strftime('%y-%m-%d')

conn1, cursor1 = dbconnection()
# q = f"SELECT file_loc, createtime from ai_alerts where alerttype='eating';"
q = f"SELECT employee_image, capture_datetime FROM employee_images WHERE employee_name = %s AND DATE_FORMAT(capture_datetime, '%%y-%%m-%%d') = %s" ,(employee_name,capture_time)
cursor1.execute(q)
rows = cursor1.fetchall()
for i in rows:
    data = i[0]
    createtime = i[1]
    show(data, createtime)

