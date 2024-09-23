import pickle
import mysql.connector
from datetime import datetime
import os
import base64
import pandas as pd
import numpy as np

image_folder = "Employee_Images"
download_folder = "Downloaded_Images"
export_folder = "Attendance_Records"

# MySQL connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'Employee_Details'
}

# Establishing MySQL connection
def connect_to_database():
    try:
        connection = mysql.connector.connect(**db_config)
        print("Connected to MySQL database")
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

# Creating a table for employee images if not exists
def create_table(connection):
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employee  
            (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_name VARCHAR(100) NOT NULL,
                contact_no INT(10) DEFAULT 0,
                status TINYINT(1) DEFAULT 1,
                faceEncoding BLOB
            );
        """)
        print("Table 'employee' created successfully.")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employee_images  
            (
                id INT AUTO_INCREMENT PRIMARY KEY,
                emp_id INT NOT NULL,
                employee_name VARCHAR(100) NOT NULL,
                employee_image LONGTEXT,
                capture_datetime DATETIME NOT NULL
            )
        """)
        connection.commit()
        print("Table 'employee_images' created successfully.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")


# Check if attendance is already marked for the current date
def check_attendance(connection , employee_id):

    connection = connect_to_database()
    # Get today's date in the format you are using in your database
    today = datetime.now().date()

    # Create a database connection
    conn = connect_to_database()
    cursor = conn.cursor()

    # Query to check if the employee has already marked attendance today
    query = """
        SELECT capture_datetime FROM employee_images 
        WHERE emp_id = %s AND DATE(capture_datetime) = %s
    """
    cursor.execute(query, (employee_id, today))

    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if result:
        return 'present'  # Attendance marked today
    else:
        return 'absent'  # No attendance record for today


def image_exists(cursor, employee_name, capture_date):
    cursor.execute("""
        SELECT COUNT(*) FROM employee_images
        WHERE employee_name = %s AND DATE_FORMAT(capture_datetime,'%Y-%m-%d') = %s
    """, (employee_name, capture_date))
    return cursor.fetchone()[0] > 0

def employee_exists(cursor, employee_name,faceEncoding):
    try:
        serialized_encoding = pickle.dumps(faceEncoding)

        cursor.execute("""
        SELECT COUNT(*) FROM employee WHERE employee_name = %s AND faceEncoding = %s
        """, (employee_name,serialized_encoding))

        return cursor.fetchone()[0] > 0
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False

# Encoding image to Base64
def encode_image_to_base64(realtime_image_path):
    with open(realtime_image_path, 'rb') as image_file:
        image_data = image_file.read()
    return base64.b64encode(image_data).decode('utf-8')

# Inserting captured image data into MySQL database
def insert_image_data(connection, emp_id, realtime_image_path, face_detected):
    try:
        if face_detected:
            cursor = connection.cursor()
            capture_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Check if the attendance is already marked
            if check_attendance(connection, emp_id):
                print("Attendance Already Marked.")
                return {'status': 'already_marked'}

            # Fetch the employee face encoding from the database using emp_id
            faceEncoding = get_encoding_from_db(connection, emp_id)
            if faceEncoding is None:
                print(f"No encoding found for employee ID {emp_id}. Cannot mark attendance.")
                return

            # Fetch the employee name using the emp_id
            cursor.execute("""SELECT employee_name FROM employee WHERE id = %s""", (emp_id,))
            result = cursor.fetchone()

            if result:
                employee_name = result[0]
                print(f"Employee Name for ID {emp_id} found: {employee_name}")
            else:
                print(f"Employee with ID {emp_id} not found.")
                return

            # Convert the image to base64
            image_data_base64 = encode_image_to_base64(realtime_image_path)
            if image_data_base64:
                # Insert the employee image record
                cursor.execute("""INSERT INTO employee_images (employee_name, emp_id, employee_image, capture_datetime)
                                  VALUES (%s, %s, %s, %s)""",
                               (employee_name, emp_id, image_data_base64, capture_datetime))
                connection.commit()

                print("Attendance Marked. Image Data Inserted Into Database.")
                return {'status': 'marked'}

    except mysql.connector.Error as err:
        print(f"Error: {err}")
def load_encodings_from_db():

    connection = connect_to_database()
    # Load face encodings from the database
    print("Fetching encodings from the database...")
    existing_encodings = []
    empNames = []

    # Check if the connection is valid
    if connection is None:
        print("Error: Database connection is not established.")
        return empNames, existing_encodings

    try:
        # Fetch all employee names and encodings from the database
        cursor = connection.cursor()
        cursor.execute("SELECT employee_name, faceEncoding FROM employee")
        rows = cursor.fetchall()

        if not rows:
            print("No records found in the employee table.")
            return empNames, existing_encodings  # Return empty lists if no data is found

        for row in rows:
            empNames.append(row[0])
            try:
                # Deserialize the encoding using pickle
                encoding = pickle.loads(row[1])
                if encoding is None or len(encoding) == 0:
                    raise ValueError("Invalid encoding retrieved from database")
            except Exception as e:
                print(f"Error decoding encoding for {row[0]}: {e}")
                continue

            existing_encodings.append(encoding)

        print(f"Loaded {len(existing_encodings)} face encodings from the database.")
        return empNames, existing_encodings

    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        return  [], []  # Return empty lists in case of an error

def insert_encoding_to_db(connection,employee_name, faceEncoding):
    try:
        cursor = connection.cursor()
        if not employee_exists(cursor, employee_name,faceEncoding):
            serialized_encoding =pickle.dumps(faceEncoding)
            #print(f"Serialized encoding size: {len(serialized_encoding)} bytes")
            query = "INSERT INTO employee (employee_name, faceEncoding) VALUES (%s, %s)"
            cursor.execute(query, (employee_name, serialized_encoding))  # Store encoding as binary data
            connection.commit()
            print(f"Encoding for {employee_name} inserted into database.")
        else:
            print(f"Encoding for {employee_name} already exists in the database.")

    except Exception as e:
        print(f"Failed to insert encoding into database: {e}")
def get_encoding_from_db(connection, emp_id):
    try:
        cursor = connection.cursor()
        # Query to fetch the face encoding based on emp_id
        query = "SELECT faceEncoding FROM employee WHERE id = %s"
        cursor.execute(query, (emp_id,))
        result = cursor.fetchone()

        if result:
            # Deserialize the face encoding using pickle
            serialized_encoding = result[0]
            face_encoding = pickle.loads(serialized_encoding)
            # Print statement for debugging (optional)
            # print(f"Encoding for employee ID {emp_id} retrieved from database.")
            return face_encoding
        else:
            print(f"No encoding found for employee ID {emp_id}.")
            return None

    except Exception as e:
        print(f"Failed to fetch encoding from database: {e}")
        return None

def download_image(connection, employee_name, capture_datetime_str, download_folder, image_data_base64):
    try:
        capture_datetime = datetime.strptime(capture_datetime_str,'%Y-%m-%d %H:%M:%S')
        image_data = base64.b64decode(image_data_base64)
        image_filename = f"{employee_name}_{capture_datetime.strftime('%Y-%m-%d_%H-%M-%S%p')}.png"

        if not os.path.exists(download_folder):
            os.makedirs(download_folder)

        image_path = os.path.join(download_folder, image_filename)
        with open(image_path, 'wb') as image_file:
            image_file.write(image_data)
        print(f"Image downloaded and saved as {image_path}")

    except mysql.connector.Error as err:
        print(f"Error: {err}")

# Fetch attendance records and export to Excel
def export_attendance_to_excel(connection, capture_date, export_folder,download_folder):
    try:
        cursor = connection.cursor(dictionary=True)
        #cursor = connection.cursor()
        query = """
            #SELECT employee_name,capture_datetime
            SELECT employee_name,employee_image,DATE_FORMAT(capture_datetime, '%Y-%m-%d %H:%i:%S') AS capture_datetime
            FROM employee_images
            WHERE DATE(capture_datetime) = %s
        """
        cursor.execute(query, (capture_date,))
        records = cursor.fetchall()
        #print(f"Debug: fetched records = {records}")

        if records:
            df = pd.DataFrame(records, columns=['employee_name','capture_datetime'])
            #df = pd.DataFrame(records)
            # Debugging: Print DataFrame to verify its content
            print(f"Debug: DataFrame = {df}")
            if not os.path.exists(export_folder):
                os.makedirs(export_folder)
            export_path = os.path.join(export_folder, f"attendance_report_{capture_date}.xlsx")
            df.to_excel(export_path, index=False)
            print(f"Attendance records exported to {export_path}")
            for record in records:
                employee_name = record['employee_name']
                capture_datetime = record['capture_datetime']
                image_data_base64 = record['employee_image']
                download_image(connection,employee_name, capture_datetime,download_folder,image_data_base64)
            print(f"Employee Images Downloaded and saved to {download_folder}")
        else:
            print("No attendance records found for the specified date.")

    except mysql.connector.Error as err:
        print(f"Error: {err}")


def insert_attendance(employee_name, realtime_image_path):
    # Create a database connection
    conn = connect_to_database()
    cursor = conn.cursor()

    # Insert attendance record into your database
    query = """
        INSERT INTO employee_images (employee_name, attendance_date, image_path)
        VALUES (%s, %s, %s, %s)
    """
    cursor.execute(query, (employee_name, datetime.now(), realtime_image_path))
    conn.commit()
    cursor.close()
    conn.close()
def close_connection(connection):
    try:
        connection.close()
        print("MySQL connection closed.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")

if __name__ == "__main__":
    connection = connect_to_database()
    if connection:
        create_table(connection)
    close_connection(connection)