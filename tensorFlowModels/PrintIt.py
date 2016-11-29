# Here I want to show this data

# may also want to store the dimensions of the file.

import sqlite3

# first open the file

fileName = 'leftElbow.txt'
file_object = open(fileName, 'r')


sql_file = 'pose_DB.sqlite'
table_name = 'pose_train_table'
id_column = 'file_key'
name_column = 'fileName'
field_type = 'INTEGER'

file_object.readline()
tup1 = ()

# now we need 7 body parts
# suppose this is the Table schema

lwristx = 'left_x_wrist'
lwristy = 'left_y_wrist'
rwristx = 'right_x_wrist'
rwristy = 'right_y_wrist'
lelbowx = 'left_x_elbow'
lelbowy = 'left_y_elbow'
relbowx = 'right_x_elbow'
relbowy = 'right_y_elbow'
headx = 'x_head'
heady = 'y_head'
lshoulderx = 'left_x_shoulder'
lshouldery = 'left_y_shoulder'
rshoulderx = 'right_x_shoulder'
rshouldery = 'right_y_shoulder'
column_type = "REAL"



conn = sqlite3.connect(sql_file)
c = conn.cursor()

c.execute("SELECT filename, left_x_shoulder, left_y_shoulder, right_x_shoulder, right_y_shoulder FROM pose_train_table WHERE file_key = 0")
conn.commit()
row = c.fetchone()
print row[0], row[1], row[2], row[3], row[4]
#row = c.fetchone()
#print row[0], row[1]

# now we need 7 body parts
# suppose this is the Table schema





conn.close()
