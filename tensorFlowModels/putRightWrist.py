# may also want to store the dimensions of the file.

import sqlite3
import sys, os

# first open the file



sql_file = 'pose_DB.sqlite'
table_name = 'pose_train_table'
id_column = 'file_key'
name_column = 'fileName'
field_type = 'INTEGER'


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
rhipx = "right_hip_x"
rhipy = "right_hip_y"
lhipx = "left_hip_x"
lhipy = "left_hip_y"
rknex = "right_knee_x"
rkney = "right_knee_y"
lknex = "left_knee_x"
lkney = "left_knee_y"
rankx =  "right_ankle_x"
ranky = "right_ankle_y"
lankx = "left_ankle_x"
lanky = "left_ankle_y"
column_type = "REAL" 

myList = [lwristx, lwristy, rwristx, rwristy, lelbowx, lelbowy, relbowx, relbowy, lshoulderx, lshouldery, rshoulderx, rshouldery, headx, heady]
#myList = []
myList += [rhipx]
myList += [rhipy]
myList += [lhipx]
myList += [lhipy]
myList += [rknex]
myList += [rkney]
myList += [lknex]
myList += [lkney]
myList += [rankx]
myList += [ranky]
myList += [lankx]
myList += [lanky]

# reads the data from the text file and puts it in a list
def read_file(filename):
    file_object = open(filename, 'r')
    file_object.readline()
    tup1 = []
    for x in range(0,20899):
        myString = file_object.readline()
        myString = myString.replace("\r\n","")
        myString = myString.split(' ')
        tup2 = (myString[0], myString[1],)
        tup1 += tup2
    file_object.close()
    return tup1

def insert_database(limb_data, columnx_name, columny_name):
    conn = sqlite3.connect(sql_file)
    c = conn.cursor()
    data_string = "UPDATE pose_train_table SET " + columnx_name + " \
            = ?, "+ columny_name + " = ? WHERE file_key = ?"
    for x in range(0,20899):
        s= 2*x
        c.execute(data_string, (limb_data[s], limb_data[s+1], x))
    conn.commit()
    conn.close()

# now we should loop over every limb of interest (myList)
# one problem is that the list of names of the columns is not the
# same as the filenames.  Would be nice if only one list was needed

for x in range(0, int(len(myList)/2)):
    indices = 2*x
    file_path = myList[x]
    file_path = file_path.replace("_","")
    file_path = file_path.replace("x","")
    file_path = file_path.replace("y","")
    file_path = "/home/david/programming/mcmaster-text-to-motion-database/tensorFlowModels/PoseData/" + file_path + ".txt"
    #print(file_path)
    listData = read_file(file_path)
    insert_database(listData, myList[indices], myList[indices + 1])
