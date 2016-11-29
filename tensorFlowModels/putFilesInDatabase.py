# may also want to store the dimensions of the file.

import sqlite3

# first open the file

fileName = 'filenames.txt'
file_object = open(fileName, 'r')


file_object.readline()
tup1 = ()

for x in range(0,20900):
    myString = file_object.readline()
    myString = myString.replace("\r\n","")
    tup2 = (myString,)
    tup1 += tup2

file_object.close()



sql_file = 'pose_DB.sqlite'
table_name = 'pose_train_table'
id_column = 'file_key'
name_column = 'fileName'
field_type = 'INTEGER'

# now we need 7 body parts
# suppose this is the Table schema


conn = sqlite3.connect(sql_file)
c = conn.cursor()

for x in range(0,20899):
    s= 2*x
    c.execute("UPDATE pose_train_table SET filename = ?\
        WHERE file_key = ?", (tup1[x], x))



#for x in range(0,20900):
#    c.execute("INSERT INTO\
#              pose_train_table (file_key, fileName) VALUES (?, ?)", (x, tup1[x]))


conn.commit()
conn.close()




