# may also want to store the dimensions of the file.
import sqlite3

sql_file = 'pose_DB.sqlite'
table_name = 'pose_train_table'
id_column = 'file_key'
name_column = 'fileName'
field_type = 'INTEGER'

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

print(myList)



print(myList[0])
conn = sqlite3.connect(sql_file)

c = conn.cursor()

for x in range(0,len(myList)):
    new_column1 = myList[x]
    c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"\
              .format(tn=table_name, cn = new_column1, ct=column_type))



column_type = 'TEXT'
c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' {ct}"\
              .format(tn=table_name, cn = name_column, ct=column_type))


conn.commit()
conn.close()
