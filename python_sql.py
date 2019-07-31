import sqlite3
#import pandas as pd

#create a new database file and connect to it
con = sqlite3.connect("mydata.sqlite")

#execute queries one by one
#query="DROP TABLE TEST;"
#con.execute(query)

query="""
CREATE TABLE TEST(
A VARCHAR(20),
B VARCHAR(20),
C REAL,
D INTEGER
);"""
con.execute(query)
con.commit()

data = [("Atlanta","Georgia",1.25,6),("Tallahassee","Florida",2.6,3),("Sacremento","California",1.7,5)]

stmt="INSERT INTO TEST VALUES(?,?,?,?)"

con.executemany(stmt,data)
con.commit()

cursor = con.execute("select * from test")

rows = cursor.fetchall()

print rows

print cursor.description

#DF = pd.DataFrame(rows, columns=[x[0] for x in cursor.description])
#print DF