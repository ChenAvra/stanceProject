import sqlite3
from os.path import exists
import pandas as pd

class DataBase:

    def __init__(self):
        if not exists('Stance_Detection.db'):
            self.conn = sqlite3.connect('Stance_Detection.db')
            self.cursor = self.conn.cursor()
            self.create_claimTable()
            self.fill_claim_table()
        else:
            self.conn = sqlite3.connect('Stance_Detection.db')
            self.cursor = self.conn.cursor()



    def create_claimTable(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Claims(Dataset_Number INTEGER NOT NULL, Claim TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        self.conn.commit()

    def fill_claim_table(self):
        # self.cursor.executemany("INSERT INTO store  VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);", csv_reader)
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);",(1,"Trump","I Love Him","Support") )
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (1, "Trump", "I Hate Him", "Deny"))
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (1, "Arnon", "I Love Him", "Support"))
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (1, "Arnon", "I Love Him Very Much", "Support"))
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);",(2,"meir_kelach","I Love Him","Support") )
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (2, "meir_kelach", "I Hate Him", "Deny"))
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (2, "koby_gal", "I Love Him", "Support"))
        self.cursor.execute("INSERT INTO Claims  VALUES (?,?,?,?);", (2, "koby_gal", "I Love Him Very Much", "Support"))
        self.conn.commit()

    def get_dataset(self, dataset_number):
        query = "SELECT Claim,Sentence,Stance FROM Claims WHERE Dataset_Number=" + str(dataset_number)
        df = pd.read_sql_query(query, self.conn)
        return df

db = DataBase()
print(db.get_dataset(1))
print(db.get_dataset(2))