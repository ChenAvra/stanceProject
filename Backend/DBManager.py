import sqlite3
from os.path import exists
import pandas as pd
import csv

class DataBase:

    def __init__(self):
        if not exists('Stance_Detection.db'):
            self.conn = sqlite3.connect('Stance_Detection.db')
            self.cursor = self.conn.cursor()
            self.create_claimTable()
            # self.fill_claim_table()
        else:
            self.conn = sqlite3.connect('Stance_Detection.db')
            self.cursor = self.conn.cursor()



    def create_claimTable(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Claims(Dataset_Number INTEGER NOT NULL, Claim TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        self.conn.commit()

    def fill_claim_table(self,path,dataset_number):
        # self.cursor.execute("CREATE TABLE IF NOT EXISTS Claims(Dataset_Number INTEGER NOT NULL, Claim TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # for line in csv_reader:
            #     print(line)
            query = "INSERT INTO Claims VALUES({},?,?,?);".format(dataset_number)
            self.cursor.executemany(query, csv_reader)
        self.conn.commit()



    def get_dataset(self, dataset_number):
        query = "SELECT Claim,Sentence,Stance FROM Claims WHERE Dataset_Number=" + str(dataset_number)
        df = pd.read_sql_query(query, self.conn)
        return df

db = DataBase()
# db.fill_claim_table("semEval2016.csv",1)
db.fill_claim_table("MLP dataset.csv",3)
# db.fill_claim_table("MPCHI.csv",4)
# db.fill_claim_table("EmergentLite.csv",5)
print(db.get_dataset(3))

