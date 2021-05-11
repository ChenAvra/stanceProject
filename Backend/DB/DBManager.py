import os
import sqlite3
from os.path import exists
import pandas as pd
import csv
import json
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from pandas import DataFrame

class DataBase:


    def __init__(self):
        PROJECT_ROOT = os.path.abspath(__file__)
        BASE_DIR = os.path.dirname(PROJECT_ROOT)
        db_path = BASE_DIR + '\\Stance_Detection.db'
        if not exists(db_path):
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()
            self.create_claimTable()
            self.create_result_table()
            # self.fill_claim_table()
        else:
            self.conn = sqlite3.connect(db_path)
            self.cursor = self.conn.cursor()

    def create_Stance_Result_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Stance_Result(Topic TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        self.conn.commit()

    def create_index_table(self):
        self.cursor.execute('CREATE TABLE IF NOT EXISTS index_dataset (id text)')
        self.conn.commit()

    def drop_Index(self):
        query = "DROP TABLE index_dataset;"
        self.cursor.execute(query)
        self.conn.commit()

    def reset_index(self):
        sql='insert into index_dataset values (0)'
        self.cursor.execute(sql)
        self.conn.commit()



    def get_index(self):
        query = 'SELECT * FROM index_dataset'
        index = pd.read_sql_query(query, self.conn)
        index=index.iloc[0]['id']
        index=int(index)+1
        self.cursor.execute('UPDATE index_dataset SET id="{}"'.format(index))
        self.conn.commit()
        return index

    def drop_Request(self):
        query = "DROP TABLE Request;"
        self.cursor.execute(query)
        self.conn.commit()
    def create_Request_table(self):

        self.cursor.execute(
                "CREATE TABLE IF NOT EXISTS Request ( id INTEGER PRIMARY KEY AUTOINCREMENT, Model TEXT NOT NULL, Dataset TEXT NOT NULL, Train_percent INTEGER NOT NULL)")
        self.conn.commit()

    def insert_records_request(self, model, dataset, train_percent):
        query = 'INSERT INTO Request VALUES(?,?,?,?);'
        self.cursor.execute(query, (None,model, dataset, train_percent))
        self.conn.commit()
        query = 'SELECT id FROM Request where Model="{}" and Dataset="{}" and Train_percent="{}"'.format(model,dataset,train_percent)
        index = pd.read_sql_query(query, self.conn)
        index = index.iloc[0]['id']
        return int(index)

    def get_record_from_request(self,model, dataset,train_percent ):
        query = 'SELECT id FROM Request WHERE Model="{}" AND Dataset="{}" AND Train_percent={}'.format(model,dataset,train_percent/100)
        # query = "SELECT * FROM Result"
        index = pd.read_sql_query(query, self.conn)
        if(index.empty):
            return None
        index = index.iloc[0]['id']
        return int(index)

    def get_records_from_request_by_id(self,id):
        query = 'SELECT * FROM Request WHERE id="{}"'.format(id)
        result = pd.read_sql_query(query, self.conn)

        return result


    def insert_to_Stance_Result(self,topic,sentence,stance):
        query = 'INSERT INTO Stance_Result VALUES(?,?,?);'
        self.cursor.execute(query,(topic,sentence,stance))
        self.conn.commit()

    def get_record_from_Stance_Result(self,topic,sentence):
        query = 'SELECT * FROM Stance_Result WHERE Topic="{}" AND Sentence="{}"'.format(topic,sentence)
        # query = "SELECT * FROM Result"
        df = pd.read_sql_query(query, self.conn)
        return df

    def create_result_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Result (Model TEXT NOT NULL,Dataset TEXT NOT NULL,Train_percent INTEGER NOT NULL,Accuracy INTEGER NOT NULL,Class_report TEXT NOT NULL,roc_acc INTEGER NOT NULL,actual TEXT NOT NULL,predict TEXT NOT NULL,array_labels TEXT NOT NULL,cm TEXT NOT NULL,target TEXT NOT NULL,df_train_records TEXT NOT NULL,df_test_records TEXT NOT NULL,tpr_fpr TEXT NOT NULL,type TEXT NOT NULL)")
        self.conn.commit()

    def insert_records_to_result(self,model, dataset,train_percent, accuracy, class_report,roc_acc,actual,predict,array_labels,cm,target,df_train_records,df_test_records,dict_tpr_fpr_string,type):
        query = 'INSERT INTO Result VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);'
        self.cursor.execute(query,(model,dataset,train_percent,accuracy,class_report,roc_acc,actual,predict,array_labels,cm,target,df_train_records,df_test_records,dict_tpr_fpr_string,type))
        self.conn.commit()

    def get_record_from_result(self,model, dataset,train_percent ):
        query = 'SELECT * FROM Result WHERE Model="{}" AND Dataset="{}" AND Train_percent={}'.format(model,dataset,str(train_percent))
        # query = "SELECT * FROM Result"
        df = pd.read_sql_query(query, self.conn)
        return df

    def delete_record_from_result(self, key,model, dataset,train_percent ):
        query = 'DELETE FROM Result WHERE Model="{}" AND Dataset="{}" AND Train_percent={}'.format(model, dataset,str(train_percent))
        self.cursor.execute(query)
        self.conn.commit()

    def delete_from_result(self):
        query = 'DELETE FROM Result;'
        self.cursor.execute(query)
        self.conn.commit()

    def drop_result(self):
        query = "DROP TABLE Result;"
        self.cursor.execute(query)
        self.conn.commit()

    def create_claimTable(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Claims(Dataset_Number INTEGER NOT NULL, Claim TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        self.conn.commit()

    def fill_claim_table(self,path,dataset_number):
        df = self.get_dataset(dataset_number)
        if df.shape[0]>0:
            return
        # self.cursor.execute("CREATE TABLE IF NOT EXISTS Claims(Dataset_Number INTEGER NOT NULL, Claim TEXT NOT NULL, Sentence TEXT NOT NULL, Stance TEXT NOT NULL)")
        if dataset_number==6 or dataset_number==10:
            with open(path, "r", encoding='utf-8') as csv_file:
                print(type(csv_file))
                csv_reader = csv.reader(csv_file, delimiter=',')
                # for line in csv_reader:
                #     print(line)
                query = "INSERT INTO Claims VALUES({},?,?,?);".format(dataset_number)
                self.cursor.executemany(query, csv_reader)
        else:
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                # for line in csv_reader:
                #     print(line)
                query = "INSERT INTO Claims VALUES({},?,?,?);".format(dataset_number)
                self.cursor.executemany(query, csv_reader)
        self.conn.commit()

    def insert_semEveal_2017(self, path, dataset_number):
        df = self.get_dataset(dataset_number)
        if df.shape[0]>0:
            return
        f = open(path, "r", encoding='utf-8')
        for x in f:
            d = eval(x)
            print(d)
            topic = ""
            if not d['text']=="":
                if d['topic']=='charliehebdo':
                    topic = "charlie hebdo"
                elif d['topic']=='ferguson':
                    topic = "ferguson"
                elif d['topic']=='germanwings-crash':
                    topic = "germanwings crash"
                elif d['topic'] == 'ottawashooting':
                    topic = "ottawa shooting"
                elif d['topic'] == 'sydneysiege':
                    topic = "sydneysiege"
                if not topic=="":
                    self.cursor.execute("INSERT INTO Claims VALUES (?,?,?,?);",(dataset_number, topic, d['text'], d['label']))
        self.conn.commit()

        # f = open(path, "r")
        # print(f.readline())
        # for x in f:
        #     print(x)

    def get_dataset(self, dataset_number):
        query = "SELECT Claim,Sentence,Stance FROM Claims WHERE Dataset_Number=" + str(dataset_number)
        df = pd.read_sql_query(query, self.conn)
        return df


    def delete_dataset(self, dataset_number):
        self.conn = sqlite3.connect('Stance_Detection.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute("DELETE FROM Claims WHERE Dataset_Number=?", (dataset_number,))
        self.conn.commit()

    def get_all_result(self):
        query = 'SELECT Model,Dataset,Train_percent,Accuracy FROM Result'
        # query = "SELECT * FROM Result"
        df = pd.read_sql_query(query, self.conn)
        return df

    def insert_external_dataset(self,df,number):
        PROJECT_ROOT = os.path.abspath('__file__')
        BASE_DIR = os.path.dirname(PROJECT_ROOT)
        df.to_csv(BASE_DIR+"\\DB\\ds.csv",index=False)
        with open(BASE_DIR+"\\DB\\ds.csv", "r", encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # for line in csv_reader:
            #     print(line)
            query = "INSERT INTO Claims VALUES({},?,?,?);".format(number)
            self.cursor.executemany(query, csv_reader)
            self.conn.commit()

    def get_topics_db(self):
        query = "SELECT Claim,Sentence,Stance FROM Claims WHERE Dataset_Number=" + str(1)
        df = pd.read_sql_query(query, self.conn)
        return df.Claim.unique()

    def create_model_desc_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS model_desc (model text NOT NULL, desc TEXT NOT NULL)")
        self.conn.commit()

    def insert_desc_model(self, model, desc):
        query = 'INSERT INTO model_desc VALUES(?,?);'
        self.cursor.execute(query, (model,desc))
        self.conn.commit()

    def get_model_desc_db(self,model):
        query = 'SELECT * FROM model_desc where model="{}"'.format(model)
        df = pd.read_sql_query(query, self.conn)
        return df

    def create_dataset_desc_table(self):
        self.cursor.execute("CREATE TABLE IF NOT EXISTS dataset_desc (dataset text NOT NULL, desc TEXT NOT NULL)")
        self.conn.commit()

    def insert_desc_dataset(self, dataset, desc):
        query = 'INSERT INTO dataset_desc VALUES(?,?);'
        self.cursor.execute(query, (dataset, desc))
        self.conn.commit()

    def get_dataset_desc_db(self, dataset):
        query = 'SELECT * FROM dataset_desc where dataset="{}"'.format(dataset)
        df = pd.read_sql_query(query, self.conn)
        return df


db = DataBase()
# db.create_dataset_desc_table()
# db.insert_desc_dataset('semEval2016','semEval 2016 description')
# db.drop_result()
# db.create_result_table()
# db.create_model_desc_table()
# db.create_dataset_desc_table()
# db.insert_desc_dataset('MPCHI',"This dataset contains health-related online news articles. The data provided contains instances of: tweets, id, target and stance, where stance is one of  the following: favor, against, none.")

# db.drop_Request()
# db.create_Request_table()
# print(db.insert_records_request('TRANSFORMER','10',0.6))
# db.reset_index()
# db.drop_Index()
# db.create_index_table()
# print(db.get_index())
# db.drop_Index()
# db.insert_records_to_result("TRANSFORMER","semEval2016",0.6)
# db.drop_result()
# db.get_all_result()
# db.create_result_table()
# query = 'DELETE FROM Result WHERE Model="UCLMR" AND Dataset="EmergentLite" AND Train_percent=60;'
# db.cursor.execute(query)
# db.conn.commit()
# db.create_Stance_Result_table()
# print(db.get_all_result())
# db.insert_semEveal_2017("semeval2017.txt",2)
# db.delete_dataset(10)
# db.fill_claim_table("semEval2016.csv",1   )
# print(db.get_dataset(20))
# df = db.get_dataset(1)
# print(db.get_dataset(1))
# print(df.columns)

# db.fill_claim_table("semEval2016.csv",1)
# db.fill_claim_table("FNC.csv",3)
# db.fill_claim_table("MPCHI.csv",4)
# db.fill_claim_table("EmergentLite.csv",5)
# print(db.get_dataset(4))

# db = DataBase()
# df = db.get_dataset(6)
# # print(df)
# print(df.groupby(['Stance']).count())
# labels = 'Against: 3160', 'Favor: 3961'
# sizes = [3160, 3961]
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
# ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
# plt.show()

# db = DataBase()
# db.drop_result()
# db.create_result_table()
# db.insert_records_to_result(1,1,100,30,"report","path")
# db.delete_from_result()
# df = db.get_record_from_result("UCLMR","MPCHI",80)
# print(df["roc_acc"][0])
# print(df.shape[0])
# print(df)

# accuracy = df['Accuracy'][0]
# class_report = str(df['Class_report'][0])
# class_report=class_report.replace('\n','')
# arr = class_report.split(" ")
# arr = list(filter(lambda x: len(x)>0,arr))
# photo_path = df['Cm_path'][0]
#
# print(accuracy)
# print(photo_path)
# print(arr)


#writing record to csv
# db = DataBase()
# df=db.get_record_from_result('TAN', 'EmergentLite', 60)
# import os
# PROJECT_ROOT = os.path.abspath(__file__)
# BASE_DIR = os.path.dirname(PROJECT_ROOT)
# df.to_csv(BASE_DIR+"\\myRecord.csv", index=False)
# #
#
# #read record from csv and write to db
# db = DataBase()
# import os
# PROJECT_ROOT = os.path.abspath(__file__)
# BASE_DIR = os.path.dirname(PROJECT_ROOT)
# df=pd.read_csv(BASE_DIR+"\\myRecord.csv", header=0)
# for i in range(len(df)):
#     db.insert_records_to_result(df[i][0], )
