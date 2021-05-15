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


    def delete_record_from_Request(self,id ):
        query = 'DELETE FROM Request WHERE id="{}"'.format(id)
        self.cursor.execute(query)
        self.conn.commit()

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


    def drop_model_desc(self):
        query = "DROP TABLE model_desc;"
        self.cursor.execute(query)
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

    def drop_dataset_desc(self):
        query = "DROP TABLE dataset_desc;"
        self.cursor.execute(query)
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
# db.delete_record_from_Request(11)
# db.delete_record_from_result(None,'TRANSFORMER','EmergentLite',0.6)

# print(db.get_records_from_request_by_id(11))
# db.create_Stance_Result_table()
# db.create_dataset_desc_table()
# db.insert_desc_dataset('semEval2016','semEval 2016 description')
# db.drop_result()
# db.create_result_table()
# db.drop_dataset_desc()
# db.create_dataset_desc_table()
# db.drop_model_desc()
# db.create_model_desc_table()
# db.create_dataset_desc_table()
# db.insert_desc_dataset('MPCHI',"This dataset contains health-related online news articles. The data provided contains instances of: tweets, id, target and stance, where stance is one of  the following: favor, against, none.")
# db.insert_desc_dataset('semEval2016',"This dataset was provided at the SemEval competition in 2016. The data provided contains instances of: tweets, id, target, and stance, where stance is one of  the following: for, against, none.")
# db.insert_desc_dataset('FNC',"This dataset was provided at the Fake News Chalenge (FNC-1) in 2017. The data provided contains instances of: headline, body and stance,where stance is one of  the following: unrelated, discuss, agree, disagree.")
# db.insert_desc_dataset('EmergentLite',"This dataset contains claims extracted from rumour sites and Twitter, with 300 claims and 2,595 headlines.The stance is one of the following: for, against, observing.")
# db.insert_desc_dataset("semEval2017","This dataset was provided at the SemEval competition in 2017. The data provided contains instances of: a statement, a reply tweet and a stance, where stance is one of  the following: support, deny, query (the author of the response asks for additional evidence in relation to the veracity of the rumour they are responding to) and comment (the author of the response makes their own comment without a clear contribution to assessing the veracity of the rumour they are responding to.")
# db.insert_desc_dataset("MPQA","This dataset contains political debates about several topics such as healthcare, gay rights, abortion and more and their stance towards that topic (for or against). It was taken from MPQA (Multi-Perspective Question Answering).")
# db.insert_desc_dataset("IBMDebator","This claim stance dataset includes stance annotations for claims, as well as auxiliary annotations for intermediate stance classification subtasks. They are manually identified and annotated claims from Wikipedia. ")
# db.insert_desc_dataset("VAST","VAST (VAried Stance Topics) consists of a large range of topics covering broad themes, such as politics, education, and public health. In addition, the data includes a wide range of similar expressions (e.g., ‘guns on campus’ versus ‘firearms on campus’). This variation captures how humans might realistically describe the same topic and contrasts with the lack of variation in existing datasets.")
# db.insert_desc_dataset("Procon","Procon20 contains 419 different controversial issues with 6094 samples. Each sample is a pair of a (question, argument) that is either a pro or a con. A novel stance detection dataset covering 419 different controversial issues and their related pros and cons collected by procon.org in nonpartisan format.")
# db.insert_desc_dataset("covid","This dataset contains 5379 tweew about the covid 19 with three stances : 0-against, 1-favor, 2-none")
# db.drop_Request()
# db.create_Request_table()
# print(db.insert_records_request('TRANSFORMER','10',0.6))
# db.reset_index()
# db.drop_Index()
# db.create_index_table()
# db.reset_index()

# print(db.get_index())
# db.drop_Index()
# db.drop_result()
# db.get_all_result()
# db.create_result_table()
# db.cursor.execute(query)
# db.conn.commit()
# db.insert_desc_model("TRANSFORMER","This model is called the TRANSFORMER  contains positional encoding addition into input and encoder layer which contains multihead attention layer followed by feed forward layers.")
# db.insert_desc_model("TAN","TAN - Target-specific Attention Neural Network. This method consists of two main components: a recurrent neural network (RNN) as the feature extractor for text and a fully-connected network  as the target-specific attention selector. It’s a special mechanism which drives the model to concentrate on salient parts in text with respect to a specific target. This algorithm is based on LSTM (similar to RNN).  ** Note that running this algorithm takes a long time due to its complexity.")
# db.insert_desc_model("SEN","A SVM based stance detection model using three sets of features – stance vector, textual entailment and sentiment feature.The stance vector is created on a sentence level based on an assumption that the main information present in a sentence revolves around some particular parts-of-speech. Thus these parts-of-speech are the main building blocks of the stance expressed by a sentence towards a particular claim. To identify the sentiment feature a standard sentiment analyzer given in Stanford CoreNLP Toolkit is used. For the textual entailment feature, Tensor Flow4 is used, where textual entailment is estimated using word vectorization, recurrent neural networks with LSTM and dropout as a regularization method.")
# db.insert_desc_model("UCLMR","This algorithm was created by UCL Machine  Reading (UCLMR) during Stage 1 of the Fake News Challenge (FNC-1) in 2017. It is based on a single, end-to-end system consisting of lexical  as well as similarity features passed through a multi-layer perceptron with one hidden layer. UCLMR won third place in  the FNC however out of the three best scoring teams,  UCLMR’s classifier is the simplest and easiest to understand.")
# db.insert_desc_model()





# db.create_Stance_Result_table()
# print(db.get_all_result())
# db.insert_semEveal_2017("semeval2017.txt",2)
# db.delete_dataset(10)
# db.fill_claim_table("semEval2016.csv",1)
# print(db.get_dataset(20))
# df = db.get_dataset(1)
# print(db.get_dataset(1))
# # print(df.columns)
# db.fill_claim_table("MPQA.csv",6)
# db.fill_claim_table("Procon.csv",8)

# # db.delete_dataset(8)
# db.fill_claim_table("VAST.csv",9)
# db.fill_claim_table("covid.csv",10)


# db.fill_claim_table("IBMDebator.csv",7)
# db.fill_claim_table("FNC.csv",3)
# db.fill_claim_table("MPCHI.csv",4)
# db.fill_claim_table("EmergentLite.csv",5)
# print(db.get_dataset(5))


