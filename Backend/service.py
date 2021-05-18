import json
from smtplib import SMTPException
from threading import Thread

import pandas
# from boto.gs.cors import CORS
from flask_cors import CORS
from flask import Flask
from flask import jsonify
from flask import request
from Backend.controller import get_dataset_name_controller, get_algorithmes_names_controller, add_dataset, \
    get_topics_controller, get_stance_controller, start_specific_model_controller, get_models_results_controller, \
    get_models_request_controller, get_models_desc_controller, get_categories_dataset_controller, \
    get_5_sen_ds_controller, get_labels_count_controller, get_dataset_desc_controller, get_topic_count_controller, \
    get_positive_negative_controller
from flask import Flask, request, Response,abort, jsonify, send_from_directory
import os

from Backend.main_model import start_Specific_Model
from Backend.util import get_num_of_records_controller
app = Flask(__name__)

# app.run(threaded=False)

CORS(app, supports_credentials=True)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/tpr_fpr',methods=['POST'])
def tpr_fpr():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]

    arr_to_split=result['tpr_fpr']
    res = json.loads(arr_to_split)

    return jsonify(res)


@app.route('/train_test_records',methods=['POST'])
def train_test_records():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]

    test_rec=result['df_test_records']
    train_rec=result['df_train_records']

    arr=[int(train_rec),int(test_rec)]

    obj={
        'series': arr
    }

    return obj



@app.route('/statisticsTable',methods=['POST'])
def statisticsTable():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]

    cm=result['Class_report']
    cm_arr=cm.split("\n")
    cm_arr = cm_arr[:-5]
    matrix=[]
    # cm_array=cm.split(" ")
    # cm_array.pop(len(cm_array)-1)
    arr_0=['Label','Precision','Recall','F-score','Support']
    matrix.append(arr_0)
    for arr in range(1,len(cm_arr)):
        import re
        sen = cm_arr[arr]
        if len(sen)>1:
            a=sen.strip()

            array_num=a.split()
            matrix.append(array_num)

    dic = {'data': matrix}
    return dic



@app.route('/get_positive_negative/<dataset>', methods=['GET'])
def get_positive_negative(dataset):
    datasets = get_dataset_name_controller()

    if not dataset in datasets:
        return jsonify("invalid dataset name"), 401

    arr=get_positive_negative_controller(dataset)

    obj={
        'series': arr,
    }
    return obj



@app.route('/get_topics_and_number/<dataset>', methods=['get'])
def get_topics_and_number(dataset):
    datasets = get_dataset_name_controller()

    if not dataset in datasets:
        return jsonify("invalid dataset name"), 401

    names_topic,num_topic=get_topic_count_controller(dataset)

    obj={
        'data': num_topic,
        'categories':names_topic
    }
    return obj




@app.route('/dataSetInfo/<dataset>',methods=['GET'])
def dataSetInfo(dataset):
    type=""
    dataset_name = get_dataset_name_controller()

    if not dataset in dataset_name:
        return jsonify("invalid dataset"), 401

    names = get_dataset_desc_controller(dataset)
    desc = names.iloc[0]['desc']
    numOfRecord=get_num_of_records_controller(dataset)


    if dataset=='covid' or dataset=='semEval2016' or dataset=='semEval2017' or dataset=='MPCHI' or  dataset=='MPQA':
        type='topic'
    else:
        type="headline"

    obj={
        'type':type,
        'datasetInfo':desc,
        'numOfRecords':numOfRecord
    }

    return obj


@app.route('/resultsModelDataset',methods=['POST'])
def resultsModelDataset():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]

    roc_acc=result['roc_acc']

    accuracy=result['Accuracy']

    dict={
        'accuracy': accuracy,
        'rocaucscore': round(roc_acc, 2)
    }

    return dict



@app.route('/labelPieChart/<dataset>',methods=['GET'])
def labelPieChart(dataset):
    datasets = get_dataset_name_controller()

    if not dataset in datasets:
        return jsonify("invalid dataset name"), 401


    arr_num, arr_labels_num = get_labels_count_controller(dataset)
    dic = {'series': arr_num, 'labels': arr_labels_num}
    return dic


#add name of dataset or file according to type=external/internal

@app.route('/five_sentences_dataset/<dataset>',methods=['GET'])
def five_sentences_dataset(dataset):
    datasets = get_dataset_name_controller()

    if not dataset in datasets:
        return jsonify("invalid dataset name"), 401


    fd_5 = get_5_sen_ds_controller(dataset)
    arr_return = []
    for i in range(5):
        obj = {

            'claim': fd_5.iloc[i]['Claim'],
            'sentence': fd_5.iloc[i]['Sentence'],
            'stance': fd_5.iloc[i]['Stance']

        }
        arr_return.append(obj)

    dic = {'tableData': arr_return}
    return dic


@app.route('/dataset_names',methods=['GET'])
def get_datasets_names():
    # startlocation = request.args.get('startlocation')
    # timeduration = int(request.args.get('timeduration'))
    # k = int(request.args.get('k'))

    names =list( get_dataset_name_controller())

    # print(jsonify(rows))
    return jsonify(names)


@app.route('/models_desc/<model>',methods=['GET'])
def get_models_desc(model):

    names = get_models_desc_controller(model)
    desc = names.iloc[0]['desc']
    # print(jsonify(rows))
    return jsonify(desc)


@app.route('/algo_names',methods=['GET'])
def get_algorithmes_names():
    names =get_algorithmes_names_controller()
    # print(jsonify(rows))
    return jsonify(names)


@app.route('/add_dataset_run_model',methods=['post'])
def add_dataset_to_db():
    # print(jsonify(rows))

    csv_data=request.files["file"]
    data=pandas.read_csv(csv_data)
    valid,error_message=isValid_df(data)
    if( valid):

        algo_names = get_algorithmes_names_controller()
        params = request.values
        array_algo_param = params['array']
        array_algo_param=array_algo_param.split(",")
        array_algo_param.pop(len(array_algo_param) - 1)
        for i in array_algo_param:
            if (not i in algo_names):
                status_code = Response(status=501)
                return jsonify("algorithmes names arn't valid"), 501

        # dataset_names = get_dataset_name_controller()
        # name = params['ds_name']
        # if not name in dataset_names:
        #     return jsonify("dataset isn't  valid"), 501
        percent = int(params['percent'])
        email = params['email']
        based=params['fileType']
        if email != '':

            # start_result(array_algo_param, name, int(params['percent']),email)
            thread = Thread(target=start_result,
                            kwargs={'array_algo_param': array_algo_param, 'name': None, 'percent': percent,'df_extenal':data,
                                    "email": email,'based':based})
            thread.start()
            return jsonify("ok"), 201
        else:
            index=start_specific_model_controller(array_algo_param, None, int(percent),data,based)
            return jsonify(index), 201
        # add_dataset(data)
        # status_code = Response(status=201)
        # return status_code
    else:
        return jsonify(error_message),501


def isValid_df(df):
    if(df.shape[1]!=3):
        error_message="the dataset does not have 3 columns"
        return False,error_message
    if(df.isnull().sum().sum()>0):
        error_message='the dataset has null values'
        return False,error_message
    try:
        col=df.iloc[0]
        col=col.index
        claim=col[0]
        sentence=col[1]
        stance=col[2]
        if not (claim=='Claim'):
            error_message = "the column name Claim is not correct,please fix the column name"
            return False,error_message
        if not (sentence=='Sentence'):
            error_message = "the column name Sentence is not correct,please fix the column name"
            return False,error_message
        if not (stance=='Stance'):
            error_message = "the column name Stance is not correct,please fix the column name"
            return False,error_message

    except:
        error_message='run time error'
        return False,error_message
    return True,""


@app.route('/ActualVSPredict',methods=['POST'])
def ActualVSPredict():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]
    actual=result['actual']
    list_actual=actual.split(",")
    list_actual.pop(len(list_actual)-1)
    list_actual2=[]
    for i in list_actual:
        list_actual2.append(int(i))
    predict=result['predict']
    list_predict=predict.split(",")
    list_predict.pop(len(list_predict)-1)
    list_predict2 = []
    for i in list_predict:
        list_predict2.append(int(i))

    dic={'Actual':list_actual2,'Predict':list_predict2}


    catagories = get_categories_dataset_controller(name)

    dic2 = {'categories': list(catagories)}
    return jsonify(dic,dic2)


@app.route('/catagories/<dataset>',methods=['GET'])
def catagories(dataset):
    datasets = get_dataset_name_controller()

    if not dataset in datasets:
        return jsonify("invalid dataset"), 401


    catagories = get_categories_dataset_controller(dataset)

    dic = {'categories': list(catagories)}
    return dic


@app.route('/confusionMatrix',methods=['POST'])
def confusionMatrix():
    algo_names = get_algorithmes_names_controller()
    params = request.values
    model = params['model']

    if (not model in algo_names):
        # status_code = Response(status=501)
        return jsonify("invalid model names"), 401

    dataset_name = get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_name:
        return jsonify("invalid dataset"), 401
    percent = int(params['percent'])/100
    result=get_models_results_controller(model,name,percent)
    result=result.iloc[0]

    cm=result['cm']

    pre_matric=[]
    cm_array=cm.split("\n")
    cm_array.pop(len(cm_array)-1)
    for arr in cm_array:
        new_l=[]
        l=arr.split(",")
        l.pop(len(l)-1)
        for num in l:
            new_l.append(num)
        pre_matric.append(new_l)

    target=result['target']
    arr_target=target.split(",")
    arr_target.pop(len(arr_target)-1)
    new_target=[]
    for t in arr_target:
        new_target.append(t)


    len_target=len(new_target)

    matrix = []
    l_true_label = [''] * (len_target + 1)
    l_true_label[0] = 'TRUE LABEL'
    l_pred_label = [''] * (len_target + 1)
    l_pred_label[0] = 'PREDICTED LABEL'
    for i in range(1, len_target + 1):
        l_pred_label[i] = new_target[i - 1]

    matrix.append(l_true_label)

    for array, ind in zip(pre_matric, new_target):
        l1 = list(array)
        l1.insert(0, ind)
        matrix.append(l1)

    matrix.append(l_pred_label)

    matrix=list(matrix)

    dic = {'data': matrix}
    return dic




@app.route('/result/<id>',methods=['GET'])
def get_results_models(id):
    array_details=[]
    req_details=get_models_request_controller(id)
    if not req_details.empty:
        try:
            models=req_details.iloc[0]['Model']
            models=models.split(" ")
            models.pop(len(models)-1)
            dataset=req_details.iloc[0]['Dataset']
            train_percent=req_details.iloc[0]['Train_percent']
            # type=req_details.iloc[0]['type']
            array_details.append(models)
            array_details.append(dataset)
            array_details.append(train_percent*100)
            # array_details.append(type)

            # for model in range(len(models)-1):
            #     df=get_models_results_controller(models[model], dataset, train_percent)
            #     data_frame=data_frame.append(df)

            # return data_frame.to_json(orient="split"),200

            return jsonify(array_details)
        except:
            return jsonify("invalid index")
    else:
        return jsonify("invalid index")


@app.route('/run_model',methods=['POST'])
def run_model():
    algo_names=get_algorithmes_names_controller()
    params = request.values
    array_algo_param= params['array']
    array_algo_param = array_algo_param.split(",")
    if len(array_algo_param)>1:
        array_algo_param.pop(len(array_algo_param)-1)
    for i in array_algo_param:
        if(not i in algo_names):
            # status_code = Response(status=501)
            return jsonify("invalid model names"), 401

    dataset_names=get_dataset_name_controller()
    name = params['ds_name']
    if not name in dataset_names:
        return jsonify("invalid dataset"), 401
    percent=int(params['percent'])
    email=params['email']
    # based = params['based']
    if email !='':
        # start_result(array_algo_param, name, int(params['percent']),email)
        thread = Thread(target=start_result, kwargs={'array_algo_param':array_algo_param,'name':name , 'percent':percent,"email":email,'based':None})
        thread.start()
        return jsonify("ok"), 201
    else:
        index=start_specific_model_controller(array_algo_param, name,percent,None,None)
        return jsonify(str(index)),201



def start_result(array_algo_param, name,percent,df_extenal,email,based):
    import time
    time.sleep(3)
    index=start_specific_model_controller(array_algo_param, name, percent,df_extenal,based)
    url='http://127.0.0.1:5000/'+index

    send_email(url, email)


def send_email(results,email):
    f = open("../../../Desktop/stanceProject - Copy/Backend/result.txt", "a")
    result = results.to_json(orient="records")
    data = json.loads(result)
    # data=json.dumps(parsed, indent=4)

    for res in data:
        f.write(str(res['Model']))
        f.write(str(res['Accuracy']))

    f.close()
    # send email to user

    # Import smtplib for the actual sending function
    # Import smtplib for the actual sending function
    import smtplib

    # Import the email modules we'll need
    # from email.message import EmailMessage
    #
    # # Open the plain text file whose name is in textfile for reading.
    # with open("result.txt") as fp:
    #     # Create a text/plain message
    #     msg = EmailMessage()
    #     msg.set_content(fp.read())
    #
    # # me == the sender's email address
    # # you == the recipient's email address
    # msg['Subject'] = ""
    # msg['From'] = "chenavra@post.bgu.ac.il"
    # msg['To'] = [email]
    #
    # # Send the message via our own SMTP server.
    # s = smtplib.SMTP('localhost',)
    # s.send_message(msg)
    # s.quit()

    import smtplib

    sender = 'chen12971@gmail.com'
    receivers = ['chen12971@gmail.com']

    message = """From: From Person <from@fromdomain.com>
    To: To Person <to@todomain.com>
    Subject: SMTP e-mail test

    This is a test e-mail message.
    """

    try:
        smtpObj = smtplib.SMTP('localhost')
        smtpObj.sendmail(sender, receivers, message)
        print
        "Successfully sent email"
    except SMTPException:
        print
        "Error: unable to send email"



@app.route('/get_topics', methods=['get'])
def get_topics():
    list_topics=get_topics_controller().tolist()
    return jsonify(list_topics)


@app.route('/get_stance/<sentence>/<topic>', methods=['get'])
def get_stance(sentence,topic):
    stance=get_stance_controller(sentence,topic)
    return jsonify(stance)


if __name__=='__main__':
    app.run(debug=True,use_reloader=False)
