from Backend.main_model import start_Specific_Model
from Backend.service import get_models_request_controller

def test_start_Specific_Model():
    models = list()
    models.append("UCLMR")
    id = start_Specific_Model(models, "semEval2016", 60, None, None)
    req_details = get_models_request_controller(id)
    m = req_details.iloc[0]['Model']
    m = m.split(" ")
    m.pop(len(m) - 1)
    dataset = req_details.iloc[0]['Dataset']
    train_percent = req_details.iloc[0]['Train_percent']
    assert (dataset == "semEval2016") and (m[0] == models[0]) and (train_percent == 0.6)

