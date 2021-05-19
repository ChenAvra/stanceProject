from Backend.util import *

def test_get_model_result_main():
    result = get_model_result_main("TRANSFORMER", "semEval2016", 0.6)
    assert isinstance(result, DataFrame)

def test_get_req_details_main():
    result = get_req_details_main(1)
    assert isinstance(result, DataFrame)

def test_get_models_desc_controller_main1():
    result = get_models_desc_controller_main("UCLMR")
    assert isinstance(result, DataFrame)

def test_get_models_desc_controller_main2():
    result = get_models_desc_controller_main("UCLMR")
    assert isinstance(result["desc"][0], str)

def test_get_categories_dataset_main():
    result = get_categories_dataset_main("semEval2016")
    assert len(result) == 3

def test_get_5_sen_ds_main2():
    result = get_5_sen_ds_main("semEval2016")
    assert isinstance(result, DataFrame)

def test_get_5_sen_ds_main2():
    result = get_5_sen_ds_main("semEval2016")
    assert len(result) == 5

def test_get_dataset_desc_controller_main1():
    result = get_dataset_desc_controller_main("semEval2016")
    assert isinstance(result, DataFrame)

def test_get_dataset_desc_controller_main2():
    result = get_dataset_desc_controller_main("semEval2016")
    assert isinstance(result["desc"][0], str)

def test_get_num_of_records_controller():
    num = get_num_of_records_controller("semEval2016")
    assert num == 4063