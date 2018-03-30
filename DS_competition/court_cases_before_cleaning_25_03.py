# -*- coding: utf-8 -*-
import data_setup

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor



def get_score_from_cross_validation(model, input_train, target, all_columns_except_target, valid_split_size=5):
    train = input_train.copy(deep=True)
    items_in_valid_set = len(train) // valid_split_size
    accum_score_from_validation = 0

    for i in range(valid_split_size):
        valid_set = train[i*items_in_valid_set:(i+1)*items_in_valid_set]
        train_set = train[:i*items_in_valid_set].append(train[(i+1)*items_in_valid_set:])
        model.fit(train_set[all_columns_except_target],train_set[target])
        y_pred = model.predict(valid_set[all_columns_except_target])
        score_on_train = mean_squared_error(valid_set[target], y_pred)  
        accum_score_from_validation += score_on_train

    score_from_validation = accum_score_from_validation/valid_split_size
    return score_from_validation

def score_model(model, train, target, all_columns_except_target, model_name, model_setup="default"):
    print("Model: {}, {}".format(model_name, model_setup))
    print("Score on train data using cross-validation: {}".format(get_score_from_cross_validation(model, train, target, all_columns_except_target, valid_split_size=5)))
    print("---------------------------------")

def score_model_on_test_data(model, train_data, test_data, target, all_columns_except_target, model_name, model_setup="default"):
    model.fit(train_data[all_columns_except_target], train_data[target])
    y_pred_test = model.predict(test_data[all_columns_except_target])
    score_on_test = mean_squared_error(test_data[target], y_pred_test)  
    print("Model: {}, {}".format(model_name, model_setup))
    print("Score on test data: {}".format(score_on_test))
    print("---------------------------------")
    
def remove_items_from_list(main_list, items_to_be_removed_list):
    for item_to_remove in items_to_be_removed_list:
        if item_to_remove in main_list:
            main_list.remove(item_to_remove)
    return main_list

def from_2_lists_get_1_with_common_items(list1, list2):
    common_list = list1[:]
    for item in list1:
        if item not in list2:
            common_list.remove(item)
    return common_list

def main():
    start_date_is_used = True #change to False if start date (Ienākšanas datums) should not be used
    case_end_year_in_test_data_is_only_2018 = True #set to True if in test set all cases end in 2018, otherwise set to False
        
    #data cleaning and features engineering section
    print("Reading input files")
    raw_data = pd.read_csv('train_input.csv')
    public_data = pd.read_csv('public_data.csv')

    print("===========================================================")
    print("===============     Train data setup  =====================")
    print("===========================================================")
    
    transformed_data = data_setup.setup(raw_data, public_data, 
            start_date_is_used=start_date_is_used, 
            train_data_setup=True, 
            case_end_year_in_test_data_is_only_2018=case_end_year_in_test_data_is_only_2018)
    
    data = transformed_data.copy(deep=True)
    
    print("================   Train data setup done     ==============")
    print("===========================================================")
   
    data.to_csv("out.csv") # delete
   
    print("Splitting data")
    train, test = train_test_split(data, test_size=0.2, random_state=1)
    full_train_set = data.copy(deep=True)

    #store the name of the variable we want to predict. Separately store the names of all other variables
    target = 'duration_m'
 
    columns_to_exclude = [ 
         'start_date', 'end_date', 'court_name', 'case_id', 'court_id', 'date_of_birth',                            
         'duration_m', 'region',
        # 'reciept_procedure_case_join', # 'reciept_procedure_case_split', # 'reciept_procedure_court_belonging',
        # 'reciept_procedure_first', 
        # 'reciept_procedure_quick',
        # 'reciept_procedure_ruling_cancel',
        # 'money_amount_req_less_2',
        # 'money_amount_req_zero',
        # 'p_to_p_dummy',
        # 'p_to_b_dummy',
        # 'b_to_b_dummy',
        # 'b_to_p_dummy',
        # 'b_and_p_other_dummy',
        # 'case_matter_contract_breach',
        # 'case_matter_insurance',
        # 'case_matter_loan_recovery',
        # 'case_matter_loss_recovery',
        # 'case_matter_money_recovery',
        # 'case_matter_other',
        # 'case_matter_pay_recovery',
        # 'case_matter_property',
        # 'case_matter_rent',you
        # 'case_matter_support',
        # 'court_aizkraukles_rajona_tiesa',
        # 'court_alūksnes_rajona_tiesa',
        # 'court_bauskas_rajona_tiesa',
        # 'court_cēsu_rajona_tiesa',
        # 'court_daugavpils_tiesa',
        # 'court_dobeles_rajona_tiesa',
        # 'court_gulbenes_rajona_tiesa',
        # 'court_jelgavas_tiesa',
        # 'court_jēkabpils_rajona_tiesa',
        # 'court_kuldīgas_rajona_tiesa',
        # 'court_kurzemes_rajona_tiesa',
        # 'court_liepājas_tiesa',
        # 'court_limbažu_rajona_tiesa',
        # 'court_madonas_rajona_tiesa',
        # 'court_ogres_rajona_tiesa',
        # 'court_rēzeknes_tiesa',
        # 'court_rīgas_pilsētas_kurzemes_rajona_tiesa',
        # 'court_rīgas_pilsētas_latgales_priekšpilsētas_tiesa',
        # 'court_rīgas_pilsētas_pārdaugavas_tiesa',
        # 'court_rīgas_pilsētas_vidzemes_priekšpilsētas_tiesa',
        # 'court_rīgas_pilsētas_zemgales_priekšpilsētas_tiesa',
        # 'court_rīgas_pilsētas_ziemeļu_rajona_tiesa',
        # 'court_rīgas_rajona_tiesa',
        # 'court_rīgas_rajona_tiesa_jūrmalas_tiesu_nams',
        # 'court_saldus_rajona_tiesa',
        # 'court_talsu_rajona_tiesa',
        # 'court_tukuma_rajona_tiesa',
        # 'court_valkas_rajona_tiesa',
        # 'court_valmieras_rajona_tiesa',
        # 'court_ventspils_tiesa',
        # 'loadiness_of_court',
        # 'not_subject_to_duty_not_zero',
        # 'persons_and_compnies_answering',
        # 'lives_abroad_over_persons_and_companies_answering',
        # 'start_date_month',
        # 'start_date_quarter',
        # 'persons_and_compnies_started',
        # 'single_person_or_company_started',
        # 'single_person_or_company_answered',
        # 'court_judge_count',
        # 'court_productivity',
        # 'money_amount_requested',
        # 'person_answering_to_case',
        # 'legal_entity_answering_to_case',
        # 'person_started_the_case',
        # 'legal_entity_started_the_case',
        # 'court_meetings_set',
        # 'court_meetings_happened',
         'admin_penalty',
         'admin_penalty_not_zero',
        # 'lives_abroad',
        # 'not_subject_to_duty',
         'money_amount_from_2_to_15',
         'money_amount_from_15_to_50',
         'money_amount_from_50_to_500',
         'money_amount_from_500_to_1000',
         'money_amount_from_1000_to_2000',
         'money_amount_from_2000_to_5000',
         'money_amount_from_5000_to_50_000',
         'money_amount_from_50_000_to_1000_000',
         'money_amount_over_1000_000',
         'court_meetings_set_is_zero',
         'court_meetings_happened_is_zero',
     ]
    
    all_columns_except_target = list(train) #list with all columns in train data
    all_columns_except_target = remove_items_from_list(all_columns_except_target, columns_to_exclude)
    all_columns_except_target_train = all_columns_except_target[:]

    print("===========================================================")
    print("================  Testing the model  ======================")
    print("===========================================================")


    #-----------------------------------------------------------------------------------------------------
    #TESTING SCORES FOR DIFFERENT MODELS
    #default settings vs manually calibrated settings - RandomForestRegressor
    model = RandomForestRegressor(random_state=1, n_jobs=-1)
    model_name='RandomForestRegressor'
    score_model(model, train, target, all_columns_except_target, model_name)
    score_model_on_test_data(model, train, test, target, all_columns_except_target, model_name)
    
    model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=2, random_state=1, n_jobs=-1)
    score_model(model, train, target, all_columns_except_target, model_name, "calibrated 1")
    score_model_on_test_data(model, train, test, target, all_columns_except_target, model_name,'calibrated 1')

  
    
    print("===========================================================")
    print("===  Training the model and predicting unseen data  =======")
    print("===========================================================")
    print("Reading input file for prediction and the public data file")
    
    raw_data_to_predict = pd.read_csv('input_data_to_predict.csv')
    public_data = pd.read_csv('public_data.csv')

    print("===========================================================")
    print("===============  Preparing data  ==========================")
    print("===========================================================")
    
    transformed_data_to_predict = data_setup.setup(raw_data_to_predict, public_data, 
            start_date_is_used=start_date_is_used, 
            train_data_setup=False, 
            case_end_year_in_test_data_is_only_2018=case_end_year_in_test_data_is_only_2018)
    
    unseen_data = transformed_data_to_predict.copy(deep=True)
    
    all_columns_except_target = list(unseen_data)
    all_columns_except_target = remove_items_from_list(all_columns_except_target, columns_to_exclude)
    all_columns_except_target = from_2_lists_get_1_with_common_items(all_columns_except_target, all_columns_except_target_train)
    
    
    print("================   Data setup done     ====================")
    print("===========================================================")
    
    #training the model on the full training set and then making the prediction on the unseen data
    final_model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=2, random_state=1, n_jobs=-1)
    final_model.fit(full_train_set[all_columns_except_target],full_train_set[target])
    unseen_data['predictions'] = final_model.predict(unseen_data[all_columns_except_target])
    unseen_data.to_csv('predictions.csv')   

if __name__ == "__main__":
    main()
