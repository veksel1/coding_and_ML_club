# -*- coding: utf-8 -*-
import data_setup

import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import IsolationForest # for outlier identification

import xgboost as xgb
from xgboost.sklearn import XGBRegressor

from sklearn.svm import SVR
from sklearn.svm import NuSVR

from sklearn.linear_model import LinearRegression

# It's retarded, doesn't fill the line, only prints 80 chars, uncomment for printing needs
#pd.options.display.width = 120
#pd.options.display.max_rows = 999



def get_score_from_cross_validation(model, input_train, target, all_columns_except_target, valid_split_size=5):
    train = input_train.copy(deep=True)
    items_in_valid_set = len(train) // valid_split_size
    accum_score_from_validation = 0

    for i in range(valid_split_size):
        valid_set = train[i*items_in_valid_set:(i+1)*items_in_valid_set]
        train_set = train[:i*items_in_valid_set].append(train[(i+1)*items_in_valid_set:])
        model.fit(train_set[all_columns_except_target],train_set[target])
        y_pred = model.predict(valid_set[all_columns_except_target])
        score_on_train = r2_score(valid_set[target], y_pred)  #calculating R^2 score manually
        accum_score_from_validation += score_on_train

    score_from_validation = accum_score_from_validation/valid_split_size
    return score_from_validation

def score_model(model, train, target, all_columns_except_target, model_name, model_setup="default"):
    print("Model: {}, {}".format(model_name, model_setup))
    print("Score: {}".format(get_score_from_cross_validation(model, train, target, all_columns_except_target, valid_split_size=5)))
    print("---------------------------------")



def main():
    #data cleaning and features engineering section
    print("Reading input files")
    raw_data = pd.read_csv('input.csv')
    public_data = pd.read_csv('public_data.csv')

    print("===========================================================")
    print("===============     Data setup     ========================")
    print("===========================================================")
    data = data_setup.setup(raw_data, public_data)

    print("================    Setup done     ========================")
    print("===========================================================")

    #print("Columns after data setup: ", list(data))
    data.to_csv("out.csv")


    print("Removing columns")
    data.pop('start_date')
    data.pop('end_date')
    data.pop('court_name')
    data.pop('case_id')
    data.pop('court_id')
    data.pop('date_of_birth')
    # Depends if start_date will be available in final data
    data.pop('start_date_year')

    print("Splitting data")

    train, test = train_test_split(data, test_size=0.2, random_state=1)


    #store the name of the variable we want to predict. Separately store the names of all other variables
    target = 'duration_m'
 
    columns_to_exclude = [ 
 'duration_m',
 #'region',
# 'reciept_procedure_case_join',
# 'reciept_procedure_case_split',
# 'reciept_procedure_court_belonging',
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
# 'case_matter_rent',
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
# 'admin_penalty',
# 'lives_abroad',
# 'not_subject_to_duty',
 ]
    
    all_columns_except_target = train.columns.difference(columns_to_exclude)

    print("===========================================================")
    print("===============  Models and stuff  ========================")
    print("===========================================================")

#    #-----------------------------------------------------------------------------------------------------
#    #MODEL CALIBRATION SECTION
#    #part 1 - calibrating/optimizing parameters for RandomForestRegressor
#    #tree amount calibration
#    for tree_amount in range(10,60,10):
#        model = RandomForestRegressor(n_estimators=tree_amount)
#        score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=3)
#        print('number of trees=', tree_amount)
#        print("score from cross-validation:", score_from_cross_validation)
#
#    #max_features calibration
#    for max_features in ['auto','sqrt','log2']:
#        model = RandomForestRegressor(max_features=max_features)
#        score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
#        print('max_features_type=', max_features)
#        print("score from cross-validation:", score_from_cross_validation)
#
#    #min_samples_leaf calibration
#    for min_samples_leaf in range(1,5,1):
#        model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=min_samples_leaf)
#        score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=3)
#        print('min_samples_leaf=', min_samples_leaf)
#        print("score from cross-validation:", score_from_cross_validation)

#    #part 2 - calibrating/optimizing parameters for XGBoost
#    for max_depth in range(1,9,1):
#        model = XGBRegressor(max_depth=max_depth)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "max depth = "+str(max_depth))
#        
#    for i in range(1,50,1):
#        learning_rate = i / 100
#        model = XGBRegressor(learning_rate=learning_rate)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "learning_rate = "+str(learning_rate))
#    
#    for n_estimators in range(196,202,1):
#        model = XGBRegressor(n_estimators=n_estimators)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "n_estimators = "+str(n_estimators))    
#        
#    for nthread in range(1,12):
#        model = XGBRegressor(nthread=nthread)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "nthread = "+str(nthread)) 
#
#    for gamma in range(0,20,2):
#        model = XGBRegressor(nthread=8, gamma=gamma)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "gamme = "+str(gamma))    
#        
#    for min_child_weight in range(1,12,1):
#        model = XGBRegressor(nthread=8, min_child_weight=min_child_weight)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "min_child_weight = "+str(min_child_weight))    
#
#    for max_delta_step in range(0,101,10):
#        model = XGBRegressor(nthread=8, max_delta_step=max_delta_step)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "max_delta_step = "+str(max_delta_step)) 
# 
#    for i in range(1,11,1):
#        subsample = i / 10
#        model = XGBRegressor(subsample=subsample, nthread=8)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "subsample = "+str(subsample))
#
#    for i in range(50,90,5):
#        colsample_bytree = i / 100
#        model = XGBRegressor(colsample_bytree=colsample_bytree, nthread=8)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "colsample_bytree= "+str(colsample_bytree))
#
#    for i in range(30,85,5):
#        colsample_bylevel = i / 100
#        model = XGBRegressor(colsample_bylevel=colsample_bylevel, nthread=8)
#        score_model(model, train, target, all_columns_except_target, 'XGBoost', "colsample_bylevel= "+str(colsample_bylevel))



    #-----------------------------------------------------------------------------------------------------
    #TESTING SCORES FOR DIFFERENT MODELS
    #default settings vs manually calibrated settings
    model = RandomForestRegressor()
    score_model(model, train, target, all_columns_except_target, "RandomForestRegressor")
    
    model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=2)
    score_model(model, train, target, all_columns_except_target, "RandomForestRegressor", "manually calibrated 1")

    model = RandomForestRegressor(n_estimators = 200, min_samples_leaf=1, max_features = 'sqrt')
    score_model(model, train, target, all_columns_except_target, "RandomForestRegressor", "manually calibrated 2")
    
    score_model(model, test, target, all_columns_except_target, "RandomForestRegressor",'calibrated')

#    model = RandomForestRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)
#
#    model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=2)
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('manually calibrated model:')
#    print("score for test data:", score_on_test)

    #default settings for GradientBoostingRegressor ~ a bit better than RandomForestRegressor
    model = GradientBoostingRegressor()
    score_model(model, train, target, all_columns_except_target, "GradientBoostingRegressor")

#    model = GradientBoostingRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for ADAboost - sucks!
    #model = AdaBoostRegressor()
    #score_model(model, train, target, "AdaBoostRegressor")

#    model = AdaBoostRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for ExtraTreesRegressor - sucks
    model = ExtraTreesRegressor()
    score_model(model, train, target, all_columns_except_target, "ExtraTreesRegressor")

#    model = ExtraTreesRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for BaggingRegressor ~ almost like RandomForestRegressor
    model = BaggingRegressor()
    score_model(model, train, target, all_columns_except_target, "BaggingRegressor")

#    model = BaggingRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for XGBModel ~ 1% better than GradientBoostingRegressor
    model = XGBRegressor()
    score_model(model, train, target, all_columns_except_target, "XGBModel")

    model = XGBRegressor(max_depth=2, min_child_weight=1, n_estimators=275, 
                subsample=1, colsample_bytree=1, colsample_bylevel=1,
                gamma=0, reg_alpha=0.01, reg_lambda=1, learning_rate=0.075, nthread=8)
    score_model(model, train, target, all_columns_except_target, "XGBModel",'calibrated, on train(validation) data')
    score_model(model, test, target, all_columns_except_target, "XGBModel",'calibrated, on test data')
    
    #default settings for SVR - very bad!!!!!!!
    #model = SVR()
    #score_model(model, train, target, "SVR")

#    model = SVR()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for NuSVR - very bad as well!!!
    #model = NuSVR()
    #score_model(model, train, target, "NuSVR")

#    model = NuSVR()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for LinearRegression - not too bad for such model
    model = LinearRegression()
    score_model(model, train, target, all_columns_except_target, "LinearRegression")

#    model = LinearRegression()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

#    #trying hyper-parameter searching here with GridSearchCV and RandomizedSearchCV
#    print("Columns: ", train.columns)
#    param_grid = {
#        'n_estimators': list(range(100, 2000, 100)),
#        #'max_features': ['auto', 'sqrt', 'log2', None]#,
#        #'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]
#    }

    #model = RandomForestRegressor(max_features='log2')#, oob_score=True, max_features="sqrt", min_samples_leaf=2)
    #CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    #CV.fit(data, y)
    #print(CV.best_params_)
    #model.fit(train, train_y)
    #y_oob = model.oob_prediction_
    #print(roc_auc_score(y, y_oob))
    #print("Train: ", model.score(train, train_y))
    #print("Test: ", model.score(test, test_y))

#    predictions_test = model.predict(test)
#
#    test['predicted_duration'] = predictions_test
#    test['actual_duration'] = test_y
#
#    answers = test[['predicted_duration','actual_duration']]
#    answers.to_csv('answers.csv')

if __name__ == "__main__":
    main()
