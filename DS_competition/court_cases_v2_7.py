# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import IsolationForest # for outlier identification

import xgboost as xgb
from xgboost.sklearn import XGBModel

from sklearn.svm import SVR
from sklearn.svm import NuSVR

from sklearn.linear_model import LinearRegression

# It's retarded, doesn't fill the line, only prints 80 chars, uncomment for printing needs
#pd.options.display.width = 120
#pd.options.display.max_rows = 999

def name_mapping(df):
    result = pd.DataFrame()
    result['case_id'] = df['Lietas ID (caseID)']
    result['court_id'] = df['Tiesas ID (courtID)']
    result['court_name'] = df['Tiesa']
    result['date_of_birth'] = df['Dzimšanas datums']
    result['reciept_procedure'] = df['Ienākšanas kārtība']
    result['start_date'] = df['Ienākšanas datums']
    result['end_date'] = df['Izskatīšanas datums']
    result['case_matter'] = df['Būtība']
    result['money_amount_requested'] = df['Prasījuma apmērs']
    result['person_answering_to_case'] = df['atb_fiz']
    result['legal_entity_answering_to_case'] = df['atb_jur']
    result['person_started_the_case'] = df['pras_fiz']
    result['legal_entity_started_the_case'] = df['pras_jur']
    result['court_meetings_set'] = df['Nozīmētās sēdes']
    result['court_meetings_happened'] = df['Notikušās sēdes']
    result['admin_penalty'] = df['naudaS_sods']
    result['lives_abroad'] = df['dzivo_arzemes']
    result['not_subject_to_duty'] = df['atbr_no_nodevas']
    return result

def fix_data_encoding(df):
    strColumns = list(df.select_dtypes(include=['object']).columns)
    for column in strColumns:
        df[column] = df[column].str.replace('', '')
    return df

def get_duration(df):
    result = df.copy(deep=True)
    result['start_date'] = pd.to_datetime(df['start_date'])
    result['end_date'] = pd.to_datetime(df['end_date'])
    result['duration_m'] = (result['end_date'] - result['start_date'])  / np.timedelta64(1, 'M')
    return result

def add_date_groups(df):
    result = df.copy(deep=True)
    result['start_date_month'] = pd.DatetimeIndex(df['start_date']).month
    result['start_date_year'] = pd.DatetimeIndex(df['start_date']).year
    result['start_date_quarter'] = pd.DatetimeIndex(df['start_date']).quarter
    return result

#TODO check this
def get_court_city_type(df):
    result = df.copy(deep=True)
    result['riga'] = df['court_name'].str.split().str[0]
    result['riga'].replace(r'^(?!.*Rīga).*$', 'other', regex=True, inplace=True)
    result['riga'].replace(['Rīgas', 'other'], [1, 0], inplace=True)
    result['region'] = df['court_name'].str.split().str[1]
    result['region'].replace(['tiesa'], 'pilsētas', inplace=True)
    result['region'].replace(['pilsētas', 'rajona'], [0, 1], inplace=True)
    return result

def fix_leading_zeros(df):
    result = df.copy(deep=True)
    result['date_of_birth'] = df['date_of_birth'].apply(str).apply(lambda x: x.zfill(6))
    return result

def encode_receipt_procedure(df):
    result = df.copy(deep=True)
    result['reciept_procedure'].replace(
        ['Pirmo reizi',
        'CPL 32.1 panta kārtībā (ātrākas izskatīšanas nodrošināšanai)',
        'Sakarā ar tiesu apvienošanu',
        'Pēc piekritības no citas tiesas',
        'Pēc lēmuma atcelšanas',
        'Lietas atdalīšana'],
        ['first',
        'quick',
        'case_join',
        'court_belonging',
        'ruling_cancel',
        'case_split'],
        inplace=True
        )
    result = pd.get_dummies(result, columns=['reciept_procedure'])
    return result

#TODO: Check if other cases can be extracted, I'm too tired for this
def encode_case_matter(df):
    result = df.copy(deep=True)
    result['case_matter'] = result['case_matter'].str.replace(',','')
    result['case_matter'] = result['case_matter'].str.lower()
    result.loc[result['case_matter'].str.contains('parād|aizd')==True, 'case_matter'] = 'loan_recovery'
    result.loc[result['case_matter'].str.contains('naud|apmēr|summ|maksā')==True, 'case_matter'] = 'money_recovery'
    result.loc[result['case_matter'].str.contains('zaud|kait|lait')==True, 'case_matter'] = 'loss_recovery'
    result.loc[result['case_matter'].str.contains('apdro')==True, 'case_matter'] = 'insurance'
    result.loc[result['case_matter'].str.contains('komu|dzīvoj|noma')==True, 'case_matter'] = 'rent'
    result.loc[result['case_matter'].str.contains('morā')==True, 'case_matter'] = 'morale'
    result.loc[result['case_matter'].str.contains('alg|darbu sam')==True, 'case_matter'] = 'pay_recovery'
    result.loc[result['case_matter'].str.contains('līg|saist')==True, 'case_matter'] = 'contract_breach'
    result.loc[result['case_matter'].str.contains('pab|uzturl|atbals')==True, 'case_matter'] = 'support'
    result.loc[result['case_matter'].str.contains('īpaš')==True, 'case_matter'] = 'property'
    result.loc[result['case_matter'].isin(['loan_recovery', 'money_recovery', 'loss_recovery', 'insurance', \
               'contract_breach', 'rent', 'pay_recovery', 'support', 'property', 'morale']) == False, 'case_matter'] = 'other'
#    result['case_matter'].unique()
    result = pd.get_dummies(result, columns=["case_matter"])
    return result

def add_money_amount_indicator(df):
    result = df.copy(deep=True)
    result['money_amount_req_less_2'] = result['money_amount_requested'].apply(lambda x: 1 if x > 0 and x < 2 else 0)
    result['money_amount_req_zero'] = result['money_amount_requested'].apply(lambda x: 1 if x == 0 else 0)
    return result

def add_loadiness_of_courts(df):
    result = df.copy(deep=True)
    result['loadiness_of_court'] = df['court_id'].apply(lambda court_id: len(df[df['court_id']==court_id]))
    return result

def create_person_business_indicators(input_df):
    df = input_df.copy(deep=True)
    df['p_to_p_dummy'] = df['person_started_the_case'].apply(lambda item: 1 if item>0 else 0)
    df['p_to_p_dummy'][(df['person_answering_to_case']==0) | \
      (df['legal_entity_answering_to_case']>0) | (df['legal_entity_started_the_case']>0)] = 0

    df['p_to_b_dummy'] = df['person_started_the_case'].apply(lambda item: 1 if item>0 else 0)
    df['p_to_b_dummy'][(df['person_answering_to_case']>0) | \
      (df['legal_entity_answering_to_case']==0) | (df['legal_entity_started_the_case']>0)] = 0

    df['b_to_b_dummy'] = df['legal_entity_started_the_case'].apply(lambda item: 1 if item>0 else 0)
    df['b_to_b_dummy'][(df['person_answering_to_case']>0) | \
      (df['legal_entity_answering_to_case']==0) | (df['person_started_the_case']>0)] = 0

    df['b_to_p_dummy'] = df['legal_entity_started_the_case'].apply(lambda item: 1 if item>0 else 0)
    df['b_to_p_dummy'][(df['person_answering_to_case']==0) | \
      (df['legal_entity_answering_to_case']>0) | (df['person_started_the_case']>0)] = 0

    df['b_and_p_other_dummy'] = df['p_to_p_dummy'] + df['p_to_b_dummy'] + df['b_to_b_dummy'] \
        + df['b_to_p_dummy'] - 1
    df['b_and_p_other_dummy'] = df['b_and_p_other_dummy'].apply(lambda item: 1 if item==-1 else 0)

    return df

def add_judge_age(df):
    result = df.copy(deep=True)
    result['judge_age'] = df['date_of_birth'].apply(str).apply(lambda x: relativedelta(date(2018, 1, 1), date(int("19" + x[4:]), int(x[2:4]), int(x[:2]))).years)
    return result

def create_court_indicators(df):
    result = df.copy(deep=True)
    result['court_clean_name'] = result['court_name'].str.lower()
    result['court_clean_name'] = result['court_clean_name'].str.replace(' ','_')
    result = pd.get_dummies(result, columns=["court_clean_name"], prefix="court")
    return result

def add_public_data(data, court_data):
    result = data.copy(deep=True)
    court_data_dict = {c: v for c, v in zip(court_data.to_dict(orient="list")['court_name'], court_data.to_dict(orient="list")['court_judge_count'])}
    result['court_judge_count'] = result['court_name'].apply(lambda crt_name: int(court_data_dict[crt_name]) if crt_name in court_data_dict else int(np.mean(list(court_data_dict.values()))))
    return result

def add_not_subject_to_duty_not_zero(input_df):
    df = input_df.copy(deep=True)
    df['not_subject_to_duty_not_zero'] = df['not_subject_to_duty'].apply(lambda x: 1 if x > 0 else 0)
    return df

def add_lives_abroad_over_persons_and_companies_involved(input_df):
    df = input_df.copy(deep=True)
    df['persons_and_compnies_answering'] = df['person_answering_to_case'] + df['legal_entity_answering_to_case']
    df['lives_abroad_over_persons_and_companies_answering'] = df['lives_abroad'] / df['persons_and_compnies_answering']
    df['lives_abroad_over_persons_and_companies_answering'][df['persons_and_compnies_answering']==0] = 0
    return df

def add_court_productivity(input_df):
    df = input_df.copy(deep=True)
    df['court_productivity'] = df['loadiness_of_court'] / df['court_judge_count']
    return df

def get_score_from_cross_validation(model, input_train, target, valid_split_size=5):
    train = input_train.copy(deep=True)
    items_in_valid_set = len(train) // valid_split_size
    accum_score_from_validation = 0
    all_columns_except_target = train.columns.difference([target])
    
    for i in range(valid_split_size):
        valid_set = train[i*items_in_valid_set:(i+1)*items_in_valid_set]
        train_set = train[:i*items_in_valid_set].append(train[(i+1)*items_in_valid_set:])
        model.fit(train_set[all_columns_except_target],train_set[target])
        y_pred = model.predict(valid_set[all_columns_except_target])
        score_on_train = r2_score(valid_set[target], y_pred)  #calculating R^2 score manually
        accum_score_from_validation += score_on_train
        
    score_from_validation = accum_score_from_validation/valid_split_size    
    return score_from_validation

def remove_outliers(input_df, target='duration_m'):
    df = input_df.copy(deep=True)
    output_df = df[df[target] > 0]
    return output_df

def get_total_persons_and_companies_started(input_df):
    df = input_df.copy(deep=True)
    df['persons_and_compnies_started'] = df['person_started_the_case'] + df['legal_entity_started_the_case']
    return df

def add_single_person_or_company_started(input_df):
    df = input_df.copy(deep=True)
    df['single_person_or_company_started'] = df['persons_and_compnies_started'].apply(lambda x: 1 if x==1 else 0)
    return df

def add_single_person_or_company_answered(input_df):
    df = input_df.copy(deep=True)
    df['single_person_or_company_answered'] = df['persons_and_compnies_answering'].apply(lambda x: 1 if x==1 else 0)
    return df    

def main():
    #data cleaning and features engineering section
    data = pd.read_csv('input.csv')
    data = fix_data_encoding(data)
    data = name_mapping(data)
    data = get_duration(data)
    data = get_court_city_type(data)
    data = fix_leading_zeros(data)
    data = add_judge_age(data)
    data = encode_receipt_procedure(data)
    data = add_money_amount_indicator(data)
    data = create_person_business_indicators(data)
    data = encode_case_matter(data)
    data = create_court_indicators(data)
    data = add_loadiness_of_courts(data)
    data = add_not_subject_to_duty_not_zero(data)
    data = add_lives_abroad_over_persons_and_companies_involved(data)
    data = add_date_groups(data)
    data = get_total_persons_and_companies_started(data)
    data = remove_outliers(data)
    data = add_single_person_or_company_started(data)
    data = add_single_person_or_company_answered(data)
    

    public_data = pd.read_csv('public_data.csv')
    print("Public data columns: ", list(public_data))
    data = add_public_data(data, public_data)
    print("After adding data: ", list(data))

    data = add_court_productivity(data)

    data.pop('start_date')
    data.pop('end_date')
    data.pop('court_name')
    data.pop('case_id')
    data.pop('court_id')
    data.pop('date_of_birth')

    # Depends if start_date will be available in final data
    data.pop('start_date_year')

    data.to_csv("out.csv")

    train, test = train_test_split(data, test_size=0.2, random_seed=1)


    #store the name of the variable we want to predict. Separately store the names of all other variables
    target = 'duration_m'
    all_columns_except_target = train.columns.difference([target])

#    #-----------------------------------------------------------------------------------------------------
#    #model calibration section
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

    #default settings vs manually calibrated settings
    print('RadnomForestRegressor from scikit-learn')
    model = RandomForestRegressor()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

    model = RandomForestRegressor(n_estimators = 60, min_samples_leaf=2)
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('manually calibrated model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

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

    #-----------------------------------------------------------------------------------------------------
    #trying different models/algorithms
    #default settings for GradientBoostingRegressor ~ a bit better than RandomForestRegressor
    print('GradientBoostingRegressor from scikit-learn')
    model = GradientBoostingRegressor()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = GradientBoostingRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for ADAboost - sucks!
    print('AdaBoostRegressor from scikit-learn')
    model = AdaBoostRegressor()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = AdaBoostRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for ExtraTreesRegressor - sucks
    print('ExtraTreesRegressor from scikit-learn')
    model = ExtraTreesRegressor()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = ExtraTreesRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for BaggingRegressor ~ almost like RandomForestRegressor
    print('BaggingRegressor from scikit-learn')
    model = BaggingRegressor()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = BaggingRegressor()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for XGBModel ~ 1% better than GradientBoostingRegressor
    print('XGBModel for scikit-learn')
    model = XGBModel()
    model.fit(train[all_columns_except_target], train[target]) 
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)  #calculating R^2 score manually
    print('default model:')
    print("score for test data:", score_from_cross_validation) 

    #default settings for SVR - very bad!!!!!!!
    print('SVR from scikit-learn')
    model = SVR()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = SVR()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for NuSVR - very bad as well!!!
    print('NuSVR from scikit-learn')
    model = NuSVR()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

#    model = NuSVR()
#    model.fit(train[all_columns_except_target], train[target])
#    score_on_test = model.score(test[all_columns_except_target], test[target])
#    print('default model:')
#    print("score for test data:", score_on_test)

    #default settings for LinearRegression - not too bad for such model
    print('LinearRegression from scikit-learn')
    model = LinearRegression()
    score_from_cross_validation = get_score_from_cross_validation(model, train, target, valid_split_size=5)
    print('default model:')
    print("score from cross-validation on train data:", score_from_cross_validation)

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
