# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

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

def get_court_city_type(df):
    result = df.copy(deep=True)
    result['city'] = df['court_name'].str.split().str[0]
    result['region'] = df['court_name'].str.split().str[1]
    result['region'].replace(['tiesa'], 'pilsētas', inplace=True)
    result['region'].replace(['pilsētas', 'rajona'], [0, 1], inplace=True)
    result = pd.get_dummies(result, columns=['city'])
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
               'contract_breach', 'rent', 'pay_recovery', 'support', 'property']) == False, 'case_matter'] = 'other'
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

def add_court_data(data, court_data):
    result = data.copy(deep=True)
    result.join(court_data.set_index('court_name'), on="court_name")
    return result

def add_not_subject_to_duty_not_zero(input_df):
    df = input_df.copy(deep=True)
    df['not_subject_to_duty_not_zero'] = df['not_subject_to_duty'].apply(lambda x: 1 if x > 0 else 0)
    return df

def add_lives_abroad_over_persons_and_companies_involved(input_df):
    df = input_df.copy(deep=True)
    df['persons_and_compnies_answering'] = df['person_answering_to_case'] + df['legal_entity_answering_to_case']
    df['lives_abroad_over_persons_and_companies_answering'] = 0
    df['lives_abroad_over_persons_and_companies_answering'].loc[df['legal_entity_answering_to_case']>0] = df['lives_abroad'] / df['legal_entity_answering_to_case']
    return df

def main():
    file_path = 'C:\\Users\\ilja.surikovs\\Documents\\GitHub\\coding_and_ML_club\\DS_competition\\'
    data = pd.read_csv(file_path + 'input.csv')
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
    
    court_data = pd.read_csv(file_path + 'court_data.csv')
    print("Extra data columns: ", court_data.columns)
    data = add_court_data(data, court_data)

    data.pop('start_date')
    data.pop('end_date')
    data.pop('court_name')
    data.pop('case_id')

    train, test = train_test_split(data, test_size=0.2)

    train_y = train.pop("duration_m")
    test_y = test.pop("duration_m")

    print("Columns: ", train.columns)
    param_grid = {
        'n_estimators': list(range(100, 2000, 100)),
        #'max_features': ['auto', 'sqrt', 'log2', None]#,
        #'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]
    }
    model = RandomForestRegressor(n_estimators=10)#, oob_score=True, max_features="sqrt", min_samples_leaf=2)
    #CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    #CV.fit(data, y)
    #print(CV.best_params_)
    model.fit(train, train_y)
    #y_oob = model.oob_prediction_
    #print(roc_auc_score(y, y_oob))
    print("Train: ", model.score(train, train_y))
    print("Test: ", model.score(test, test_y))

    predictions_test = model.predict(test)

    test['predicted_duration'] = predictions_test
    test['actual_duration'] = test_y

    answers = test[['predicted_duration','actual_duration']]
    answers.to_csv(file_path + 'answers.csv')

if __name__ == "__main__":
    main()