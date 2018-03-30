import pandas as pd
import numpy as np
pd.options.display.width=160


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


def get_duration(df):
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['duration_m'] = (df['end_date'] - df['start_date'])  / np.timedelta64(1, 'M')


def get_court_city_type(df):
    df['city'] = df['court_name'].str.split().str[0]
    df['region'] = df['court_name'].str.split().str[1]
    df['region'].replace(['tiesa'], 'pilsētas', inplace=True)

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
    
    

def main():
    data = pd.read_csv('input.csv')
    clean_data = name_mapping(data)
    get_duration(clean_data)
    get_court_city_type(clean_data)
    create_person_business_indicators(clean_data)
    
    feature_list_for_print = ['person_started_the_case','person_answering_to_case' \
      ,'legal_entity_started_the_case','legal_entity_answering_to_case','p_to_p_dummy'\
      ,'p_to_b_dummy','b_to_b_dummy','b_to_p_dummy','b_and_p_other_dummy']
    print(clean_data[feature_list_for_print])
    data_for_csv = clean_data[feature_list_for_print]
    data_for_csv.to_csv('temp_output.csv')
    #print(clean_data[['court_name', 'city', 'region', 'start_date', 'end_date', 'duration_m']])

if __name__ == "__main__":
    main()
