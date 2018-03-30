# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

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

def fix_data_encording(df):
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
    result[result['case_matter'].str.contains('parād|aizd')==True] = 'loan_recovery'
    result[result['case_matter'].str.contains('naud|apmēr|summ|maksā')==True] = 'money_recovery'
    result[result['case_matter'].str.contains('zaud|kait|lait')==True] = 'loss_recovery'
    result[result['case_matter'].str.contains('apdro')==True] = 'insurance'
    result[result['case_matter'].str.contains('komu|dzīvoj|noma')==True] = 'rent'
    result[result['case_matter'].str.contains('morā')==True] = 'morale'
    result[result['case_matter'].str.contains('alg|darbu sam')==True] = 'pay_recovery'
    result[result['case_matter'].str.contains('līg|saist')==True] = 'contract_breach'
    result[result['case_matter'].str.contains('pab|uzturl|atbals')==True] = 'support'
    result[result['case_matter'].str.contains('īpaš')==True] = 'property'
    result[result['case_matter'].isin(['loan_recovery', 'money_recovery', 'loss_recovery', 'insurance', 'contract_breach', 'rent', 'pay_recovery', 'support', 'property']) == False] = 'other'
    result = pd.get_dummies(result, columns=['case_matter'])
    return result

def add_money_amount_indicator(df):
    result = df.copy(deep=True)
    result['money_amount_req_less_2'] = result['money_amount_requested'].apply(lambda x: 1 if x > 0 and x < 2 else 0)
    result['money_amount_req_zero'] = result['money_amount_requested'].apply(lambda x: 1 if x == 0 else 0)
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

def main():
    data = pd.read_csv('input.csv')
    data = fix_data_encording(data)
    data = name_mapping(data)
    data = get_duration(data)
    data = get_court_city_type(data)
    data = fix_leading_zeros(data)
    data = encode_receipt_procedure(data)
    data = add_money_amount_indicator(data)
    data = create_person_business_indicators(data)
    data = encode_case_matter(data)
    print(data)

if __name__ == "__main__":
    main()
