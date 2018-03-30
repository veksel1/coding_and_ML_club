import pandas as pd
import numpy as np

from datetime import datetime, date
from dateutil.relativedelta import relativedelta

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

def add_money_amount_requested_categories(input_df):
    df = input_df.copy(deep=True)
    df['money_amount_from_2_to_15'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=2 and amount<15) else 0)
    df['money_amount_from_15_to_50'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=15 and amount<50) else 0)
    df['money_amount_from_50_to_500'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=50 and amount<500) else 0)
    df['money_amount_from_500_to_1000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=500 and amount<1000) else 0)
    df['money_amount_from_1000_to_2000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=1000 and amount<2000) else 0)
    df['money_amount_from_2000_to_5000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=2000 and amount<5000) else 0)
    df['money_amount_from_5000_to_50_000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=5000 and amount<50000) else 0)
    df['money_amount_from_50_000_to_1000_000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=50000 and amount<1000000) else 0)
    df['money_amount_over_1000_000'] = df.loc[:,('money_amount_requested')].apply(lambda amount: 1 if (amount>=1000000) else 0)  
    return df
    
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
    df['lives_abroad_over_persons_and_companies_answering'][df.loc[:,('persons_and_compnies_answering')]==0] = 0
    return df

def add_court_productivity(input_df):
    df = input_df.copy(deep=True)
    df['court_productivity'] = df['court_judge_count'] / df['loadiness_of_court']
    return df

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

def setup(input_df, public_data):
    data = input_df.copy(deep=True)

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
    data = add_money_amount_requested_categories(data)

    # Load public data
    data = add_public_data(data, public_data)
    # Depends on public data
    data = add_court_productivity(data)

    return data
