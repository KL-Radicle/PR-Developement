import numpy as np
import pandas as pd

xl = pd.ExcelFile('./PII_Destress_Python.xlsx')
df = xl.parse('PII Data')

anxiety_scores = {
    'columns': ['PROMIS_Anx_Fearful', 'PROMIS_Anx_FeltUneasy', 'PROMIS_Anx_AnythingAnxiety',
                'PROMIS_Anx_WorriesOverwhelmed'],
    'code': {
        "Always": 5,
        "Often": 4,
        "Sometimes": 3,
        "Rarely": 2,
        "Never": 1,
        'Nan': np.nan,
    },

}

stress_scores = {
    'columns': ['PROMIS_Stress_FeltOverwhelmed', 'PROMIS_Stress_ProblemsPilling', 'PROMIS_Stress_Stressed',
                'PROMIS_Stress_UnabletoManageLife'],
    'code': {''
             "Always": 4,
             "Often": 3,
             "Sometimes": 2,
             "Rarely": 1,
             "Never": 0,
             'Nan': np.nan,
             }
}

wellbeing_scores = {
    'columns': ['WHO5_CalmedRelaxed', 'WHO5_CheerfulGoodSpirits', 'WHO5_LifeFilledInterests', 'WHO5_WokeupFreshRested'],
    'code': {
        "All the time": 5,
        "Most of the time": 4,
        "More than half of the time": 3,
        "Less than half of the time": 2,
        "Some of the time": 1,
        "At no time": 0,
        'Nan': np.nan,
    }
}

sleep_scores = {
    'columns_1': ['PROMIS_SleepDis_DiffFalling', 'PROMIS_SleepDis_Problem'],
    'code_1': {
        "Very much": 5,
        "Quite a bit": 4,
        "Somewhat": 3,
        "A little bit": 2,
        "Not at all": 1,
    },
    'columns_2': ['PROMIS_SleepDis_Refreshing'],
    'code_2': {
        "Very much": 1,
        "Quite a bit": 2,
        "Somewhat": 3,
        "A little bit": 4,
        "Not at all": 5,
    },
    'columns_3': ['PROMIS_SleepDis_Quality'],
    'code_3': {
        "Very good": 5,
        "Good": 4,
        "Fair": 3,
        "Poor": 2,
        "Very poor": 1,
    },

}
pain_scores = {
    'columns': ["PEG_AveragePain", "PEG_Pain_EnjoymentLife", "PEG_Pain_GeneralActivity"]
}

df_anxiety_scores = df[anxiety_scores['columns']].replace(anxiety_scores['code']).sum(axis='columns', skipna=False)
df_stress_scores = df[stress_scores['columns']].replace(stress_scores['code']).sum(axis='columns', skipna=False)
df_wellbeing_scores = df[wellbeing_scores['columns']].replace(wellbeing_scores['code']).sum(axis='columns',
                                                                                            skipna=False)
df_pain_scores = df[pain_scores['columns']].mean(axis='columns')
df_sleep_scores1 = df[sleep_scores['columns_1']].replace(sleep_scores['code_1']).sum(axis='columns', skipna=False)
df_sleep_scores2 = df[sleep_scores['columns_2']].replace(sleep_scores['code_2']).sum(axis='columns', skipna=False)
df_sleep_scores3 = df[sleep_scores['columns_3']].replace(sleep_scores['code_3']).sum(axis='columns', skipna=False)
df_sleep_scores = df_sleep_scores1 + df_sleep_scores2 + df_sleep_scores3

df['anxiety_scores'] = df_anxiety_scores
df['stress_scores'] = df_stress_scores
df['wellbeing_scores'] = df_wellbeing_scores
df['pain_scores'] = df_pain_scores
df['sleep_scores'] = df_sleep_scores

filter = df_wellbeing_scores.isna() & df_stress_scores.isna()
df_valid = df[~filter]

df.to_csv('cleaned_data_pre_filter.csv')
df_valid.to_csv('cleaned_data.csv')
