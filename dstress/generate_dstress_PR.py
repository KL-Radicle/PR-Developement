import json
import os
import pprint
from functools import reduce

import numpy as np
from tqdm import tqdm

from ParticipantReport import get_hash
from data_exploration import *
from templates.templating_functions import generate_from_document

if __name__ == "__main__":
    CONFIG = {
        'c_email': 'RecipientEmail',
        'c_name': 'RecipientFirstName',
        'c_study_week': 'Study_Week',

    }
    # read file
    df = pd.read_csv('cleaned_data.csv')
    # process data
    # create formatters
    formatters = {}
    cohort_scores = {

    }
    for i, r in df.iterrows():
        # if r['Completer'] != 1:
        #     print("Skipping incomplete response -- shoundnt print on good data")
        #     continue
        info = {}

        name = r[CONFIG['c_name']]
        email = r[CONFIG['c_email']]

        scores = {
            'PainScore': r['pain_scores'],
            'PainCondition': r['PainCondition'] == 'Yes',
            'SleepScore': r['sleep_scores'],
            'SleepCondition': r['SleepCondition'] == 'Yes',
            'AnxietyScore': r['anxiety_scores'],
            'AnxietyCondition': r['AnxietyCondition'] == 'Yes',
            'WellbeingScore': r['wellbeing_scores'],
            'WellbeingCondition': True,
            'StressScore': r['stress_scores'],
            'StressCondition': True,
            # Stress scores and conditions??
        }

        week = int(r['Study_Week'])
        if week > 4:
            continue

        conds = [('Stress', True), ('Pain', True), ('Sleep', True), ('Wellbeing', False), ('Anxiety', True)]

        if email not in formatters:
            conditions = {}
            for c, invert in conds:
                d = {
                    'condition_valid': scores[f'{c}Condition'],
                    'invert': invert,
                    'scores_cohort': [0, 0, 0, 0, 0],
                    'scores_individual': [None, None, None, None, None],
                    'num_cohort': 0,
                }
                conditions[c] = d

            formatters[email] = {
                'first_name': name,
                'email': email,
                'conditions': conditions
            }

        # change the appropriate scores
        for c, info in formatters[email]['conditions'].items():
            if info['condition_valid']:
                formatters[email]['conditions'][c]['scores_individual'][week] = scores[f'{c}Score']

                if c not in cohort_scores:
                    print(f"reset{c} -- should only print once")
                    cohort_scores[c] = [0, 0, 0, 0, 0]
                    cohort_scores[f'{c}Count'] = [0, 0, 0, 0, 0]

                cohort_scores[c][week] += scores[f'{c}Score']
                cohort_scores[f'{c}Count'][week] += 1

    filter = {}
    # post-process
    for email in formatters.keys():
        for c, info in formatters[email]['conditions'].items():
            if info['condition_valid']:
                info['scores_cohort'] = [x / y for x, y in zip(cohort_scores[c], cohort_scores[f'{c}Count'])]

    for email, d in formatters.items():
        for c, info in d['conditions'].items():
            if info['condition_valid']:
                num_incomplete = info['scores_individual'].count(None) > 3
                # ic(info['scores_individual'].count(None), num_incomplete, info['scores_individual'], c, email )
                if num_incomplete:
                    break
        if not num_incomplete:
            filter[email] = formatters[email]

    # formatter_ex = \
    #     {
    #         'first_name': 'Kaus',
    #         'email': 'kaus@radiclescience.com',
    #         'conditions':
    #             {
    #                 'Stress': {
    #                     'condition_valid': True,
    #                     'invert': True,  # means LOWER scores are BETTER
    #                     'scores_cohort': [4, 1, 2, 3, 3, 4, 4, 5, 5, 6],
    #                     'scores_individua l': [6, 1, 2, 3, 3, 6, 7, 8, 9, 6]
    #                 },
    #
    #                 'Anxiety': {
    #                     'condition_valid': True,
    #                     'invert': True,  # means LOWER scores are BETTER
    #                     'scores_cohort': [4, 1, 2, 3],
    #                     'scores_individual': [6, 1, 2, 3]
    #                 },
    #
    #                 'Wellbeing': {
    #                     'condition_valid': True,
    #                     'invert': False,  # means HIGHER scores are BETTER
    #                     'scores_cohort': [10, 11, 12, 13],
    #                     'scores_individual': [5, 3, None, 5]
    #                 },
    #
    #                 'Sleep': {
    #                     'condition_valid': True,
    #                     'invert': True,  # means HIGHER scores are BETTER
    #                     'scores_cohort': [5, 7, 3, 8],
    #                     'scores_individual': [5, None, None, 4]
    #                 },
    #
    #                 'Pain': {
    #                     'condition_valid': True,
    #                     'invert': True,  # means HIGHER scores are BETTER
    #                     'scores_cohort': [5, 7, 3, 8],
    #                     'scores_individual': [5, 2, 3, 4]
    #                 },
    #             },
    #
    #         'primary_outcome': 'Pain',
    #         'comparison_group': 'the control',
    #         'study_size': '2323',
    #         'top_states': 'CA and TX',
    #         'top_demographic': 'Females over 40',
    #     }

    study_values = {

        'primary_outcome': 'stress',
        'comparison_group': 'the control group',
        'study_size': '2738 ',
        'top_states': 'CA and TX',
        'top_demographic': 'Females under 40'
    }

    # os.mkdir('out')

    sub_formaters = {}
    json.dump(formatters, open('formatters.json', 'w'))
    json.dump(filter, open('filter.json', 'w'))
    open('filter_pretty.json', 'w').write(pprint.pformat(json.dumps(filter)))
    for i, (k, v) in enumerate(formatters.items()):
        sub_formaters[k] = v
        if i > 150:
            break


    def generate_pr(args):
        code = get_hash(args['email'])
        args = args | study_values
        html = generate_from_document(args,
                                      doc='C:/Users/kaus/PycharmProjects/Data-Insights/Participant Report/templates/email_template.html',
                                      savepath=f'out/{code}.html'
                                      )
        return code, html


    # for args in tqdm(filter.values()):
    #     generate_pr(args)
    # break
    # for args in tqdm(sub_formaters.values()):
    #     generate_pr(args)
    res = list(map(generate_pr, filter.values()))
