import base64
import hashlib
import json
import os
import random
import shutil
import sys
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from tqdm import tqdm
from uszipcode import SearchEngine

from data_exploration import *
from pytemplate import generate_html

zcdb = SearchEngine()
import plotly.express as px
import pandas as pd

import logging


def get_hash(message: str) -> str:
    """
    Convert inp "message" string into a 256-bit sha string.
    :param message:  inp string to hash.
    :return: returns 256 bit digest.
    """
    if isinstance(message, float): return np.nan
    message = message.lower()
    m = hashlib.sha256()
    m.update(message.encode('utf-8'))
    return m.hexdigest()


def generate_section_text_by_condition(condition, hash, text_scores_input_1, text_scores_input_2,
                                       text_scores_cohort, text_baseline_input):
    fmt = CONFIG['STUDY_CONDITION_TO_TEXT'][condition]
    text_baseline = fmt['text_baseline'].format(text_baseline_input)
    text_scores = fmt['text_scores'].format(text_scores_input_1=text_scores_input_1,
                                            text_scores_input_2=text_scores_input_2,
                                            text_scores_cohort=text_scores_cohort)

    baseline_text = [
        CONFIG['STUDY_CONDITION_TO_STR'][condition],
        '',
        'Your Baseline Score vs All participants',
        (CONFIG['S3_prefix']) + ('%s/%s_baseline.png' % (hash, condition)),
        text_baseline,
        'Your Score Change vs All participants',
        (CONFIG['S3_prefix']) + ('%s/%s_pvc.png' % (hash, condition)),
        text_scores
    ]

    return baseline_text


def get_columns_by_condition(df, condition, numeric_weeks=True):
    """
    Gets scores of questions for a study. TODO AUTOMATE FOR FUTURE STUDIES
    """
    assert condition in CONFIG['STUDY_CONDITIONS']
    condition_flag = CONFIG['STUDY_CONDITION_TO_SCORE'][condition]
    columns = [(condition_flag, s) for s in
               ['week 0'] + [x.lower() for x in get_weeks(df, numeric_weeks=numeric_weeks)]]
    return columns


def round_target(number, target_multiple, up=False):
    """
    Rounde number to the nearest multiple up or down.
    :param number: inp number
    :param target_multiple: number to round to a multiple of
    :return:
    """
    assert target_multiple > 0
    if number % target_multiple == 0:
        return number

    if up:
        result = number + abs((number % target_multiple) - target_multiple)
    else:
        result = number + abs((number % target_multiple) - target_multiple) - target_multiple

    return result


def round_ranges(range_input, ct):
    """/
    Round ranges out to flatter values.
    """
    range_min, range_max = range_input
    assert range_max > range_min
    if range_max - range_min < ct:
        range_min = round_target(range_min, 2, up=False)
        range_max = round_target(range_max, 2, up=True)
    else:
        range_min = round_target(range_min, 5, up=False)
        range_max = round_target(range_max, 5, up=True)
    return [range_min, range_max]


precompute = dict()


def get_condition_data_for_cohort(df, condition, brandNo):
    # check if value is already computed
    global precompute
    key = str((condition, brandNo))
    if key in precompute.keys():
        return precompute[key]
    columns = get_columns_by_condition(df, condition)

    # filter by brand
    df = df[df[CONFIG['BRAND_COL']] == brandNo]

    # compute mean
    df_cohort = df[columns]
    df_cohort_means = DataFrame(df_cohort.astype(float).mean()).transpose()
    df_cohort_minmax = [df_cohort.astype(float).min().min(), df_cohort.astype(float).max().max()]
    precompute[key] = df_cohort_means, df_cohort_minmax
    assert not df_cohort_means.isna().any().any(), 'Cohort is NAN'
    return get_condition_data_for_cohort(df, condition, brandNo)


def get_condition_data_for_patient(df, id, condition, numeric_weeks=False):
    """
    Locate the patient by id in df and find their scores for the given conditions as an array.
    :returns: one row df for patient, one row df for cohort means, boolean flag if all surveys are complete
    """
    assert condition in CONFIG['STUDY_CONDITIONS']
    assert 0 <= id < len(df)
    # columns =
    row = df.iloc[id].to_frame().transpose()
    columns = get_columns_by_condition(df, condition, numeric_weeks=numeric_weeks)
    # compute patient score and if they completed conditions
    df_patient = row[columns]
    # compute same for cohort and select appropriate option based on above day
    brand = row[CONFIG['BRAND_COL']].at[id]
    df_cohort_means, df_cohort_minmax = get_condition_data_for_cohort(df, condition, brand)
    completed_all = row['completed_all_surveys_weekly'].at[id]

    # ic(df_cohort_means, df_patient, completed_all)

    return df_patient, df_cohort_means, df_cohort_minmax, completed_all


def gen_age_chart(df, brandNo, save_path):
    '''
    Generate sex at birth pie chart given each brandno

    '''
    df_brand_subset = df.loc[df[CONFIG['BRAND_COL']] == brandNo]
    age_dict = df_brand_subset[CONFIG['AGE_COL']].value_counts().to_dict()
    age_dict_bygroup = {
        "21-30": 0,
        "31-40": 0,
        "41-50": 0,
        "51-60": 0,
        "61-70": 0,
        "Older than 70": 0
    }

    for k, v in age_dict.items():

        if 21 <= int(float(k)) <= 30:  # inches
            age_dict_bygroup["21-30"] += v

        elif 31 <= int(float(k)) <= 40:  # inches
            age_dict_bygroup["31-40"] += v

        elif 41 <= int(float(k)) <= 50:  # inches
            age_dict_bygroup["41-50"] += v

        elif 51 <= int(float(k)) <= 60:  # inches
            age_dict_bygroup["51-60"] += v

        elif 61 <= int(float(k)) <= 70:  # inches
            age_dict_bygroup["61-70"] += v

        elif int(float(k)) > 70:  # inches
            age_dict_bygroup["Older than 70"] += v

        else:
            logging.warning("Error age out of range %s" % k)
    # convert ages to percentages
    total = sum(age_dict_bygroup.values())
    age_dict_bygroup_percent = {}
    for k, v in age_dict_bygroup.items():
        age_dict_bygroup_percent[k] = (v / total * 100)
    age_dict_bygroup = age_dict_bygroup_percent

    x = list(age_dict_bygroup.keys())
    y = list(age_dict_bygroup.values())
    plot_bar(x, y, ylabel="Percentage of people in age group", yticks=True)
    plt.savefig(save_path / 'plt_age.png')
    plt.clf()
    plt.cla()

    vals = [float(x) for x in age_dict.keys()]
    val_min = max(21, min(vals))
    val_max = max([x for x in vals if 21 <= x < 90])
    val_mean = sum([x for x in vals if 21 <= x < 90]) / len([x for x in vals if 21 <= x < 90])
    return val_min, val_max, val_mean


def plot_line(x, y, title="", ylabel="", xtick_label="", save=False, legend=None, yrange=None, xticks=False):
    """
    Helper function to plot line charts
    :param x: list or np array of x values
    :param y: list or np array of y values
    :return: None
    """
    colors = ['#FA773E', CONFIG['BAR_COLOR'], '#ff91a4', '#880808', '#008000', '#DEB887']
    fig, ax = plt.subplots()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(axis='y')
    if yrange:
        plt.ylim(yrange)
        if yrange[1] - yrange[0] > 10:
            plt.yticks(np.arange(yrange[0], yrange[1] + 1, 5))
        else:
            plt.yticks(np.arange(yrange[0], yrange[1] + 1, 2))
    while len(colors) < len(y):
        colors.append("#%06x" % random.randint(0, 0xFFFFFF))

    if len(np.shape(y)) > 1:
        for k, i in enumerate(y):
            # plt.plot(x, i, color=colors[k], linewidth=3, marker='o', )
            plt.plot(x, i, color=colors[k], linewidth=3, marker='o', label=legend[k], clip_on=False)
    else:
        plt.plot(x, y, color=CONFIG['BAR_COLOR'], linewidth=3, marker='o', clip_on=False)

    plt.title(title)
    plt.ylabel(ylabel, labelpad=10)
    # plt.xlabel(xlabel, labelpad=10)
    if xticks:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_label)

    if len(np.shape(y)) > 1:
        plt.legend(loc=(1.04, 0))
    if save:
        plt.savefig(ylabel + '_line.png', dpi=1200, bbox_inches='tight')
        text_file = open(ylabel + '_line.txt', "w")
        n = text_file.write(str(base64.b64encode(open(ylabel + '_line.png', "rb").read())))
        text_file.close()
    # plt.savefig('distplt.png')


def plot_bar(x, y, title="", ylabel="", save=False, colors=False, yticks: bool = False):
    """
    Helper function to plot bar charts
    :param x: list or np array of x values
    :param y: list or np array of y values
    :return: None
    """
    fig, ax = plt.subplots()

    if colors:
        barlist = plt.bar(x, y)
        colors_list = ['#FA773E', CONFIG['BAR_COLOR'], '#ff91a4', '#880808', '#008000', '#DEB887']
        for column in range(len(y)):
            barlist[column].set_color(colors_list[column])
    else:
        plt.bar(x, y, facecolor=CONFIG['BAR_COLOR'])

    plt.title(title)
    plt.ylabel(ylabel)
    plt.axhline(y=0, linestyle='-', color='black')

    if yticks:
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%2.0f%%'))
    if save:
        plt.savefig(ylabel + '_line.png', dpi=100, bbox_inches='tight')
        text_file = open(ylabel + '_line.txt', "w")
        n = text_file.write(str(base64.b64encode(open(ylabel + '_line.png', "rb").read())))
        text_file.close()


# plt.show()
# plt.clf()
# plt.cla()


def plt_pie(df, in_dict, pie_title, thin_chart=True, cen_title=True, radicle_colors=False):
    """
    :param in_dict: Input dictionary containing feature_names and values
    :type in_dict: dict
    :param thin_chart: Toggle pie chart asthetic to thin
    :type thin_chart: bool
    :param pie_title: Pie chart title
    :type pie_title: str

    :return: none
    """

    labels = in_dict.keys()
    val = in_dict.values()

    fig, ax = plt.subplots()

    if radicle_colors:
        colors_list = [CONFIG['BAR_COLOR'], '#1c0a41', "#5bb6cf", '#DDA0DD', '#5C83F8', '#880808', "#fc9755", "#fccd9b"]

    else:
        colors_list = ['#FA773E', '#ff91a4', '#880808', '#008000', '#DEB887', '#5C83F8', '#87D9DE', '#87DEB8',
                       "#b787de", "#DADE87", "#D11141", "#00b159", "#00aedb", "#f37735", "#ffc425", "#05878a",
                       "#074e67", "#5a175d", "#67074e", "#dd9933"]

    if thin_chart:
        _, _, autotexts = plt.pie(val, labels=labels, autopct='%0.1f%%', colors=colors_list, pctdistance=0.8)

        for autotext in autotexts:
            autotext.set_color('white')

        cen_circle = plt.Circle((0, 0), 0.6, fc='white')  # Plots a cicle
        fig = plt.gcf()  # Get's current figure and assigns it to centre circle
        fig.gca().add_artist(cen_circle)  # superimpose the circle on top of the pie chart
        plt.tight_layout()

        ax.axis('equal')  # Make sure plot if circular

        if cen_title:
            ax.set_title(pie_title, y=1.0, pad=-170)
            plt.rcParams['axes.titley'] = 1.0  # y is in axes-relative coordinates.
            plt.rcParams['axes.titlepad'] = -170  # pad is in points...
        plt.show()

    else:
        plt.pie(val, labels=labels, autopct='%0.1f%%', colors=colors_list)
        plt.title(pie_title)
        plt.show()


def generate_distribution_graph(df, condition, patient, brand, pid, save_path):
    df = df[df[CONFIG['BRAND_COL']] == brand]
    cohort_basline_series = df[get_columns_by_condition(df, condition)[0]].dropna().astype(float).astype(int)

    ax = sns.histplot(cohort_basline_series, color=CONFIG['BAR_COLOR'], discrete=True, alpha=1)

    patient_baseline_col = (CONFIG['STUDY_CONDITION_TO_SCORE'][condition], 'week 0')
    patient_baseline = patient[patient_baseline_col].to_list()

    patient_score = int(float(patient_baseline[0]))

    line_pos = cohort_basline_series.mean()
    highlighted_pos = list(np.arange(cohort_basline_series.min(), cohort_basline_series.max() + 1, 1)).index(
        patient_score)

    assert highlighted_pos >= 0

    ax.patches[highlighted_pos].set_facecolor('#FA773E')
    ax.axvline(x=line_pos, color='black', ls='--')

    orange_patch = mpatches.Patch(color='#FA773E', label='Your Baseline')
    plt.legend(handles=[orange_patch])

    plt.xlabel(CONFIG['STUDY_CONDITION_TO_STR'][condition] + ' Score')
    plt.ylabel('Number of participants with score')
    plt.savefig(save_path / (condition + '_' + 'baseline.png'), bbox_inches='tight')

    plt.clf()
    plt.cla()

    # return ax


def generate_score_change_graph(df, flag, cohort_mean, cohort_minmax, condition, patient, brand, save_path):
    '''
    '''
    # ic(flag, cohort, day, patient, brand, save_path)
    if patient.isnull().values.any():
        flag = False
    if flag:  # Completed Study. Make line plot
        cohort_val = cohort_mean.squeeze().to_list()  # DF to list
        patient_val = pd.to_numeric(patient.squeeze()).tolist()  # DF to list

        assert [isinstance(x, float) for x in patient_val]
        assert [isinstance(x, float) for x in cohort_val]

        stacked_y = [patient_val, cohort_val]

        # calculating y_ranges
        y_range = round_ranges(cohort_minmax, 14)

        # Pulling xlabel names
        x_label_list = list(get_weeks(df))
        x_label_list.insert(0, 'Week 0')

        # plotting
        plot_line(x_label_list, stacked_y, ylabel=CONFIG['STUDY_CONDITION_TO_STR'][condition] + ' Score', title='',
                  legend=['You', 'All Participants'], yrange=y_range)
        plt.savefig(save_path / (condition + '_' + 'pvc.png'), bbox_inches='tight')
        plt.clf()
        plt.cla()


    else:
        cohort_val = cohort_mean.squeeze().to_list()  # DF to list
        patient_val = pd.to_numeric(patient.squeeze()).ffill().tolist()  # DF to list forward filled

        # accounting for infinity error in percentage change - Jeff's suggestion (maybe rework)
        percent_change_cohort = 100 * (cohort_val[-1] - cohort_val[0]) / cohort_val[0]
        if patient_val[0] != 0:
            percent_change_patient = 100 * (patient_val[-1] - patient_val[0]) / patient_val[0]
        else:
            percent_change_patient = 100 * (patient_val[-1] - patient_val[0]) / 1

        if percent_change_cohort == 0:
            percent_change_cohort = 1

        if percent_change_patient == 0:
            percent_change_patient = 1

        y = [percent_change_patient, percent_change_cohort]
        x = ["You", "All Participants"]

        plot_bar(x, y, ylabel="Percent change in " + CONFIG['STUDY_CONDITION_TO_STR'][condition], colors=True)
        plt.savefig(save_path / (condition + '_' + 'pvc.png'), bbox_inches='tight')
        plt.clf()
        plt.cla()


def gen_text(df, patient, cohort, condition):
    cohort_val = cohort.squeeze().to_list()  # DF to list
    patient_val = pd.to_numeric(patient.squeeze()).ffill().tolist()  # DF to list forward filled

    if patient_val[0] > cohort_val[0]:
        text_baseline_input = 'HIGHER than'

    elif patient_val[0] < cohort_val[0]:
        text_baseline_input = 'LOWER than'

    else:
        text_baseline_input = 'EQUAL TO'

    # accounting for infinity error in percentage change - Jeff's suggestion (maybe rework)
    percent_change_cohort = 100 * (cohort_val[-1] - cohort_val[0]) / cohort_val[0]
    if patient_val[0] != 0:
        percent_change_patient = 100 * (patient_val[-1] - patient_val[0]) / patient_val[0]
    else:
        percent_change_patient = 100 * (patient_val[-1] - patient_val[0]) / 1

    # Fix for Empty Graph - Jeff's Suggestion
    if percent_change_patient == 0:
        percent_change_patient = 1 * np.sign(percent_change_cohort)

    if patient_val[-1] < patient_val[0]:
        text_scores_input_1 = 'a ' + str(abs(int(percent_change_patient))) + '% DECLINE'
        if condition == 'wellbeing' or condition == 'qol':
            text_scores_input_2 = 'WORSENED'
        else:
            text_scores_input_2 = 'IMPROVED'
    elif patient_val[-1] > patient_val[0]:
        text_scores_input_1 = 'a ' + str(abs(int(percent_change_patient))) + '% INCREASE'
        if condition == 'wellbeing' or condition == 'qol':
            text_scores_input_2 = 'IMPROVED'
        else:
            text_scores_input_2 = 'WORSENED'
    else:
        text_scores_input_1 = 'NO CHANGE'
        text_scores_input_2 = 'DID NOT IMPROVE OR WORSEN'

    text_scores_cohort = str(int(abs(percent_change_cohort)))

    return text_scores_input_1, text_scores_input_2, text_scores_cohort, text_baseline_input


def generate_graph_and_text(df, patient, cohort_mean, cohort_minmax, condition, completed_study: bool, brand, pid,
                            save_path):
    """
    Wrapper function to generate plot
    """
    save_path = Path(save_path)
    generate_distribution_graph(df, condition, patient, brand, pid, save_path)
    generate_score_change_graph(df, completed_study, cohort_mean, cohort_minmax, condition, patient, brand, save_path)

    text_scores_input_1, text_scores_input_2, text_scores_cohort, text_baseline_input = gen_text(df, patient,
                                                                                                 cohort_mean,
                                                                                                 condition)

    return text_scores_input_1, text_scores_input_2, text_scores_cohort, text_baseline_input


def generate_map(df, brandNo, save_path):
    """
    Generate cloropleth map based on brand
    """
    file_path = 'map_fig.png'

    if 'map' in precompute.keys():
        fig = precompute['map']
        fig.write_image(str(save_path / file_path))  ##### Make sure we save properly
        return
    # get states based off their zip codes
    df = df[df[CONFIG['BRAND_COL']] == brandNo]

    # get states based off their zip codes
    mailing_zips = list(
        df[CONFIG['COL_ZIPCODE']])
    mailing_states = []
    for i, z in enumerate(mailing_zips):
        if len(str(z)) > 5 and type(z) == str:
            z = z[0:5]

        try:
            mailing_states.append(zcdb.by_zipcode(z).state)
        except AttributeError:
            logging.warning('Error during conversion of zipcodes, ', z)
            continue

    zip_counts = Counter(mailing_states)
    del zip_counts[None]

    # initialize the map and store it in a m object
    df_geo = pd.DataFrame.from_dict(zip_counts, orient='index').reset_index()
    df_geo = df_geo.rename({'index': 'State', 0: 'counts'}, axis=1)

    fig = px.choropleth(df_geo,
                        locations='State',
                        color='counts',
                        color_continuous_scale=[[0, "rgb(91,182,207)"],
                                                [1, 'rgb(126,96,245)']],
                        hover_name='State',
                        locationmode='USA-states',
                        labels={'counts': 'Density'},
                        scope='usa')

    # fig.show()
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0}, coloraxis_colorbar=dict(
            # title="unemployment rate",
            thicknessmode="pixels",
            lenmode="pixels",
            # yanchor="top",y=1,
            ticks="outside",
            tickvals=[1, 80, 161],
            ticktext=["Low", "Medium", "High"],
            dtick=3
        ))
    fig.update_layout(
        # geo=dict(
        #     showframe=False,
        #     showcoastlines=False,
        #     projection_type='equirectangular'
        #     ),
        annotations=[dict(
            x=1,
            y=0,
            xref='paper',
            bgcolor="#e5ecf6",
            yref='paper',
            text='No Participants',
            showarrow=False,
            borderpad=8
        )]
    )

    precompute['map'] = fig

    fig.write_image(str(save_path / file_path))  ##### Make sure we save properly


from datetime import datetime


def generate_participant_report(df, CONFIG, contacts):
    logging.basicConfig(filename=f'{CONFIG["STUDY_NAME"]}-{datetime.now().strftime("(%H-%M)-%d_%m_%Y")}.log',
                        level=logging.DEBUG)

    setup_config(CONFIG, contacts)
    output_folder_path = Path('out/')

    #### 0. PREPROCESSING IMPORTANT INFORMATION
    emails = df[CONFIG['EMAIL_COL']]
    hashes = [get_hash(x) for x in emails]
    df[CONFIG['HASH_COL']] = hashes

    ## 0.1 check duplicate hashed_emails
    all_emails = len(emails)
    emails.drop_duplicates(keep="last")
    if len(emails) != all_emails:
        warnings.warn('DUPLICATE EMAILS DETECTED IN GENERATING PR. Check Data.')
        sys.exit(-1)

    ## 0.2 create output directory and delete contents if exits
    if output_folder_path.exists() and output_folder_path.is_dir():
        shutil.rmtree(output_folder_path)
    os.mkdir(output_folder_path)
    os.mkdir((output_folder_path / 'Reports'))

    ## 0.3.1 Process participants conditions
    df_eligible_anxi = df[CONFIG['conditions'][0]].fillna(False)
    df_eligible_slee = df[CONFIG['conditions'][1]].fillna(False)
    df_eligible_pain = df[CONFIG['conditions'][2]].fillna(False)

    df_eligible_qoll = df_eligible_slee.copy()
    df_eligible_well = df_eligible_slee.copy()

    df_eligible_qoll.name = 'QOL Eligible'
    df_eligible_well.name = 'Wellbeing eligible'

    df_eligible_qoll.replace(False, True, inplace=True)
    df_eligible_well.replace(False, True, inplace=True)

    STUDY_CONDITION_TO_SURVEY_ELIGIBILITY = {
        'wellbeing': df_eligible_well,
        'qol': df_eligible_qoll,
        'anxiety': df_eligible_anxi,
        'sleep': df_eligible_slee,
        'pain': df_eligible_pain,
    }

    df_eligible_qoll['columns'] = ['QOL Eligible']
    df_eligible_well['columns'] = ['Wellbeing eligible']
    cols = [True for _ in range(len(df))]

    df_eligible_qoll['QOL Eligible'] = cols
    df_eligible_well['Wellbeing eligible'] = cols

    ## 0.3.2 Process participants as complete or incomplete responses across all days of the study
    df_completed_weekly = df[CONFIG['completed_all_surveys_weekly']]
    # df_completed_daily = df[CONFIG['completed_all_surveys_daily']]
    df_dropout = df[CONFIG['all_weeklies_without_baseline']]

    def completed_all(df, invert=False):
        """
        Adds column to df if all entries in df along the rows are true.
        """
        completed = []
        for i, r in df.fillna(False).iterrows():
            r.replace('FALSE', False, inplace=True)
            r.replace('TRUE', True, inplace=True)
            cond = all(r.values)
            if invert:
                cond = all([not bool(x) for x in r.values])
            completed.append(cond)
        df['completed'] = completed
        return df, completed

    df_completed_weekly, completed_response_weekly = completed_all(df_completed_weekly)
    df['completed_all_surveys_weekly'] = completed_response_weekly
    # df_completed_daily, completed_response_daily = completed_all(df_completed_daily)
    # df['completed_all_surveys_daily'] = completed_response_daily
    df_dropout, dropout_list = completed_all(df_dropout, invert=True)
    df['dropped_out'] = dropout_list

    #### LOGGING

    # write dataframes
    df_completed_weekly.to_csv(output_folder_path / 'df_completed_weekly.csv')
    # df_completed_daily.to_csv(output_folder_path / 'df_completed_daily.csv')
    df.to_csv(output_folder_path / 'df.csv')

    df_eligible_anxi.to_csv(output_folder_path / 'df_eligible_anxi.csv')
    df_eligible_pain.to_csv(output_folder_path / 'df_eligible_pain.csv')
    df_eligible_slee.to_csv(output_folder_path / 'df_eligible_slee.csv')
    df_eligible_well.to_csv(output_folder_path / 'df_eligible_well.csv')
    df_eligible_qoll.to_csv(output_folder_path / 'df_eligible_qoll.csv')
    df_dropout.to_csv(output_folder_path / 'df_dropout.csv')

    email_list, errors = [], []
    #### REPORT CREATION START ####
    for i, row in tqdm(df.iterrows(), total=len(df), desc='Participant Report Generation Progress'):
        try:
            if dropout_list[i]:
                # warnings.warn('Person did not fill in any weeklies and has been dropped.' + str(row))
                continue

            format_list = []
            #### 1.0 Pull participant info
            brand = row[CONFIG['BRAND_COL']]
            email = row[CONFIG['EMAIL_COL']]
            code = row[CONFIG['HASH_COL']]
            name = row[CONFIG['NAME_COL']]
            save_path = (output_folder_path / 'Reports') / (code + '/')

            #### 1.01 - Drop people that are opted out
            if contacts is not None:
                if len(contacts[contacts['Email'] == email]) < 1:
                    errors.append('Not found in contacts')
                    continue
                if contacts[contacts['Email'] == email]['Unsubscribed'].values[0] == 1:
                    logging.info('skipped - Unsubscribed: %s' % email)
                    errors.append(('Unsubscribed', email))
                    continue

            os.mkdir(save_path)
            #### 1 CREATE GENERAL INFO SECTION
            ### 1.1 compare age to cohort
            min_age, max_age, mean_age = gen_age_chart(df, brand, save_path)
            num_participants = len(df)
            ### 1.2 create map
            # TODO FIX
            generate_map(df, brand, save_path)

            # find age range in study

            gi_code = [
                'General information',
                CONFIG['STUDY_CONDITION_TO_TEXT']['general_information']['intro_text'],
                'Geographic distribution in Study',
                (CONFIG['S3_prefix']) + '%s/map_fig.png' % code,
                CONFIG['STUDY_CONDITION_TO_TEXT']['general_information']['text_gender'],
                'Age Distribution in Study',
                (CONFIG['S3_prefix']) + '%s/plt_age.png' % code,
                CONFIG['STUDY_CONDITION_TO_TEXT']['general_information']['text_age'] % (min_age, max_age, mean_age)
            ]
            format_list.append(gi_code)

            #### 2 ITERATE CONDITIONS
            for condition in CONFIG['STUDY_CONDITIONS']:
                eligible = bool(STUDY_CONDITION_TO_SURVEY_ELIGIBILITY[condition].loc[i])
                ### 2.1 compare complete vs incomplete survey
                if eligible:
                    patient, cohort_mean, cohort_minmax, flag = get_condition_data_for_patient(df, i, condition,
                                                                                               numeric_weeks=False)
                    text_scores_input_1, text_scores_input_2, text_scores_cohort, text_baseline_input = \
                        generate_graph_and_text(df, patient, cohort_mean, cohort_minmax, condition, flag, brand, i,
                                                save_path)
                    section_code = generate_section_text_by_condition(condition, code, text_scores_input_1,
                                                                      text_scores_input_2, text_scores_cohort,
                                                                      text_baseline_input)
                    format_list.append(section_code)

            email_list.append(email)

            # control specific wording
            if CONFIG['BRAND_MAP'][int(float(brand))] == 'Control Group':
                str_to_write = generate_html(sections=format_list, part_name=name, brand='our brand partners',
                                             feedback_link=CONFIG['FEEDBACK_LINK']).__str__()
            else:
                str_to_write = generate_html(sections=format_list, part_name=name,
                                             brand='our CBD brand partners',
                                             feedback_link=CONFIG['FEEDBACK_LINK']).__str__()
            text_file = open(save_path / 'index.html', "w")
            n = text_file.write(str_to_write)
            text_file.close()
            errors.append((False, None, email))

        # except Exception as e:
        except AssertionError as e:
            err = 'Error occurred in generating PR: ', e, email
            logging.warning(err)
            print(err)
            errors.append((True, e.__str__(), email))
    json.dump(email_list, open(output_folder_path / 'email_list.json', 'w'))
    json.dump(errors, open(output_folder_path / 'errors.json', 'w'))

    # TODO fix this -(if you need errors in df)
    # df['errors'] = errors
    df.to_csv(output_folder_path / 'df_output.csv')


### configs for the PR
contacts = None

CONFIG = {}


def setup_config(conf, cont):
    """
    MUST BE CALLED WITH APPROPRIATE CONFIG FILE AT START. Exports arguments into local scope of the module.
    :param conf:
    :return:
    """
    global CONFIG, contacts
    CONFIG = conf
    contacts = cont
