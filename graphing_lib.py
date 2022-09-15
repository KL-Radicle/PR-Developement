import base64

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from icecream import ic
from matplotlib import ticker
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objs import Marker, scatter


def number_line_plot(xmin, xmax, y1, y2, savepath=False, missing_data=False, c1='bo', c2='ro'):
    """
    Used for PRs showing distribution and ticks.
    :return:
    """

    lower_limit = str(xmin)
    upper_limit = str(xmax)
    label1 = 'Your baseline score'
    label2 = 'Average baseline\n score of control group'
    # set up the figure
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_xlim(xmin - 10, xmax + 10)
    ax.set_ylim(3, 7)

    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    # draw lines
    y = 5
    height = 1

    plt.hlines(y, xmin, xmax, linestyles="dotted")
    plt.vlines(xmin, y - height / 2., y + height / 2.)
    plt.vlines(xmax, y - height / 2., y + height / 2.)

    if not missing_data:
        c1 = '#FA773E'
        # draw a point on the line
        px = y1
        plt.plot(px, y, 'o', mfc=c1, ms=15, mec='black')

        # add an arrow
        plt.annotate(label1, (px, y), xytext=(px - 1, y + 1), fontsize=14,
                     arrowprops=dict(facecolor=c1, shrink=0.1),
                     horizontalalignment='center')

    c2 = [0.2698072805139186, 0.20556745182012848, 0.5246252676659529]
    # draw a  second point on the line
    px = y2
    plt.plot(px, y, 'o', mfc=c2, ms=15, mec='black')

    # add an arrow
    plt.annotate(label2, (px, y), xytext=(px - 1, y - 2), fontsize=14,
                 arrowprops=dict(facecolor=c2, shrink=0.1),
                 horizontalalignment='center')

    # add numbers

    plt.text(xmin - 1, y, lower_limit, horizontalalignment='right', fontsize=16)
    plt.text(xmax + 1, y, upper_limit, horizontalalignment='left', fontsize=16)
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    # plt.axis('scaled')

    if savepath is not False:
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)

        plt.clf()
        plt.cla()
    else:
        plt.show(bbox_inches='tight', pad_inches=0)


def dot_plot(xmin, xmax, y1, y2):
    fig = go.Figure()
    m = scatter.Marker(color='black', size=25)
    m.symbol = 25
    m.size = 25

    fig.add_trace(go.Scatter(x=[xmin, xmax], y=[0, 0], marker=m, showlegend=False))
    fig.add_annotation(x=xmin, y=0, text=xmin, showarrow=False, font_size=20, yshift=-50)
    fig.add_annotation(x=xmax, y=0, text=xmax, showarrow=False, font_size=20, yshift=-50)

    def add_point(fig, pt, color, name, yshift):
        m = scatter.Marker(color=color, size=50, )
        m.symbol = 0
        fig.add_trace(go.Scatter(x=[pt], y=[0], marker=m, name=name))
        fig.add_annotation(x=pt, y=0, text=name, showarrow=True, font_size=16, yshift=yshift)

    add_point(fig, y1, '#cd4205', 'Your Score', 75)
    add_point(fig, y2, '#5720e6', 'Aggregate Average of Database', 150)
    # left most extreme
    # right most extreme
    # value of patient dot
    fig.show()


def patient_vs_cohort_line(patientdata: list, cohortdata: list, labels: list = None, text_fmt: dict = None,
                           save_path_html='fig.html', height=300, width=550, text_offset_y=0, text_offset_x=0,
                           text_size=12, marker_size_mult=0.75, hide_legend=False, save_s3=False, image_format='png'):
    assert 'condition' in text_fmt, 'Got graph without condition specified'
    assert 'invert' in text_fmt, 'if true, invert means a positive change is IMPROVEMENT'

    if 'xlabel' not in text_fmt: text_fmt['xlabel'] = f'{text_fmt["condition"].title()} through time'
    if 'ylabel' not in text_fmt: text_fmt['ylabel'] = f'{text_fmt["condition"].title()} score'
    if labels is None:
        labels = ['Week %s' % x for x in range(len(patientdata))]
    df = pd.DataFrame()
    patient_legend = 'Your score'
    cohort_legend = 'Control group\'s score'
    imputed_legend = 'Your missing scores'

    color_imputed_fill = 'rgba(0, 1, 0, 0.1)'
    color_patient_fill = 'rgba(250, 119, 62, 0.1)'
    color_cohort_fill = 'rgba(126, 96, 245, 0.1)'
    color_p_text = '#cd4205'
    color_c_text = '#5720e6'
    color_legend = [', '.join(x.split(',')[:-1] + [' 1)']) for x in [
        color_cohort_fill,
        color_patient_fill,
        color_imputed_fill]]

    df[('%s' % patient_legend)] = patientdata
    df[cohort_legend] = cohortdata
    df['labels'] = labels

    # ic(df)
    df_filled = df.fillna(method='ffill')
    colors = df[patient_legend].isna() & ~df_filled[patient_legend].isna()
    df = df_filled
    df[imputed_legend] = colors * df[patient_legend]
    df[imputed_legend].replace(0, np.nan, inplace=True)
    # ic(df)

    c_str_change, c_value_change = calculate_score_change(cohort_legend, df, text_fmt['invert'])
    p_str_change, p_value_change = calculate_score_change(patient_legend, df, text_fmt['invert'])

    if str(p_value_change).lower().strip() == 'nan':
        p_value_change = ''
        p_str_change = 'baseline was missing <br> so your change in score could not be calculated'
    if str(c_value_change).lower().strip() == 'nan':
        c_value_change = ''
        c_str_change = 'baseline was missing <br> so your change in score could not be calculated'

    if isinstance(p_value_change, float): p_value_change = f"by {p_value_change:2.0f} %"
    if isinstance(c_value_change, float): c_value_change = f"by {c_value_change:2.0f} %"

    text_fmt['p_change_str'] = \
        f"Your <b> {text_fmt['condition']} {p_str_change} </b> <b><i>{p_value_change}</b> </i> <br> throughout the study."
    text_fmt['c_change_str'] = \
        f"The <b> {text_fmt['condition']}</b><br> of the control <b>{c_str_change}</b> <b><i>{c_value_change}</b></i> <br> throughout the study."
    text_fmt[
        'p_change_str'] = f"<i style=\"color:{color_p_text}\">{text_fmt['p_change_str']} </i>"
    text_fmt[
        'c_change_str'] = f"<i style=\"color:{color_c_text}\">{text_fmt['c_change_str']} </i>"

    if 'baseline was missing' in text_fmt['p_change_str']:
        text_size = text_size // 1.45

    FONT_PROPS = {'size': text_size, 'family': 'Arial'}

    fig = go.Figure()

    fig = px.line(df, x="labels", y=[cohort_legend, patient_legend], markers=False,
                  color_discrete_sequence=color_legend, line_shape='spline', height=height, width=width)
    # fig = px.line(df, x="labels", y=[cohort_legend, patient_legend], markers=False,
    #               color_discrete_sequence=color_legend, line_shape='linear')

    mark_missing = scatter.Marker()
    mark_missing.size = 25 * marker_size_mult
    # score line
    # fig.add_trace(
    #     go.Scatter(x=labels, y=df[imputed_legend], marker=mark_missing, name='Your missing data', mode='markers'), )

    # filled area
    fig.add_trace(go.Scatter(x=labels, y=df[patient_legend], fill='tozeroy', mode='none', fillcolor=color_patient_fill,
                             showlegend=False, line_shape='spline'))  # fill down to xaxis
    fig.add_trace(go.Scatter(x=labels, y=df[cohort_legend], fill='tozeroy', mode='none', fillcolor=color_cohort_fill,
                             showlegend=False, line_shape='spline'), )  # fill to trace0 y

    # dashed lines
    # fig.add_trace(go.Scatter(x=labels, y=df[patient_legend], line = dict(color=', '.join(color_patient_fill.split(',')[:-1] + ['0.5)']), width=2, dash='dash'),
    #                          showlegend=False,))  # fill down to xaxis
    # fig.add_trace(go.Scatter(x=labels, y=df[cohort_legend],line = dict(color=', '.join(color_cohort_fill.split(',')[:-1] + ['0.5)']), width=2, dash='dash'),
    #                          showlegend=False, ), )  # fill to trace0 y

    part_annotation_y = sum(df[patient_legend].to_list()[-2:]) / 2
    part_annotation_x = len(patientdata) - 2
    part_annotation_y = df[patient_legend].to_list()[-2]
    # part_change =
    cohort_annotation_y = sum(df[cohort_legend].to_list()[-2:]) / 2
    cohort_annotation_x = len(patientdata) - 2
    cohort_annotation_y = df[cohort_legend].to_list()[-2]
    # cohort_annotation_y = part_annotation_y - 2

    combined_data = df[cohort_legend].to_list() + df[patient_legend].to_list()
    cmin, cmax = min(combined_data), max(combined_data)

    # offset collisions
    overlap_score = abs(
        (cohort_annotation_y - part_annotation_y) \
        / (cohort_annotation_y + part_annotation_y))

    ic(overlap_score, text_fmt['condition'],
       cohort_annotation_y,
       part_annotation_y,
       cmax - cmin,
       )

    # part_offset_x = -100 + text_offset_x
    # cohort_offset_x = -100 + text_offset_x

    part_offset_x = -text_offset_x * 1.25
    cohort_offset_x = text_offset_x

    # set y offset if the scores overlap
    if overlap_score <= 0.1:
        part_offset_y = -100 + text_offset_y
        cohort_offset_y = -text_offset_y // 2
    else:
        part_offset_y = text_offset_y
        cohort_offset_y = -text_offset_y // 2

    fig.add_annotation(x=part_annotation_x, y=part_annotation_y,
                       text=text_fmt['p_change_str'],
                       showarrow=True,
                       arrowhead=6,
                       font=FONT_PROPS,
                       font_size=text_size,
                       arrowsize=4 * marker_size_mult,
                       arrowwidth=2,
                       xanchor='right',
                       ax=part_offset_x,
                       ay=part_offset_y,
                       arrowcolor=', '.join(color_patient_fill.split(',')[:-1] + ['0.5)'])
                       # textangle=p_value_change *90,

                       # ay=part_annotation_y+1,
                       # ax=part_annotation_x
                       )

    fig.add_annotation(x=cohort_annotation_x, y=cohort_annotation_y,
                       text=text_fmt['c_change_str'],
                       showarrow=True,
                       arrowhead=6,
                       font=FONT_PROPS,
                       font_size=text_size,
                       arrowsize=4 * marker_size_mult,
                       arrowwidth=2,
                       xanchor='right',
                       ax=cohort_offset_x,
                       ay=cohort_offset_y,
                       arrowcolor=', '.join(color_cohort_fill.split(',')[:-1] + [' 0.5)'])
                       # textangle=c_value_change *90,
                       # ay=cohort_annotation_y+1,
                       # ax=cohort_annotation_x
                       )

    # change legend
    fig.update_layout(legend=dict(
        orientation="v",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=1,
        title={'text': None},
    ))

    if hide_legend:
        fig.update_layout(showlegend=False)

    # update font size
    # fig.update_layout(autosize=True)
    fig.update_layout(plot_bgcolor='#ffffff', paper_bgcolor='#ffffff', yaxis=dict(showgrid=False), )
    fig.update_layout(xaxis_title=text_fmt['xlabel'], yaxis_title=text_fmt['ylabel'])
    # fig.show()
    html = fig.to_html()
    with open(save_path_html, 'w') as f:
        f.write(html)

    if save_s3:
        s3 = boto3.resource('s3')
        bucket_name = 'radicle-production'

        image_bytes = fig.to_image(format=("%s" % image_format), engine="orca", height=height, width=width)
        # ic(image_bytes)

        s3.Bucket(bucket_name).put_object(Key=save_s3, Body=image_bytes, ContentType=f'image/{image_format}',
                                          ACL='public-read')
        # image_b64 = base64.b64encode(image_bytes).decode('ascii')
        # image_bytes = image_b64
        # html = f"data:image/%s;base64,{image_bytes}" % image_format
        # open(f'temp_dump_bytes.{image_format}', 'w').write(image_b64)

        # png_b64 = None
        # return png_b64
        # fig.write_image(f'./image.{image_format}', engine='orca')
        # html = f"{image_bytes}"
        image_link = f"https://{bucket_name}.s3.us-west-2.amazonaws.com/{save_s3}"
        return image_link

    return html


def calculate_score_change(cohort_legend, df, invert):
    if invert:
        invert = -1
    else:
        invert = 1
    cscores = df[cohort_legend].tolist()
    c_str_change = 'improved'
    try:
        c_value_change = (cscores[-1] - cscores[0]) / cscores[0] * 100
    except:
        c_value_change = 0

    if c_value_change * invert < 0:
        c_str_change = 'worsened'
        c_value_change *= -1
    elif c_value_change == 0:
        c_str_change = 'did not change'
        c_value_change = ''

    # ic(c_str_change, c_value_change)
    return c_str_change, invert * c_value_change


# number_line_plot(1, 50, 5, 12)
# number_line_plot(1, 50, 5.1, 5)
# number_line_plot(1, 50, 5, 48)

if __name__ == '__main__':
    ...
    fmt = {
        'condition': ' pain',
        'invert': True,
    }
    #
    patient_vs_cohort_line([None, None, None, 4, 6, 3], [4, 5, 4.2, 5.1, 6, 5.2], text_fmt=fmt, save_s3='test.png',
                           height=600, width=1100)
    # patient_vs_cohort_line([2, None, None, 4, 5.5, 3], [4, 5, 4.2, 5.1, 6, 5.2], text_fmt=fmt, save_s3='test.svg',
    #                        height=600, width=1100)
    # sl = 20
    # x, y = (4 * np.random.sample(sl)) * np.random.sample(sl), (np.random.sample(sl) * 5)
    # ic(x, y)
    # patient_vs_cohort_line(x, y, text_fmt=fmt)
    #
    # patient_vs_cohort_line([1, 4, 5, 5, 6], [2, 4.2, 4.7, 5, 4.9], text_fmt=fmt)
    # patient_vs_cohort_line([None, 4, 5, None, 6], [4.2, 4.7, 5, 4.9, 5], text_fmt=fmt)
    # patient_vs_cohort_line([2, None, None, 4, None], [4.2, 4.7, 5, 4.9, 50], text_fmt=fmt)
    #
    # dot_plot(0, 7, 3, 4)
    # dot_plot(1, 20, 3, 4)
    # ...
