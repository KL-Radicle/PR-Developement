import pandas as pd
from icecream import ic


def read_ACES_csv(fname):
    """
    round_ranges to read ACES csv encoded with header scheme of [(e1, e2, e3, Baseline), (e1, e2, e3, Week1), ... ]
    :param fname: file to read from
    :return:
    """
    df = pd.read_csv(fname, dtype=str)
    cols = df.columns
    new_cols = []
    # convert string columns to tuples
    for c in cols:
        try:
            x = eval(c)
        except Exception as e:
            x = c
        new_cols.append(x)

    # replace cols
    df.columns = new_cols

    return df


def get_days(df,  numeric_days=False):
    """
    Helper method to get array of days that are in a valid report
    :param df:
    :return:
    """
    if numeric_days:
        assert False, 'Todo - implement, issue with data formatting'
        # return list(range(-1, len(counter.keys()) - 1))
    c = df.columns
    counter = {}
    for x in c:
        if 'Day' in x[-1]:
            if not counter.get(x[-1]):
                counter[x[-1]] = 0

    return counter.keys()


def get_weeks(df, include_baseline=False, numeric_weeks=False):
    """
    Helper method to get array of weeks that are in a valid report
    :param df:
    :return:
    """

    if numeric_weeks:
        if include_baseline: prepend = [0]
        else: prepend = []
        return prepend + [7,14,21,28]

    # c = df.columns
    # counter = {}
    # for x in c:
    #     if 'Week' in x[-1]:
    #         if not counter.get(x[-1]):  # see if week x already exists in counter
    #             counter[x[-1]] = 0
    # ret = list(counter.keys())
    ret = ['Week 0', 'Week 1', 'Week 2', 'Week 3', 'Week 4', ]
    if include_baseline:
        ret = ['Baseline'] + ret

    return ret


def get_columns_by_day(df, day):
    """
    Get all the columns (headers) for a certain day.

    :param tuple_list:
    :param day:
    :return:
    """
    if not isinstance(day, str):
        day = 'Day %d' % day

    assert day in get_days(df), 'Enter valid day ' + day

    c = df.columns
    ret = []
    for x in c:
        if day == x[-1]:
            ret.append(x)

    return ret


def get_columns_by_week(df, week):
    """
    Get all the columns (headers) for a certain week.
    :param tuple_list: [(e1, e2, e3, Baseline), (e1, e2, e3, Week1), ... ]
    :param week: week, str or int
    :return:
    """
    if not isinstance(week, str):
        week = 'Week %d' % week

    assert week in get_weeks(df, include_baseline=True), 'Enter valid week ' + week

    c = df.columns
    ret = []
    for x in c:
        if x[-1] == week:
            ret.append(x)

    return ret


def exploration_overlap_columns_week_day(df):
    """
    See what questions overlap and which are unique to each
    :param df:
    :return: overlap, unique_days, unique_weeks
    """
    days = get_columns_by_day(df, 1)
    weeks = get_columns_by_week(df, 1)
    overlap = set()
    unique_days, unique_weeks = set([x[:2] for x in days]), set([x[:2] for x in weeks])
    for cd in days:
        for cw in weeks:
            if cd[:2] == cw[:2]:
                overlap.add(cw[:2])

    return overlap, unique_days - overlap, unique_weeks - overlap


def search_columns_for_substring(df, key, axis=None):
    """
    Searches columns headers in df that match the given key (key could be a substring and is not required to be an
    explicit match) along given axis (if axis is None, all are searched).
    :param key: string to search for
    :param df:
    :return:
    """
    assert isinstance(key, str)
    if axis is not None:
        assert axis < 4
    matched_headers = []
    if axis is None:
        for c in df.columns:
            if not isinstance(c, str):
                res = any([True for x in c if key in x])
                if res:
                    matched_headers.append(c)
            elif key in c:
                matched_headers.append(c)
    else:
        for c in df.columns:
            if not isinstance(c, str):
                if key in c[axis]:
                    matched_headers.append(c)
            elif key in c:
                matched_headers.append(c)
    return matched_headers


def search_columns_for_str_strict(df, key, axis=None):
    """
    Searches columns headers in df that match the given key along given axis (if axis is None, all are searched).
    :param key: string to search for
    :param df:
    :return:
    """
    assert isinstance(key, str)
    if axis is not None:
        assert axis < 4
    matched_headers = []
    if axis is None:
        for c in df.columns:
            if not isinstance(c, str):
                res = any([True for x in c if key == x])
                if res:
                    matched_headers.append(c)
            elif key == c:
                matched_headers.append(c)
    else:
        for c in df.columns:
            if not isinstance(c, str):
                if c[axis] == key:
                    matched_headers.append(c)
            elif key == c:
                matched_headers.append(c)
    return matched_headers


def generate_tuple(e: tuple, choice, time):
    """
    Generates a tuple from two inp questions.
    :param e:
    :param choice:
    :param time:
    :return:
    """
    assert len(e) == 2, 'e = (e1, e2)'
    assert choice in ['choice', 'numerical']
    return *e, choice, time


def generate_tuple_list_all(e: tuple, df, choice=None, ):
    """
    Generates a tuple from two inp questions for all baseline, weeks and days in the study.
    """
    if choice is None: choice = 'choice'
    return [generate_tuple(e, choice, 'Baseline')] + generate_tuple_list_weeks(e, df,
                                                                               choice) + generate_tuple_list_days(e, df,
                                                                                                                  choice)


def generate_tuple_list_weeks(e: tuple, df, choice=None, include_baseline=False, numeric_weeks=False):
    """
    Generates a tuple from two inp questions for weeks in the study.
    """
    if choice is None: choice = 'choice'
    if include_baseline:
        return [generate_tuple(e, choice, 'Baseline')] + [generate_tuple(e, choice, s) for s in get_weeks(df, numeric_weeks=numeric_weeks)]
    return [generate_tuple(e, choice, s) for s in get_weeks(df, numeric_weeks=numeric_weeks)]


def generate_tuple_list_days(e: tuple, df, choice=None, numeric_days=False):
    """
    Generates a tuple from two inp questions for all days in the study.
    """
    if choice is None: choice = 'choice'
    assert len(e) == 2, 'e = (e1, e2)'
    assert choice in ['choice', 'numerical']
    return [generate_tuple(e, choice, s) for s in get_days(df, numeric_days= numeric_days)]


if __name__ == "__main__":
    df = read_ACES_csv('./out/df.csv')

    STR_WEEKS_LIST = get_weeks(df)
    STR_DAYS_LIST = get_days(df)

    ic(STR_WEEKS_LIST, STR_DAYS_LIST)
    day = get_columns_by_day(df, 12)
    day1 = get_columns_by_week(df, 2)

    ic(len(day), len(day1))
    overlap, unique_days, unique_weeks = (exploration_overlap_columns_week_day(df))
    ic(overlap, unique_days, unique_weeks)

    # df2 = df[search_columns_for_substring(df, 'choice')]
    # ic(df2.shape)
    #
    # ic((df2.isnull().sum().sum()), np.prod(df2.shape),
    #    (df2.isnull().sum().sum()) - np.prod(df2.shape))
    #
    # df2 = df[search_columns_for_substring(df, 'numerical')]
    # ic(df2.shape)
    #
    # ic((df2.isnull().sum().sum()), np.prod(df2.shape),
    #    (df2.isnull().sum().sum()) - np.prod(df2.shape))

    # ic(df[search_columns_for_str_strict(df, 'Day 1')])
    # ic(day, day1)
    # df.loc[:, day]
