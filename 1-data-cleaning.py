import numpy as np
import pandas as pd

D = 2
R_AD = 1
OK = 0
EMPTY = 'Empty'


def compute_outcome(row):
    if row['death.within.28.days'] == 1:
        return D  # D_1_MONTH
    elif row['death.within.3.months'] == 1:
        return D  # D_3_MONTH
    elif row['death.within.6.months'] == 1:
        return D  # D_6_MONTH
    elif row['re.admission.within.28.days'] == 1:
        return R_AD  # R_1_MONTH
    elif row['re.admission.within.3.months'] == 1:
        return R_AD  # R_3_MONTH
    elif row['re.admission.within.6.months'] == 1 or row['return.to.emergency.department.within.6.months'] == 1:
        return R_AD  # R_6_MONTH
    elif row['re.admission.time..days.from.admission.'] > 180:
        return R_AD  # R_6_MONTH_PLUS
    else:
        return OK


def create_outcome_column(df):
    df['Outcome'] = df.apply(compute_outcome, axis=1)
    return df


def remove_rows_with_missing_values(df):
    df = df.dropna()
    df = df.reset_index(drop=True)

    return df


def drop_columns(df, list_of_columns_to_drop):
    for item in list_of_columns_to_drop:
        df.pop(item)

    return df


def encoding_categorical_columns(data):
    # Handle gender
    ge_map = {
        'Male': 0,
        'Female': 1
    }
    data['gender'] = data['gender'].map(ge_map)

    # Handle type.of.heart.failure
    hf_map = {
        'Right': 1,
        'Left': 0,
        'Both': 2
    }
    data['type.of.heart.failure'] = data['type.of.heart.failure'].map(hf_map)

    # Handle NYHA.cardiac.function.classification
    cc_map = {
        'II': 0,
        'III': 1,
        'IV': 2
    }
    data['NYHA.cardiac.function.classification'] = data['NYHA.cardiac.function.classification'].map(cc_map)

    # Handle Killip.grade
    kg_map = {
        'I': 0,
        'II': 1,
        'III': 2,
        'IV': 3
    }
    data['Killip.grade'] = data['Killip.grade'].map(kg_map)

    # Handle type.II.respiratory.failure
    rf_map = {
        'NonTypeII': 0,
        'TypeII': 1
    }
    data['type.II.respiratory.failure'] = data['type.II.respiratory.failure'].map(rf_map)

    # Handle consciousness
    co_map = {
        'Clear': 0,
        'ResponsiveToSound': 1,
        'ResponsiveToPain': 2,
        'Nonresponsive': 3
    }
    data['consciousness'] = data['consciousness'].map(co_map)

    # Handle respiratory.support.
    # IMV: invasive mask ventilation and NIMV: Non-invasive mask ventilation
    rs_map = {
        'None': 0,
        'IMV': 2,
        'NIMV': 1
    }
    data['respiratory.support.'] = data['respiratory.support.'].map(rs_map)

    # Handle oxygen.inhalation
    oi_map = {
        'OxygenTherapy': 1,
        'AmbientAir': 0
    }
    data['oxygen.inhalation'] = data['oxygen.inhalation'].map(oi_map)

    # Handle ageCat
    # Value is assign to the mean of age range
    ac_map = {
        '(69,79]': 74.0,
        '(79,89]': 84.0,
        '(59,69]': 64.0,
        '(49,59]': 54.0,
        '(89,110]': 99.5,
        '(39,49]': 44.0,
        '(29,39]': 34.0,
        '(21,29]': 25.0
    }
    data['ageCat'] = data['ageCat'].map(ac_map).astype(float)
    return data


def remove_large_missing_attributes(data):
    data_missing = data.isna().sum()
    req_info = 100
    l_attrs = []
    for index, value in data_missing.items():
        if value >= req_info:
            l_attrs.append(index)
    data = data.drop(l_attrs, axis=1)
    return data


def main():
    df = pd.read_csv('datasets/dat.csv')

    df['time.of.death..days.from.admission.'].replace(np.nan, 0, inplace=True)
    df['re.admission.time..days.from.admission.'].replace(np.nan, 0, inplace=True)
    df['return.to.emergency.department.within.6.months'].replace(np.nan, 0, inplace=True)

    # create the outcome column
    df = create_outcome_column(df)

    # Drop columns now redundant because of outcome
    list_outcome_column = [
        'outcome.during.hospitalization',
        'death.within.28.days',
        're.admission.within.28.days',
        'death.within.3.months',
        're.admission.within.3.months',
        'death.within.6.months',
        're.admission.within.6.months',
        'time.of.death..days.from.admission.',
        're.admission.time..days.from.admission.',
        'return.to.emergency.department.within.6.months',
        'time.to.emergency.department.within.6.months',
    ]
    df = drop_columns(df, list_outcome_column)

    # Remove all columns with too much missing values
    df = remove_large_missing_attributes(df)

    # Now remove all rows that are missing values
    df = remove_rows_with_missing_values(df)

    # Dropping columns manually that we know will not be relevant
    list_of_columns_remove_manually = [
        'Unnamed: 0',
        'inpatient.number',
        'DestinationDischarge',
        'admission.ward',
        'admission.way',
        'discharge.department',
        'visit.times',
        'occupation',
        'dischargeDay',
    ]
    df = drop_columns(df, list_of_columns_remove_manually)

    # Encoding the categorical columns
    df = encoding_categorical_columns(df)

    # Create a cleaned data
    df.to_csv('datasets/dat_cleaned.csv')


if __name__ == '__main__':
    main()
