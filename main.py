import numpy as np
import pandas as pd

# Change pandas option
pd.set_option('display.max_rows', None)
# Load cvs data
df = pd.read_csv('datasets/dat.csv')
# Visualize data
# TODO
# Remove first columns
first_column = df.columns[0]
df = df.drop([first_column], axis=1)
# Show dataset information
# df.info()
# Show missing attributes missing data
# print(df.isna().sum())
# Create a unify outcome attribute
o_attrs = ['outcome.during.hospitalization', 'death.within.28.days', 're.admission.within.28.days',
           'death.within.3.months', 're.admission.within.3.months', 'death.within.6.months',
           're.admission.within.6.months', 'return.to.emergency.department.within.6.months']
numeric_outcome_attrs = ['time.of.death..days.from.admission.', 're.admission.time..days.from.admission.',
                         'time.to.emergency.department.within.6.months']
# 0: alive; 1: dead
dead = []
for idx in range(len(df)):
    patient = df.loc[idx]
    if patient[o_attrs[0]] == 'Dead' or patient[o_attrs[3]] == 1 or patient[o_attrs[5]] == 1:
        dead.append(1)
    else:
        dead.append(0)
# Remove numerics attributes, add unified outcome
df = df.drop(numeric_outcome_attrs, axis=1)
df['is_dead'] = dead
# Remove attributes with less than half available information
data_missing = df.isna().sum()
req_info = int(len(df)) / 2
l_attrs = []
for index, value in data_missing.items():
    if value >= req_info:
        l_attrs.append(index)
df = df.drop(l_attrs, axis=1)
# Remove unnecessary attributes
u_attrs = ['inpatient.number', 'DestinationDischarge', 'admission.ward', 'occupation', 'discharge.department']
df = df.drop(u_attrs, axis=1)
# Change category to numerical values
# print('Check data type:')
# print(df.dtypes)
c_attrs = ['admission.way', 'gender', 'type.of.heart.failure',
           'NYHA.cardiac.function.classification', 'Killip.grade', 'type.II.respiratory.failure',
           'consciousness', 'respiratory.support.', 'oxygen.inhalation', 'ageCat']
# Handle admission.way
ad = c_attrs[0]
ad_map = {
    'NonEmergency': 0,
    'Emergency': 1
}
df[ad] = df[ad].map(ad_map)
# print(df[ad].value_counts())

# Handle gender
ge = c_attrs[1]
ge_map = {
    'Male': 0,
    'Female': 1
}
df[ge] = df[ge].map(ge_map)
# print(df[g].value_counts())

# Handle type.of.heart.failure
hf = c_attrs[2]
hf_map = {
    'Right': 0,
    'Left': 1,
    'Both': 2
}
df[hf] = df[hf].map(hf_map)
# print(df[hf].value_counts())

# Handle NYHA.cardiac.function.classification
cc = c_attrs[3]
cc_map = {
    'II': 2,
    'III': 3,
    'IV': 4
}
df[cc] = df[cc].map(cc_map)
# print(df[cc].value_counts())

# Handle Killip.grade
kg = c_attrs[4]
kg_map = {
    'I': 1,
    'II': 2,
    'III': 3,
    'IV': 4
}
df[kg] = df[kg].map(kg_map)
# print(df[kg].value_counts())

# Handle type.II.respiratory.failure
rf = c_attrs[5]
rf_map = {
    'NonTypeII': 0,
    'TypeII': 1
}
df[rf] = df[rf].map(rf_map)
# print(df[rf].value_counts())

# Handle consciousness
co = c_attrs[6]
co_map = {
    'Clear': 0,
    'ResponsiveToSound': 1,
    'ResponsiveToPain': 2,
    'Nonresponsive': 3
}
df[co] = df[co].map(co_map)
# print(df[co].value_counts())

# Handle respiratory.support.
rs = c_attrs[7]
rs_map = {
    'None': 0,
    'IMV': 1,
    'NIMV': 2
}
df[rs] = df[rs].map(rs_map)
# print(df[rs].value_counts())

# Handle oxygen.inhalation
oi = c_attrs[8]
oi_map = {
    'OxygenTherapy': 0,
    'AmbientAir': 1
}
df[oi] = df[oi].map(oi_map)
# print(df[oi].value_counts())

# Handle ageCat
ac = c_attrs[9]
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
df[ac] = df[ac].map(ac_map).astype(float)
# print(df[ac].value_counts())
print('After change value to numerical')
print(df.dtypes)
# Fill in missing values
missing_attrs = []
for index, value in df.isna().sum().items():
    if value > 0:
        missing_attrs.append(index)
# Replace missing values with mean
for attr in missing_attrs:
    df[attr] = df[attr].fillna(df[attr].mean())

# Result after processing
print('Show missing value:')
print(df.isna().sum())
df.info()
print(df['is_dead'].value_counts())
