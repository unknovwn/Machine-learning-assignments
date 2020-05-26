import pandas as pd

df = pd.read_csv('titanic.csv')

total_passengers = df['PassengerId'].count()

male_count = df[df['Sex'] == 'male']['PassengerId'].count()  # df['Sex'].value_counts()[0]
female_count = df[df['Sex'] == 'female']['PassengerId'].count()  # df['Sex'].value_counts()[1]

survived_count = df[df['Survived'] == 1]['PassengerId'].count()
survived_share = survived_count / total_passengers * 100

first_class_passengers_count = df[df['Pclass'] == 1]['PassengerId'].count()
first_class_passengers_share = first_class_passengers_count / total_passengers * 100

age_without_na = df['Age'].dropna()
passengers_age_mean = age_without_na.mean()
passengers_age_median = age_without_na.median()

sibsp_parch_pearson_correlation = df[['SibSp', 'Parch']].corr().at['SibSp', 'Parch']


def get_female_first_name(full_name):
    if '(' in full_name:
        open_bracket_index = full_name.index('(')
        return full_name[open_bracket_index + 1:].split()[0]
    elif 'Miss.' in full_name:
        words = full_name.split()
        miss_index = words.index('Miss.')
        return words[miss_index + 1]
    elif 'Mrs.' in full_name:
        words = full_name.split()
        miss_index = words.index('Mrs.')
        return words[miss_index + 1]


female_full_names = df[df['Sex'] == 'female']['Name']
female_first_names = female_full_names.apply(get_female_first_name)
most_popular_female_name = female_first_names.groupby(female_first_names).count().sort_values(ascending=False).index[0]

print('Total Titanic\'s passengers:', total_passengers)
print('Count of men on the ship:', male_count)
print('Count of women on the ship:', female_count)
print('Share of survivors:', survived_share)
print('Share of first class passengers:', first_class_passengers_share)
print('Passengers mean age:', passengers_age_mean)
print('Passengers median age:', passengers_age_median)
print('Pearson correlation between Siblings/Spouses, Parents/Children:', sibsp_parch_pearson_correlation)
print('Most popular female name:', most_popular_female_name)
