import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Učitavanje podataka
data = pd.read_csv('dataset_full.csv')

# 1. Izračunavanje statističkih parametara za kolone 'median_income' i 'median_house_value'
columns = ['median_income', 'median_house_value']

for column in columns:
    print(f'Statistika za {column}:')
    print(f'Maksimum: {data[column].max()}')
    print(f'Minimum: {data[column].min()}')
    print(f'Standardna devijacija: {data[column].std()}')
    print(f'Medijana: {data[column].median()}')
    print(f'Srednja vrednost: {data[column].mean()}')
    print('\n')

# 2. Box and Whisker dijagrami za 'median_income' i 'median_house_value'
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
data.boxplot(column='median_income', ax=axes[0])
data.boxplot(column='median_house_value', ax=axes[1])
axes[0].set_title('Boxplot za median_income')
axes[1].set_title('Boxplot za median_house_value')
plt.show()

# 3. Histogrami za 'total_rooms', 'total_bedrooms', 'population' i 'households'
columns = ['total_rooms', 'total_bedrooms', 'population', 'households']

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for col, ax in zip(columns, axes.flatten()):
    ax.hist(data[col], bins=30, edgecolor='k')
    ax.set_title(f'Histogram za {col}')
plt.tight_layout()
plt.show()

# 4. Dijagram matrice korelacije
plt.figure(figsize=(10, 8))
numerical_data = data.drop(columns=['ocean_proximity'])  # Izostavljanje kategoričke kolone
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrica korelacije')
plt.show()

# 5. Dijagram matrice rasipanja
sns.pairplot(data)
plt.show()

# 6. Zaključci
# Dimenzije skupa podataka
print(f'Dimenzije skupa podataka: {data.shape}')

# Nedostajuće vrednosti
print(f'Nedostajuće vrednosti u skupu podataka:\n{data.isnull().sum()}')

# Karakteristika koja najviše utiče na prosečnu cenu kuća
print('Korelacija sa median_house_value:\n', correlation_matrix['median_house_value'].sort_values(ascending=False))

# Distribucija (skewness) podataka u kolonama 'total_rooms', 'total_bedrooms', 'population' i 'households'
print('Skewness:\n', data[['total_rooms', 'total_bedrooms', 'population', 'households']].skew())

# 7. Podela podataka na skupove za trening i testiranje
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

# 8. Utvrđivanje tipova podataka
print(f'Tipovi podataka:\n{data.dtypes}')

# 9. Transformisanje kategoričkih karakteristika
categorical_features = ['ocean_proximity']
numerical_features = data.drop(columns=categorical_features + ['median_house_value']).columns

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
    ('num', numerical_pipeline, numerical_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 10. Skaliranje karakteristika
# (Skaliranje je već urađeno u prethodnom koraku unutar numerical_pipeline)

# 11. Obrada nedostajućih vrednosti
# (Obrađeno unutar numerical_pipeline i categorical_pipeline koristeći SimpleImputer)

# 12. Generisanje novih karakteristika
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, numerical_features.get_loc('total_rooms')] / X[:, numerical_features.get_loc('households')]
        population_per_household = X[:, numerical_features.get_loc('population')] / X[:, numerical_features.get_loc('households')]
        bedrooms_per_room = X[:, numerical_features.get_loc('total_bedrooms')] / X[:, numerical_features.get_loc('total_rooms')]
        return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

# Kombinovanje u puni pipeline
full_pipeline_with_custom_features = Pipeline([
    ('preparation', full_pipeline),
    ('attribs_adder', CombinedAttributesAdder())
])

# Priprema podataka za trening
housing_prepared = full_pipeline_with_custom_features.fit_transform(train_set.drop(columns='median_house_value'))
housing_labels = train_set['median_house_value'].copy()

# Priprema podataka za testiranje
X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()
X_test_prepared = full_pipeline_with_custom_features.transform(X_test)

# 1. Spot-checking: Pronaći najbolji algoritam
models = [
    ('LR', LinearRegression()),
    ('DT', DecisionTreeRegressor()),
    ('RF', RandomForestRegressor())
]

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(model, housing_prepared, housing_labels, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_results)
    results.append(rmse_scores)
    names.append(name)
    print(f'{name}: {rmse_scores.mean()} ({rmse_scores.std()})')

# 2. Odabir najboljeg algoritma i podešavanje hiperparametara
param_grid = [
    {'n_estimators': [50, 100, 150], 'max_features': [8, 10, 12]}
]

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print("Najbolji hiperparametri:", grid_search.best_params_)

# Evaluacija najboljeg modela na trening skupu
final_model = grid_search.best_estimator_

# Predikcija i evaluacija
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_mae = mean_absolute_error(y_test, final_predictions)

print(f"RMSE na test skupu: {final_rmse}")
print(f"MAE na test skupu: {final_mae}")

# Sumiranje rezultata
print(f"Preciznost modela (RMSE): {final_rmse}")
print(f"Preciznost modela (MAE): {final_mae}")
print("Poboljšanje nakon podešavanja hiperparametara: %.2f" % (results[2].mean() - final_rmse))

# 3. Šta je moguće uraditi da bi se dodatno poboljšala preciznost modela?
# - Povećati količinu trening podataka
# - Ispitati dodatne algoritme (npr. Gradient Boosting, XGBoost)
# - Kreirati dodatne relevantne atribute
# - Ispitati dodatne metode obrade podataka
