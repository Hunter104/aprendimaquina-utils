import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from send_report import send_report

dataset_df = pd.read_csv('data/creditcard.csv')
X = dataset_df.iloc[:, :-1].values
y = dataset_df['Class'].values

start_time = time.time()

rf_model = RandomForestClassifier()

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(1, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),
    'criterion': ['gini', 'entropy', 'log_loss'],
    'class_weight': ['balanced'],
    'n_jobs': [None],
}

random_search = RandomizedSearchCV(
    rf_model,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X, y)
best_model = random_search.best_estimator_

end_time = time.time()
time_taken = end_time - start_time

send_report(
    time_taken=time_taken,
    cv_results=random_search.cv_results_,
    extra_info=random_search.best_params_
)
