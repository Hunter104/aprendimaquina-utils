import pandas as pd
import matplotlib.pyplot as plt
import time
import io
import psutil
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import randint
from send_report import send_report

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('teste.csv')

X_train = train_df.iloc[:, :-1].values
y_train = train_df['Class'].values
X_test = test_df.iloc[:, :-1].values
y_test = test_df['Class'].values

start_time = time.time()
cpu_start = psutil.cpu_percent(interval=None)
mem_start = psutil.virtual_memory().used

modelo_knn = KNeighborsClassifier()
param_dist = {
    'n_neighbors': randint(1, 20),
    'p': [1, 2],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}
random_search = RandomizedSearchCV(
    modelo_knn,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

end_time = time.time()
time_taken = end_time - start_time
cpu_end = psutil.cpu_percent(interval=None)
mem_end = psutil.virtual_memory().used

extra_info = (
    f"**Best Parameters:** `{random_search.best_params_}`\n\n"
    f"**CPU Usage:** {cpu_start}% → {cpu_end}%\n"
    f"**Memory Usage:** {mem_start / 1e6:.2f} MB → {mem_end / 1e6:.2f} MB"
)

plt.figure(figsize=(8, 6))
plt.scatter(X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('V1')
plt.ylabel('V2')
plt.title('Classificação de fraude (com RandomizedSearchCV)')
plt.colorbar(label='Classe prevista')

buf = io.BytesIO()
plt.savefig(buf, format='png')
plt.close()

send_report(y_test, y_pred, time_taken, graphs=[buf], extra_info=extra_info)
