import pandas as pd
import matplotlib.pyplot as plt
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from send_report import send_report

iris_data = pd.read_csv('creditcard.csv')

X = iris_data.iloc[:, :-1].values
y = iris_data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_knn(k, p = 2):
    start = time.time()
    modelo_knn = KNeighborsClassifier(n_neighbors=k, p=p)
    modelo_knn.fit(X_train, y_train)
    y_pred = modelo_knn.predict(X_test)
    end = time.time()
    time_taken = f"{end - start:.2f}s"

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 1], X_test[:, 2], c=y_pred, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('V1')
    plt.ylabel('V2')
    plt.title('Classificação de fraude')
    plt.colorbar(label='Classe prevista')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    send_report(y_test, y_pred, time_taken, graphs=[buf], extra_info=f'Knn hyperparameters: K={k}, p={p}')

for k in range (1, 8):
    train_knn(k)
