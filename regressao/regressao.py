import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv("../data/aerogerador.dat", sep=r"\s+", header=None, names=["velocidade", "potencia"])

plt.scatter(data["velocidade"], data["potencia"], alpha=0.7)
plt.xlabel("Velocidade do vento")
plt.ylabel("Potência gerada")
plt.title("Dispersão: Velocidade x Potência")
plt.show()

x = data[["velocidade"]].values
y = data["potencia"].values

# Modelos
model = LinearRegression()
mse_list = []
mae_list = []

for i in range(500):
    x_train, x_test, y_train, y_test = train_test_split(
        x,y, test_size=0.2, random_state=i, shuffle=True)
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    mse_list.append(mean_squared_error(y_test, y_predict))
    mae_list.append(mean_absolute_error(y_test, y_predict))

def resultados(metricas):
    return {
        "Média": np.mean(metricas),
        "Desvio-padrão": np.std(metricas, ddof=1),
        "Maior valor": np.max(metricas),
        "Menor valor": np.min(metricas)
    }

tabela = {"MSE": resultados(mse_list), "MAE": resultados(mae_list)}

print("\n Resultados Regressão Linear considerando 500 rodadas:")
print(pd.DataFrame(tabela).T)
