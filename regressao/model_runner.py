"""
Módulo para treinar modelos de regressão linear.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def run_linear_regression(x, y, n_rounds=500, test_size=0.2, random_state=None):
    """
    Executa múltiplas rodadas de regressão linear, calculando MSE e MAE.

    Parâmetros
    ----------
    x : np.ndarray
        Variáveis independentes.
    y : np.ndarray
        Variável dependente.
    n_rounds : int
        Número de rodadas para shuffle e split.
    test_size : float
        Percentual de dados para teste.
    random_state : int ou None
        Semente para reprodução.

    Retorna
    -------
    dict
        Contendo listas de MSE e MAE para cada rodada.
    """
    model = LinearRegression()
    mse_list = []
    mae_list = []

    for i in range(n_rounds):
        rs = i if random_state is None else random_state + i
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs, shuffle=True)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mse_list.append(mean_squared_error(y_test, y_pred))
        mae_list.append(mean_absolute_error(y_test, y_pred))

    return {"MSE": mse_list, "MAE": mae_list}
