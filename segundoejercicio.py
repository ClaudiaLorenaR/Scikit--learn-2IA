from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from tabulate import tabulate

#Cargar el conjunto de datos California Housing

California=fetch_california_housing()
df = pd.DataFrame(California.data, columns=California.feature_names) 
df['Price']=California.target
X_train, X_test, y_train, y_test = train_test_split(California.data, California.target, test_size=0.2, random_state=42)

#Crear el modelo de regresión lineal
model=LinearRegression()

#Entrenar el modelo
model.fit(X_train, y_train)

#Predecir los precios de las viviendas para los datos de prueba
y_pred=model.predict(X_test)

#Calcular el error cuadrático medio
mse= mean_squared_error(y_test, y_pred)

# Crear el título de la tabla
print("=" * 20 + " Dataset de California Housing " + "=" * 20)
# Mostrar las 10 primeras filas en formato tabla
print(tabulate(df.head(10), headers='keys', tablefmt='pretty', floatfmt='.3f'))

#Imprimir el error cuadrático medio
print('Error cuadrático medio:', mse)
