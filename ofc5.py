"""
Você é um cientista de dados contratado por uma empresa imobiliária para desenvolver um modelo que ajude a 
prever o preço de casas com base em algumas características. 
A empresa possui um grande banco de dados com informações sobre diversas propriedades, como tamanho do lote, 
número de quartos, idade da casa, entre outras. Eles acreditam que, com um modelo preditivo eficiente, 
poderão ajustar melhor seus preços e oferecer melhores recomendações aos seus clientes.
Nesse cenário, você será guiado na construção de um modelo de regressão linear simples, mas eficaz, para 
prever o preço das casas usando Python e a biblioteca Scikit-learn. 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# criando um dataset fictício
data = {
    "Área do lote": [5000, 6000, 5500, 7000, 6200],
    "Ano de construção": [2000, 1995, 2010, 1980, 2005],
    "Área do 1º andar": [1500, 1600, 1550, 1800, 1650],
    "Área do 2º andar": [500, 600, 550, 400, 620],
    "Banheiros completos": [2, 3, 2, 3, 2],
    "Quartos acima do solo": [3, 4, 3, 4, 3],
    "Total quartos acima do solo": [4, 5, 4, 5, 4],
    "Preço de venda": [350000, 420000, 380000, 450000, 410000]
}

df = pd.DataFrame(data)

print(df.head())
print(df.describe())

# pra verificar valores nulos e lidar com eles, se precisar
print(df.isnull().sum())
data = df.dropna()

# começaremos com uma regressao linear simples usando apenas o tamanho do lote
X = df[['Área do lote']]  
y = df['Preço de venda']

# divide em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# cria o modelo de regressão e treina
model = LinearRegression()
model.fit(X_train, y_train)

# faz a previsao e calcula as métricas de avaliacao
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

# plotamos os dados e a linha de regresssao
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Previsão do modelo')
plt.xlabel('Tamanho do Lote (m²)')
plt.ylabel('Preço da Casa')
plt.title('Regressão Linear: Tamanho do Lote vs Preço da Casa')
plt.legend()
plt.show()

# coeficientes do modelo
print(f'Intercepto (alpha): {model.intercept_:.2f}')
print(f'Coeficiente (beta): {model.coef_[0]:.2f}')

# Equação do modelo
print(f'Equação: Preço = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Tamanho_Lote')