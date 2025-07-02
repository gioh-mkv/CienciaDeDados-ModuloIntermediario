import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Gerar dataset sintético
np.random.seed(0)
n_samples = 300

data = {
    'valor_medio_gasto': np.random.randn(n_samples) * 20 + 200,
    'frequencia_compras': np.random.randn(n_samples) * 5 + 15,
    'categorias_produtos': np.random.randn(n_samples) * 2 + 8
}

df = pd.DataFrame(data)

# Ajustar valores para não terem valores negativos (por exemplo, categorias não podem ser negativas)
df['valor_medio_gasto'] = df['valor_medio_gasto'].clip(lower=0)
df['frequencia_compras'] = df['frequencia_compras'].clip(lower=0)
df['categorias_produtos'] = df['categorias_produtos'].clip(lower=0)

# 2. Normalizar os dados (importante para K-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Aplicar K-means (vamos escolher k=4 clusters por exemplo)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# 4. Adicionar a coluna do cluster no DataFrame original
df['cluster'] = kmeans.labels_

# 5. Visualizar os clusters com um gráfico 3D
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'blue', 'green', 'orange']
for cluster in range(k):
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(
        cluster_data['valor_medio_gasto'],
        cluster_data['frequencia_compras'],
        cluster_data['categorias_produtos'],
        s=50,
        c=colors[cluster],
        label=f'Cluster {cluster}'
    )

ax.set_xlabel('Valor Médio Gasto')
ax.set_ylabel('Frequência de Compras')
ax.set_zlabel('Categorias de Produtos')
ax.set_title('Segmentação de Clientes via K-means')
ax.legend()
plt.show()
