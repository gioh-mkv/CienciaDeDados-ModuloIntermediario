"""
===============================================================================================================================
1 - Quais bibliotecas e ferramentas são necessárias para realizar essa análise? Importe essas bibliotecas em um ambiente de 
desenvolvimento como Google Colab ou Jupyter Notebook.
===============================================================================================================================
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re
import os

nltk.download('stopwords')

"""
===============================================================================================================================
2 - Como podemos transformar os textos brutos em uma forma utilizável para o modelo? Utilize a técnica TfidfVectorizer para 
converter os textos em uma matriz de características numéricas. Explique por que o pré-processamento é essencial para a análise 
de texto.
===============================================================================================================================
"""

# OBS: não sabia se precisava importar os datasets direto do site, então baixei os arquivos e comprimi em zip junto da oficina
file_names = {
    'amazon': 'amazon_cells_labelled.txt',
    'imdb': 'imdb_labelled.txt',
    'yelp': 'yelp_labelled.txt'
}

# verifica se os arquivos existem
missing_files = [f for f in file_names.values() if not os.path.exists(f)]
if missing_files:
    print("ERRO: Os seguintes arquivos não foram encontrados no diretório atual:")
    for f in missing_files:
        print(f"- {f}")
    print("\nPor favor, certifique-se de que estes arquivos estão na mesma pasta do seu script:")
    print("1. amazon_cells_labelled.txt")
    print("2. imdb_labelled.txt")
    print("3. yelp_labelled.txt")
    exit()

# carrega os dados em um unico DataFrame
dfs = []
for source, file_name in file_names.items():
    try:
        df = pd.read_csv(file_name, delimiter='\t', header=None, names=['text', 'sentiment'])
        df['source'] = source
        dfs.append(df)
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_name}: {str(e)}")
        exit()

data = pd.concat(dfs, ignore_index=True)

print("\nprimeiras linhas do dataset:")
print(data.head())
print("\ndistribuição de sentimentos:")
print(data['sentiment'].value_counts())
print("\ndistribuição por fonte:")
print(data['source'].value_counts())

"""
A função preprocess_text() aplica minúsculas, remoção de pontuação, números, remoção de stopwords, stemming e tokenizacao.
Em seguida junta as palavras novamente
"""
def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = re.sub(r'\d+', '', text)
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

"""
modelos de machine learning não entende texto cru. Pré-processar padroniza e reduz ruídos linguísticos, enquanto o TF-IDF fornece 
uma representação útil do texto em números
"""

# aplicando o pré-processamento
data['processed_text'] = data['text'].apply(preprocess_text)

# visualizando o antes e depois
print("\nExemplo de pré-processamento:")
print("Antes:", data['text'].iloc[0])
print("Depois:", data['processed_text'].iloc[0])

# dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], data['sentiment'], 
    test_size=0.2, random_state=42, stratify=data['sentiment']
)

"""
===============================================================================================================================
3 - Qual modelo de classificação devemos usar e por quê? Construa um pipeline que integra o pré-processamento de texto e o modelo.
Explique a sua escolha de modelo para a tarefa de análise de sentimento.
===============================================================================================================================

usaremos o modelo Naive Bayes pois é adequado para tarefas de classificação de texto por sua eficiência e desempenho com 
vetores de contagem/peso
"""

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

"""
===============================================================================================================================
4 - Como treinamos o modelo para que ele aprenda a classificar os comentários? Treine o modelo Naive Bayes usando os dados de 
treino. Descreva o processo de treinamento e como o modelo aprende a partir dos exemplos rotulados.
===============================================================================================================================

o modelo aprende associando padrões nas palavras aos rótulos “positivo” ou “negativo”
"""
print("\nTreinando o modelo...")
pipeline.fit(X_train, y_train)

"""
===============================================================================================================================
5 - Como podemos avaliar o desempenho do modelo? Avalie o modelo usando o conjunto de teste e calcule a acurácia. Explique o que 
a acurácia nos diz sobre a qualidade das previsões do modelo.
===============================================================================================================================

a acurácia mostra quão bem o modelo generaliza: um valor alto indica que ele está capturando bem os padrões dos dados
"""

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia do modelo: {accuracy:.2f}")

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred))

def analyze_sentiment(text):
    processed = preprocess_text(text)
    pred = pipeline.predict([processed])
    return "positivo" if pred[0] == 1 else "negativo"

sample_texts = [
    "I love it!",
    "Terrible experience",
    "It was okay, nothing special.",
    "The movie was fantastic",
    "Worst service I've ever encountered.",
    "Boring...",
    "It's amazing"
]

print("\nTestando o modelo com exemplos:")
for text in sample_texts:
    sentiment = analyze_sentiment(text)
    print(f"Texto: '{text[:50]}...' | Sentimento: {sentiment}")

"""
pelas bases de dados serem em inglês, não foi possivel testar o modelo com exemplos em portugues
"""