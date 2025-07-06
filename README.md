# 🧠 Análise de Sentimentos com TF-IDF e N-Grams

Este projeto realiza uma análise de sentimentos utilizando técnicas clássicas de **Processamento de Linguagem Natural (PLN)** e aprendizado de máquina. A proposta é classificar frases como **positivas** ou **negativas** com base em suas palavras e combinações, usando como vetorização o **TF-IDF com N-Grams**.

---

## ✨ Objetivos do Projeto

- Realizar pré-processamento textual eficaz.
- Utilizar TF-IDF e N-Grams para vetorização.
- Treinar um modelo de classificação de sentimentos.
- Explorar técnicas para melhorar o desempenho do modelo.
- Mostrar a importância da escolha de features e da simplificação linguística.

---

## 🛠️ Tecnologias e Bibliotecas

- Python 3.10+
- `pandas` e `numpy` para manipulação de dados
- `nltk` para pré-processamento de texto
- `sklearn` para vetorização e modelagem (TF-IDF, N-Grams, Regressão Logística)
- `matplotlib` e `seaborn` para visualização de resultados

---

## 📊 Dataset

Utilizado um dataset de sentimentos com frases em português, contendo rótulos binários:
- `1`: Sentimento positivo
- `0`: Sentimento negativo

> Obs: O dataset já vem pré-formatado em português, evitando a necessidade de tradução.

---

## 🧹 Etapas do Pré-processamento

1. **Remoção de pontuações e números**
2. **Normalização de letras (minúsculas)**
3. **Remoção de stopwords (palavras irrelevantes)**
4. **Stemming/Lematização** (opcional)
5. **Tokenização das frases**

---

## 📈 Vetorização com TF-IDF + N-Grams

Foi utilizado o `TfidfVectorizer` com:
- `ngram_range=(1,2)` → unigrama e bigrama
- `max_features=5000` → limite no número de palavras para reduzir dimensionalidade

Isso permite capturar **expressões compostas** importantes, como "muito bom", "não gostei", entre outras.

---

## 🤖 Modelo de Machine Learning

Utilizamos **Regressão Logística** para a tarefa de classificação:
```python
from sklearn.linear_model import LogisticRegression
