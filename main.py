#%% md
# # Data Mining Assignment 3: Clustering and Anomaly Detection
# Daan Thielemans [s0210206]
#%% md
# ## Task 1: Data Exploration and Preprocessing
#%% md
# imports
#%%
import pandas as pd
display = print
#%% md
# datasets
#%%
try:
    df_anomalies = pd.read_csv('anomalies.csv')
    df_articles = pd.read_csv('articles.csv')
    df_clusters = pd.read_csv('clusters.csv')
    print("all datasets loaded successfully")
except FileNotFoundError as e:
    print(f"error: {e}")
#%% md
# check heads
#%%
datasets = {
    "anomalies.csv": df_anomalies,
    "articles.csv": df_articles,
    "clusters.csv": df_clusters
}

for name, df in datasets.items():
    print(f"{name} {df.shape}")
    display(df.head(3))
#%% md
# filtering, cleaning and bag-of-words
#%%
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer

# setup Lemmatizer
lem = WordNetLemmatizer()
def lemma_tokenizer(doc):
    return [lem.lemmatize(w.lower()) for w in doc.split() if w.isalpha()]

# setup CountVectorizer
cv = CountVectorizer(
    tokenizer=lemma_tokenizer,
    stop_words='english', # removes stop words
    min_df=3, # removes rare terms
    max_df=0.9, # removes frequent terms
    token_pattern=None
)

# fit and transform
bow_matrix = cv.fit_transform(df_articles['text'].fillna(''))
print(f"bag-of-words matrix shape: {bow_matrix.shape}")

# preview single article
ARTICLE_IDX = 1
original_text = df_articles.iloc[ARTICLE_IDX]['text']

# convert that specific row of the matrix to a DataFrame
second_row_df = pd.DataFrame(
    bow_matrix[ARTICLE_IDX].toarray(),
    columns=cv.get_feature_names_out()
)

# filter to only words that appear in this document
active_words = second_row_df.columns[second_row_df.iloc[0] > 0]
word_counts = second_row_df[active_words].T.rename(columns={0: 'Count'})

print(f"ORIGINAL TEXT (article #{ARTICLE_IDX})")
print(f"\"{original_text}\"")

print("BAG-OF-WORDS REPRESENTATION (head)")
display(word_counts.sort_values(by='Count', ascending=False).head())
#%% md
# ## Task 2: Clustering
#%% md
# tfidf
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.feature_extraction import text

# add custom words to remove
stop_words = text.ENGLISH_STOP_WORDS.union(['ha', 'doe', 'u', 'wa', 't', 'c', 'v'])

# use TF-IDF instead of simple CountVectorizer
tfidf = TfidfVectorizer(
    tokenizer=lemma_tokenizer,
    stop_words=list(stop_words),
    min_df=3,
    max_df=0.3, # Lowered slightly to be more aggressive with common words
    token_pattern=None
)

tfidf_matrix = tfidf.fit_transform(df_articles['text'].fillna(''))

# normalize to remove artice length factor
tfidf_norm = normalize(tfidf_matrix)
#%% md
# k-means clustering
#%%
from sklearn.cluster import KMeans

SEED = 66
km = KMeans(n_clusters=10, random_state=SEED, n_init=10)

# fit and predict
df_articles['kmeans_clusters'] = km.fit_predict(tfidf_norm)

print(df_articles['kmeans_clusters'].value_counts())
#%% md
# spectral clustering
#%%
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity

# calculate cosine similarity
affinity_matrix = cosine_similarity(tfidf_norm)

# use Spectral Clustering with the 'discretize' label strategy
SEED = 66
sc_improved = SpectralClustering(
    n_clusters=10,
    affinity='precomputed', # tell it we already calculated the similarity
    assign_labels='discretize',
    random_state=SEED
)

# fit and predict
df_articles['spectral_clusters'] = sc_improved.fit_predict(affinity_matrix)

print(df_articles['spectral_clusters'].value_counts())
#%% md
# evaluation
#%%
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import numpy as np

# list to store results
eval_results = []

# models
models = {
    "K-Means": df_articles['kmeans_clusters'],
    "Spectral": df_articles['spectral_clusters']
}

print("--- Clustering Evaluation Metrics ---")

for name, labels in models.items():
    # calculate silhouette score (closer to 1 is better)
    s_score = silhouette_score(tfidf_norm, labels)

    # calculate Calinski-Harabasz index (higher is better)
    ch_score = calinski_harabasz_score(tfidf_norm.toarray(), labels)

    eval_results.append({
        "Model": name,
        "Silhouette Score": round(s_score, 4),
        "Calinski-Harabasz": round(ch_score, 2)
    })

# display as a df for easy reading
df_eval = pd.DataFrame(eval_results)
display(df_eval)

# extract top keywords
def get_top_keywords(labels, n_terms=10):
    # map feature names to a dense average per cluster
    feature_names = tfidf.get_feature_names_out()
    dense_tfidf = tfidf_norm.toarray()

    # get unique labels
    unique_labels = sorted(labels.unique())
    for cluster_id in unique_labels:
        indices = np.where(labels == cluster_id)[0]
        cluster_mean = dense_tfidf[indices].mean(axis=0)
        top_indices = cluster_mean.argsort()[-n_terms:][::-1]
        words = [feature_names[i] for i in top_indices]
        count = len(indices)
        print(f"C{cluster_id} ({count} articles): {', '.join(words)}")

print("-- Method A: K-MEANS --")
get_top_keywords(df_articles['kmeans_clusters'])
print("-- Method B: SPECTRAL --")
get_top_keywords(df_articles['spectral_clusters'])
#%% md
# export
#%%
# get clusters
output_df = df_articles[['doc_id', 'kmeans_clusters']].copy()

# formatting for submission
output_df = output_df.rename(columns={'kmeans_clusters': 'label'})
output_df['label'] = output_df['label'].astype(int)

# final save
output_df.to_csv('clusters.csv', index=False)

print("'clusters.csv' filled successfully")
print(f"Total records exported: {len(output_df)}")
#%% md
# ## Task 3: Anomaly Detection
#%% md
# identify anomalies
#%%
import pandas as pd
from sklearn.ensemble import IsolationForest

# initialize and fit the model
SEED = 66
contamination_rate = 50 / len(df_articles)
iso_forest = IsolationForest(n_estimators=100, contamination=contamination_rate, random_state=SEED)
iso_forest.fit(tfidf_norm)

# get scores and extract the 50 most extreme outliers
df_articles['anomaly_score'] = iso_forest.decision_function(tfidf_norm)
anomalies_df = df_articles.nsmallest(50, 'anomaly_score').copy()

# create the specific format
anomalies_output = pd.DataFrame({
    'anomaly': range(1, 51),
    'doc_id': anomalies_df['doc_id'].values
})

# save to CSV
anomalies_output.to_csv('anomalies.csv', index=False)

print("'anomalies.csv' filled with 50 IDs")
print(anomalies_output.head())