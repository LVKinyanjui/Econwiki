# %%
import asyncio
import tiktoken
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

# %%
NAMESPACE = 'Econwiki'          # A relevant namespace to store our documents under.

# %% [markdown]
# <h3 align=center> Prepare Documents </h3>
# 

# %%
with open(r'data\imf_article_txt', encoding='utf-8') as f:
    texts = f.read()

# %%
separator = "\n\n" + "-" * 150                  # Defined earlier during webscraping

# Necessary to limit the payload to and avoid a
# 400: 'Request payload size exceeds the limit: 10000 bytes.'

text_splitter = RecursiveCharacterTextSplitter(separators=[separator, "\n\n\n", "\n\n", "\n"], 
                                               chunk_size=7000,         # Empirically set from the output of CharacterTextSplitter
                                               chunk_overlap=0)
docs = text_splitter.split_text(texts)
len(docs)

# %% [markdown]
# <h3 align=center> Exploring Docs </h3>

# %%
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# %%
import matplotlib.pyplot as plt

# Calculate the number of tokens for each document
counts = [num_tokens_from_string(d, "cl100k_base") for d in docs]

# Plotting the histogram of token counts
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Histogram of Token Counts")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)

# Display the histogram
plt.show

# %%
# Doc texts concat
concatenated_content = "\n\n\n --- \n\n\n".join(docs)
print(
    "Num tokens in all context: %s"
    % num_tokens_from_string(concatenated_content, "cl100k_base")
)

# %% [markdown]
# ### Remarks
# Most of the documents are small ( < 2000 tokens ) though some are considerably longer at around 8000 tokens. The whole corpus is just shy of 90000 tokens. This is too large to fit in standard 32k context windows.

# %% [markdown]
# <h3 align=center> Define Model </h3>

# %%
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# %%
model = ChatGoogleGenerativeAI(google_api_key=os.getenv('GOOGLE_API_KEY'), model='gemini-pro')

# %% [markdown]
# <h2 align=center> Clustering </h2>
# 
# Now onto step two. Given the embeddings we have gotten we now cluster the texts. There are various well-researched techniques available to us.

# %%
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture

RANDOM_SEED = 224  # Fixed seed for reproducibility


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


# %%
from custom.gemini_async import async_embed             # Custom module

# %%
async def async_embed_with_postprocess(texts: list[str]) -> list[float]:
    """Call custom async embedder and return embeddings with an (informal) interface acceptable to downstream operations"""

    # await the asynchronous embedding function
    # In a jupyter notebook trying to start an event loop with asyncio.run witll result in an error
    all_embeddings = await async_embed(texts)

    try:
        raw_embeddings = [sub_embedding['embedding']['values'] 
                                            for sub_embedding in [embedding['embeddings'] 
                                                                for embedding in all_embeddings]]
    except KeyError:
        raise Exception("Possible HTTP CODE 400: 'Request payload size exceeds the limit: 10000 bytes. Documents may be too large")
    return raw_embeddings

# %%
# TODO
# The summary prompts HAVE to be changed to make them generalized.
# They are currently too specific to the use case of the authors.
# Perhaps the user can specify the general topic to better guide the LLM.

# %%
async def embed_cluster_texts(texts: list[str]) -> pd.DataFrame:
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    embed_list_2d = await async_embed_with_postprocess(texts)  # Generate embeddings
    text_embeddings_np = np.array(embed_list_2d)

    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def fmt_txt(df: pd.DataFrame) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - df: DataFrame containing the 'text' column with text documents to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    unique_txt = df["text"].tolist()
    return "--- --- \n --- --- ".join(unique_txt)



async def embed_cluster_summarize_texts(
    texts: List[str], level: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = await embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique()

    print(f"--Generated {len(all_clusters)} clusters--")

    # Summarization
    template = """
        This is a summarization task in 20 or so words
        Your goal is to be descriptive but concise.
        Create something like an abstract; a fitting summarization of the whole document.
        You are expected to summarize the following document:
        ```
        {context}
        ```
        
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()

    # Format text within each cluster for summarization
    summaries = []
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i]
        formatted_txt = fmt_txt(df_cluster)
        summaries.append(chain.invoke({"context": formatted_txt}))

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


async def recursive_embed_cluster_summarize(
    texts: List[str], level: int = 1, n_levels: int = 3
) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = await embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = await recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results

# %%
# Build tree
results = asyncio.run(recursive_embed_cluster_summarize(docs,         # Leaf texts
                                            level=1, 
                                            n_levels=3)
)


# %%
results[1][1]

# %% [markdown]
# #### Remarks
# This mortly ensemble works! I have modified the original code to fit my use of asynchronous embeddings and I am glad they play well together.
# 
# However, because of the payload limits imposed on us by the Gmeini API, we have had to resort to chunking our documents. This we inteded not to do, because the whole idea was to embed entire documents as they are and perform tree based RAG/ But the limitations of practical tools have forced a compromise upon us. We must make the best of it.

# %% [markdown]
# <h3 align=center> Collapsed Tree Retrieval </h3>
# 
# > This involves flattening the tree structure into a single layer and then applying a k-nearest neighbors (kNN) search across all nodes simultaneously.
# 
# It is reported to have the best performance.
# 
# ### Strategy
# 
# We will have a two pronged strategy: upsert the texts and the summaries separately. They are flattened but we already have embeddings for the texts already. We got them during the clustering operation. There is no need to get them anew, that would be inefficient. We don't have the embeddings for the summaries though, these we get. Then we use the pinecone client to upsert them sequentially.

# %% [markdown]
# <h3 align=center> Pinecone CRUD Operations </h3>
# 
# We are going to go our own way in this section. Instead of using the absractions langchain provides us to interact with vectorstores, we will perform our operations using the `pinecone` client. This gives us finer control.

# %%
from pinecone import Pinecone
import os, uuid

# %%
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')

# Pick an index at random
index_ = pc.list_indexes()[0]
index = pc.Index(index_['name'])

# Check whether index matches our embedding dimension
dim_a = index_['dimension']
dim_b = len(results[1][0]['embd'][0])       # Pick any random embedding vector in our results

if dim_a != dim_b:
    raise Exception(f"Pinecone Index dimension: {dim_a} does not match Vector Embedding dimension {dim_b}")

# Delete namespace if found
# Will be created anew when we upsert to it. Avoids duplication
if NAMESPACE in index.describe_index_stats()['namespaces'].keys():
    index.delete(delete_all=True, namespace=NAMESPACE)
    index.describe_index_stats()

# %%
def pinecone_upsert(embeddings: list[float], texts: list[str], index: Pinecone.Index, namespace: str):
    """Store embeddings and their corresponding text metadata in the pinecone vectorstore"""
    records = []

    for embedding, text in zip(embeddings, texts):
        records.append({
            'id': str(uuid.uuid4().int),
            'values': embedding,
            'metadata': {
                'text': text
            }
        })

    # Asynchronous upsert: Faster
    def chunker(seq, batch_size):
        return (seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size))

    async_results = [
        index.upsert(vectors=chunk, namespace=namespace, async_req=True)
        for chunk in chunker(records, batch_size=100)
    ]


# %%
# Iterate through the results to extract summaries from each level and add them to all_texts
summaries = [results[level][1]["summaries"][0] for level in sorted(results.keys())]
summary_embeddings = asyncio.run(async_embed(summaries))

# %%
# Upsering summaries
pinecone_upsert([txt['embeddings']['embedding']['values'] for txt in summary_embeddings],
                [txt['text_metadata'] for txt in summary_embeddings],
                index, 
                NAMESPACE)     

# %%
# Upserting all texts
pinecone_upsert(results[1][0]['embd'].tolist(),
                results[1][0]['text'].tolist(),
                index, 
                NAMESPACE)     



# %% [markdown]
# #### Remarks
# After a long process, we have been able to upsert our documments succesfully to pinecone. The moving parts don't fit very well and the construction is brittle. We move on but we will return to refactor.

# %%

print("DONE")

