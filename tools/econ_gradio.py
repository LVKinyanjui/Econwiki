# %% [markdown]
# ## Gradio
# This is a tool that allows us to create nice user interfaces for our LLMs. It simplifies the task of deploying them by removing the need to understand web development in depth.

# %%
import os
import gradio as gr
import google.generativeai as palm
# Custom modules
from embeddings_palm import get_palm_embeddings

# %%
api_key = os.getenv("PALM_API_KEY")
palm.configure(api_key=api_key)

# %% [markdown]
# ## Pinecone

# %%
import pinecone

# %% [markdown]
# ### Credentials

# %%
pinecone_api_key = os.getenv('PINECONE_API_KEY_03')

# %% [markdown]
# ### Connecting to Pinecone Index

# %%

index_name = 'econwiki'

# initialize connection to pinecone
pinecone.init(
    api_key= pinecone_api_key,
    environment="gcp-starter"  # next to API key in console
)

# %%
# connect to index
index = pinecone.GRPCIndex(index_name)
# view index stats
index.describe_index_stats()

# %%
def make_query(query):
    # Get query vector embeddings
    xq = get_palm_embeddings(query)
    res = index.query(xq, top_k=5, include_metadata=True)

    # get list of retrieved text
    contexts = [item['metadata']['text'] for item in res['matches']]

    # Concatenate retrieved texts from vector database with the query
    ## May exceed context length if too many
    augmented_query = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"+query 
    
    # system message to 'prime' the model
    primer = f"""
        You are an econ major seeking to understand the world around you through economics \
        You will be provided with texts on economic phenomena on which you will make comments \
        These texts come from various sources such as articles, textbooks and so on \
        Your role is to condense the contents therein into language that is accessible to an intermediate student \
        You will not oversimplify key concepts but you will also not use inacessible technical language \
        After the text there will be a question you will be required to answer \
        If you do not find the answer in the text provided you will clearly state that you do not know \
        That is, you will not make answers up without validation from text \
        Provide direct quotations from key figures in the text to support your answers \
    """
    res = palm.chat(context=primer, messages=augmented_query, temperature=0.0)
    return res.last

# %%
demo = gr.Interface(fn=make_query, 
                    inputs=[gr.Textbox(label="Query to make", lines=6)],
                    outputs=[gr.Markdown(label="Result")],
                    title="EconWiki",
                    description="A central repository for macro economic data. Ask Away!"
                   )
demo.launch(share=True)

# %%
gr.close_all()

# %%



