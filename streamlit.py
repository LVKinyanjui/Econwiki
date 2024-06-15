import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone

from google.api_core.exceptions import FailedPrecondition

import os
# INITIALIZATION AND CONFIGURATION
# Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

# Pinecone initialization
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'), environment='gcp-starter')
index_name = pc.list_indexes()[0]['name']
index = pc.Index(index_name)
namespace = 'Econwiki'

@st.cache_data
def pinecone_query(query: str, with_expansion: bool = False) -> str:
    # TODO: Introduce more parameters rather than rely on  hardcoded values
    if with_expansion:
        prompt = f"""
            You are a useful informative assistant
            You are to be issued with a question
            You will create similar questions 
                similar scope, similar topics and themes,
            The point is to expound on the original question
                with related questions
            No need to write 'Similar questions'
            or do extensive formatting

            The question is attached below

            {query}

        """
        res = model.generate_content(prompt)
        query = res.text

    query_vector_ = genai.embed_content(content=query,
                                        model='models/embedding-001')
    query_vector = query_vector_['embedding']

    res = index.query(
        top_k=5,
        vector=query_vector,
        include_metadata=True,
        namespace=namespace
    )

    return '\n\n'.join([match['metadata']['text'] for match in res['matches']])


def llm_answer_query(context: str) -> str:
    prompt = f"""
        You are provided with a text to summarize
        There may be a main topic or theme that you can identify
        Synthesize the various aspects of the texts to create a concise yet informative summary
        The text follows below:

        {context}

    """
    res = model.generate_content(prompt)
    return res.text

# -------------------------------------------------------------
# STREAMLIT AREA

st.text_input(label="User Query",
              value="What is the role of the IMF in Kenya?",
              placeholder='Enter your question',
              key='user_query')

if "retrieval" not in st.session_state:
    st.session_state.retrieval = ''

st.session_state.retrieval = pinecone_query(st.session_state.user_query,
                                            with_expansion=True)

if "generation" not in st.session_state:
    st.session_state.generation = ''

st.session_state.generation = llm_answer_query(st.session_state.retrieval)


st.button('Submit')
st.write(st.session_state.generation)
st.write(st.session_state.retrieval)