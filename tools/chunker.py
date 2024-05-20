import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_chunks(data: list[str], tokens=400) -> list:
    # Tokenizer setup
    tokenizer = tiktoken.get_encoding('p50k_base')

    # Function to calculate token length
    def tiktoken_len(text):
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)

    # Text splitter setup
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=tokens,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["-"*150, "\n\n", "\n", " ", ""]
    )

    # Preprocess data
    texts = text_splitter.split_text(''.join(data))
    return texts

