import time
from tqdm import tqdm
from uuid import uuid4
from embeddings_palm import get_palm_embeddings

def pineconify_vectors(chunks: list[str]) -> list:
    """
    Gets the vector embeddings from the embedding service, like palm.generate_embeddings
    Formats those embeddings in such a way that they can be upserted to Pinecone.
    """
    vectors = []
    for text in tqdm(chunks):
        
        try:
            vectors.append(
                {
                    'id': str(uuid4()),
                    'values': get_palm_embeddings(text),
                    'metadata': {
                        'text': text
                        }
                }
            )
        except Exception as e:
            print(f"The following error occurred: {e}")
            time.sleep(2.5)
            continue

    return vectors
