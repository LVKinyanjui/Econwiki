import aiohttp
import asyncio
import os, time

api_key = os.environ['GOOGLE_API_KEY']

async def get_http_response(session, url, sentence):
    headers = {
        'Content-Type': 'application/json'
        }
    
    data = {
        'model':'models/embedding-001',
        'content': {
            'parts': [{
                'text': sentence
            }]
            }
        }

    
    async with session.post(url, 
                    headers=headers,
                    json=data,
                    params={'key': api_key}) as resp:
        
        embeddings = await resp.json()       # Ensure you await, otherwise it raises
                                        # RuntimeWarning: coroutine 'ClientResponse.text' was never awaited
        return {
            "embeddings": embeddings,
            "text_metadata": sentence
        }

async def async_embed(texts: list[str]):
    """
    Process the texts and return the embeddings.
    Args:
        texts: The texts to be processed.
    Returns:
        coroutines: The coroutines to be executed.
    """

    url = 'https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent'

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sentence in texts:
            tasks.append(asyncio.ensure_future(get_http_response(session, url, sentence)))

        coroutines = await asyncio.gather(*tasks)

        return coroutines
    
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


def embed(texts: str):
    """
    A wrapper for the asynchronous operation. Runs the coroutines and returns actual embeddings
    Args:
        texts: The texts to be processed.
    Returns:
        results: The embeddings as returned from the http request.
        time_taken: The time taken to process the texts.
    """
    start_time = time.time()
    
    results = asyncio.run(async_embed(texts))
    time_taken = round(time.time() - start_time, 1)

    print("--- %s seconds ---" % (time_taken))

    return results, f"Execution time {str(time_taken)} seconds"

if __name__ == '__main__':
    
    sentences = ['Hello World', 'This is a sentence', 'This is another sentence']
    print(embed(sentences))

    

   

    