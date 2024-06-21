import asyncio
import json, uuid, os
import itertools
from pinecone import Pinecone

# Initialize the client with pool_threads=30. This limits simultaneous requests to 30.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), pool_threads=30)

# Pick an index at random
index_name = pc.list_indexes()[0]['name']
namespace = 'test_async_upsert'


def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def pinecone_upsert(embeddings: list[float], texts: list[str], index_name: str, namespace: str):
    """
    Upsert data with 100 vectors per upsert request asynchronously
    - Pass async_req=True to index.upsert()
    """

    records = []

    for embedding, text in zip(embeddings, texts):
        records.append({
            'id': str(uuid.uuid4().int),
            'values': embedding,
            'metadata': {
                'text': text
            }
        })

    with pc.Index(index_name) as index:
        # Send requests in parallel
        async_results = [
            index.upsert(vectors=ids_vectors_chunk, async_req=True, namespace=namespace)
            for ids_vectors_chunk in chunks(records, batch_size=100)
        ]
        # Wait for and retrieve responses (this raises in case of error)
        [async_result.get() for async_result in async_results]

        return async_results
    
# Delete index after test is done


async def delete_namespace_after_wait(index_name, namespace, wait=300):
    await asyncio.sleep(wait)

    # Delete namespace if found
    # Will be created anew when we upsert to it. Avoids duplication
    with pc.Index(index_name) as index:
        if namespace in index.describe_index_stats()['namespaces'].keys():
            index.delete(delete_all=True, namespace=namespace)
            index.describe_index_stats()    

        print("Succesfully deleted the namespace!")

async def main():

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the JSON file
    json_file_path = os.path.join(script_dir, 'data', 'sample_embeddings.json')

    with open(json_file_path, "r") as file:
        embed_dict = json.load(file)

    pinecone_upsert([vect['embeddings']['embedding']['values'] for vect in embed_dict],
                    [txt['text_metadata'] for txt in embed_dict],
                    index_name, 
                    namespace=namespace)
    
    # Run the delete_namespace_after_wait asynchronously
    task = asyncio.create_task(delete_namespace_after_wait(index_name, namespace))

    # Wait for the task to complete
    await task

if __name__ == '__main__':
    asyncio.run(main())
    print("DONE")
     