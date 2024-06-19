import aiohttp
import asyncio
import os, time

api_key = os.environ['GOOGLE_API_KEY']

async def aget_http_response(url, session):

    async with session.get(url) as resp:
        return await resp.text()      # Ensure you await, otherwise it raises
                                        # RuntimeWarning: coroutine 'ClientResponse.text' was never awaited

async def acreate_coroutines(urls: list) -> list:

    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            tasks.append(asyncio.ensure_future(aget_http_response(url, session)))

        coroutines = await asyncio.gather(*tasks)

        return coroutines

def run_coroutines(urls: str):

    start_time = time.time()
    
    results = asyncio.run(acreate_coroutines(urls))

    time_taken = round(time.time() - start_time, 1)

    print(f"Executed in: {time_taken}")

    return results

if __name__ == '__main__':
    
    with open("data/world_bank_document_urls.txt", encoding='utf-8') as f:
        urls = f.readlines()

    run_coroutines(urls[:3])

    

   

    