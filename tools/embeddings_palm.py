import os
import google.generativeai as palm

def get_palm_embeddings(text: str, model_name='models/embedding-gecko-001') -> list :
    # Credentials
    api_key = os.getenv("PALM_API_KEY")
    palm.configure(api_key=api_key)

    model = palm.get_model(name=model_name)
    response = palm.generate_embeddings(text=text, model=model)

    return response['embedding']

if __name__ == '__main__':
    get_palm_embeddings("I am getting emeddings")
    print("This is a trial run")