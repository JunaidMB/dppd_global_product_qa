import cohere
from annoy import AnnoyIndex
import numpy as np
import os
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

def create_annoy_index(data_to_index: List, 
    index_name: str, 
    metric: str = "angular", 
    num_trees: int = 10) -> None:
    
    
    # Initialise Cohere Client
    cohere_apikey = os.environ.get("COHERE_APIKEY")
    co = cohere.Client(cohere_apikey)

    # Use Cohere's embeddings to embed the questions
    text_embeddings = co.embed(texts = data_to_index, model='embed-english-v2.0').embeddings

    # Create an Index to store embeddings
    search_index = AnnoyIndex(np.array(text_embeddings).shape[1], metric)

    # Add all vectors to search index
    for i in range(len(text_embeddings)):
        search_index.add_item(i, text_embeddings[i])

    search_index.build(num_trees) 

    index_name_full = "".join([index_name, ".ann"])
    search_index.save(index_name_full)

    print(f"{index_name_full} has been successfully created")

