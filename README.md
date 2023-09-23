# Dynamic Personalised Product Display - Product Question and Answer

This repository contains code which uses LLMs to answer global user queries about a product catalogue. The approach involves maintaining a lookup dictionary of global question and answers about the product catalogue. If a user asks a question about a product, the following checks will be performed to return an answer:

1. Check if the question has been asked before - if yes, lookup the final answer and serve to the user. If no, proceed to step 2.
2. Embed the question into a vector and perform a similarity search with a vector index where all questions previously asked are embedded into the same vector index. This will yield a closest question to the user question. Lookup the answer to the closest question, call this the proposed answer, and proceed to step 3.
3. Feed the user question and proposed answer to an LLM and ask it to return a score indicating if the proposed answer factually answers the user question. Return a 1 if the proposed answer answers the user question and -1 otherwise. If the LLM returns a 1, the final answer is the proposed answer and serve to the user. If the LLM returns a -1, proceed to step 4.
4. Represent the lookup dictionary of global question and answers as a dataframe and answer the user query from the dataframe, using the `pandas-llm` library, - this is the final answer. Serve the answer to the user and proceed to step 5.
5. Update the global question and answer lookup table with the user question and answer pair.

The approach uses Cohere's embeddings to perform dense retrieval.  The ChatGPT API is used to check if a proposed answer successfully answers a question. 

To experiment with a user query, run 

`python query_product_dataset.py --query "what is the square root price for the cheapest product" `

in the terminal

## Areas for Improvement

1. Reduce latency: The response is not fast (~4 seconds). The workflow must be optimised to use appropriate data structures (Polars vs Pandas), make all lists into arrays and explore a faster search index. Also look at replacing the `pandas-llm` library to see if we can have a faster implementation.

2. Expand initial Global Question and Answer lookup dictionary: Adding more questions and data will place more load on the lookup operation which is fast. 

3. Consider traditional fuzzy matching based on NLP features vs semantic search: This may be faster at finding questions asking about the same thing as the user.

4. Explore using Guardrails approach to assess query fitness.
   
5. Change the Vector Index creation workflow: Right now, the vector index is created anew everytime the lookup table is updated. This is time intensive, we should support an add vector to index operation. 
   
6. Finetune a local LLM/ agent that can perform Dataframe Operations: If we can use a local LLM to generate code to answer a user query and execute code, it may be faster.
   
7. Add Exception handling 
   

**Note**: I used pip-tools to create requirements.txt. Use `pip-sync` inside a virtual environment to load dependencies. A Cohere and OpenAI API key are required.

## References

1. [Large Language Models with Semantic Search by Cohere](https://www.deeplearning.ai/short-courses/large-language-models-semantic-search/)
2. [Cohere](https://cohere.com/)
3. [Dynamic Personalised Product Display - Product Question and Answer with Semantic Search](https://github.com/JunaidMB/dppd_product_qa)