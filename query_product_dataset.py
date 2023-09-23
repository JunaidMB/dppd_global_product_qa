import argparse
import cohere
import json
import logging
import numpy as np
import os
import pandas as pd
import time
from annoy import AnnoyIndex
from dotenv import load_dotenv
from query_global_questions_index.create_index import create_annoy_index
from query_global_questions_index.llm_proposed_answer_check import check_proposed_answer
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pydantic import BaseModel, Field
from typing import Dict

load_dotenv(override=True)

# Configure the logging settings to print logs to the console (standard output)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialise Cohere Client
cohere_apikey = os.environ.get("COHERE_APIKEY")
co = cohere.Client(cohere_apikey)

def main(query: str, global_questions_lookup: Dict) -> None:
    # Make a list of global dataset related questions
    #start_time = time.time()

    global_questions = list(global_questions_lookup.keys())
    global_questions = [i.lower() for i in global_questions]
    
    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Converting questions to list took {section_time:.2f} seconds to execute")

    # Take an input query
    #start_time = time.time()

    query = query.lower()

    # Embed user query
    query_embed = co.embed(texts=[query], model="embed-english-v2.0").embeddings

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Embedding the query took {section_time:.2f} seconds to execute")

    # Create a Vector Index with Annoy containing the Global Questions if it doesn't exist, else load it. Annoy doesn't support adding vectors to an index so we have to create a new one everytime.
    # start_time = time.time()

    index_name = "global_question_embeddings"
    index_name_full = "".join([index_name, ".ann"])

    logging.info("Creating an index")
    create_annoy_index(data_to_index=global_questions, index_name=index_name)
    global_question_index = AnnoyIndex(np.array(query_embed[0]).shape[0], 'angular')
    global_question_index.load(index_name_full)

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Creating AnnoyIndex took {section_time:.2f} seconds to execute")

    # Check for exact match
    if query in global_questions:

        # start_time = time.time()

        logging.info("Found an Exact Match")
        final_answer = global_questions_lookup[query]

        # end_time = time.time()
        # section_time = end_time - start_time
        # logging.info(f"Retrieving an exact match answer {section_time:.2f} seconds to execute")

    else:
        logging.info("Did not find an Exact Match")

        # Find the closest question in the lookup and generate a proposed answer

        # start_time = time.time()

        similar_item_ids = global_question_index.get_nns_by_vector(query_embed[0], 1, include_distances=True)
        closest_question = global_questions[similar_item_ids[0][0]]
        proposed_answer = global_questions_lookup[closest_question]
        logging.info(f"proposed_answer:{proposed_answer}")

        # end_time = time.time()
        # section_time = end_time - start_time
        # logging.info(f"Finding the closest question and retrieving a proposed answer took {section_time:.2f} seconds to execute")

        # Check to see if the question is answered
        # start_time = time.time()

        class ProposedAnswerCheck(BaseModel):
            score: int = Field(description="Score that can be either -1 or 1")
            reason: str = Field(description="This is the reason why the proposed answer does or does not answer the question")

        pydantic_parser = PydanticOutputParser(pydantic_object=ProposedAnswerCheck)
        format_instructions = pydantic_parser.get_format_instructions()

        template = """
            Given a question and a proposed answer, you will determine if the question is answered by the question. 
            If the proposed answer does answer the question, then return a +1 score and give a reason why the proposed answer answers the question.
            If the proposed answer does not answer the question, then return a -1 score and mention why the proposed answer does not answer the question.
            
            Below are the question and proposed answer pair as a tuple in the form (question, proposed_answer)
            ({question}, {proposed_answer})

            Below are format instructions which explains how you should format your response
            {format_instructions}
            
            Response:"""

        prompt_template = PromptTemplate.from_template(
            template,
            output_parser=pydantic_parser)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

        response = check_proposed_answer(
            question=query,
            proposed_answer=proposed_answer,
            llm=llm,
            prompt_template=prompt_template,
            format_instructions=format_instructions,
        )
        
        logging.info(response)
        
        # Parse the Response
        structured_response = pydantic_parser.parse(response)
        
        logging.info(structured_response)

        # end_time = time.time()
        # section_time = end_time - start_time
        # logging.info(f"Checking Proposed Answer took {section_time:.2f} seconds to execute")

        if structured_response.score == 1:

            # start_time = time.time()

            final_answer = proposed_answer

            # end_time = time.time()
            # section_time = end_time - start_time
            # logging.info(f"Generating final answer took {section_time:.2f} seconds to execute")
        else:
            # start_time = time.time()

            # Load product catalogue data
            with open("product_descriptions.json", "r") as file:
                product_descriptions_json = json.load(file)

            # Create a DataFrame that we can query
            df = pd.DataFrame(product_descriptions_json)
            df = SmartDataframe(df, config={"llm": OpenAI(api_token=os.environ.get("OPENAI_API_KEY"))})
            
            final_answer = df.chat(query)
            new_qa_pair = {query: final_answer}
            global_questions_lookup.update(new_qa_pair)

            # end_time = time.time()
            # section_time = end_time - start_time
            # logging.info(f"Generating final answer took {section_time:.2f} seconds to execute")

            # start_time = time.time()

            with open("global_questions_lookup.json", "w") as outfile:
                json.dump(global_questions_lookup, outfile)

            # end_time = time.time()
            # section_time = end_time - start_time
            # logging.info(f"Updating Global Questions Lookup table took {section_time:.2f} seconds to execute")

        logging.info(f"Final Answer: {final_answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Query Parser')
    parser.add_argument('--query', help='Enter the query for the product catalogue')
    args = parser.parse_args()

    # Load Global Questions and Answers
    # start_time = time.time()

    with open("global_questions_lookup.json", "r") as file:
        global_questions_lookup = json.load(file)

    # end_time = time.time()
    # section_time = end_time - start_time
    # logging.info(f"Loading Global Questions Lookup table took {section_time:.2f} seconds to execute")
    
    main(query=args.query, global_questions_lookup=global_questions_lookup)
