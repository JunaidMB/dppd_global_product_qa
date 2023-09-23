from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Type

load_dotenv(override=True)


def check_proposed_answer(
    question: str,
    proposed_answer: str,
    llm: Type[ChatOpenAI],
    prompt_template: Type[PromptTemplate],
    format_instructions: str
    ) -> str:
    
    # Provide input data for the prompt
    input_data = {"question": question, "proposed_answer": proposed_answer, "format_instructions": format_instructions}

    # Create LLMChain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Run the LLMChain to obtain response
    response = chain.run(input_data)

    return response
