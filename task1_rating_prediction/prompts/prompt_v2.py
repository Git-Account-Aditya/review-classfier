from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from structured_output import StructuredOutput

def get_prompt_v2():
    SYSTEM_PROMPT = """
    You are an expert product review analyst. Your task is to analyze the sentiment of a product review and predict the star rating (1-5) given by the user.

    Review Content:
    {review_text}

    Reactions:
    {reactions}

    Instructions:
    - Analyze the tone, keywords, and overall sentiment of the review.
    - Consider the context provided by any reactions (e.g., if many people found it helpful, the sentiment might be more representative).
    - Predict the star rating from 1 to 5.
    - Provide a concise explanation for your prediction in the explanation field that justifies the star rating.

    Use the structured output format below:
    {instructions}
    """

    parser = PydanticOutputParser(pydantic_object=StructuredOutput)

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions'],
        partial_variables={'instructions': parser.get_format_instructions()}
    )

    return prompt, parser