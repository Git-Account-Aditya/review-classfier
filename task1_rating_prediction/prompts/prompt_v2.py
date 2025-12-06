from langchain_core.prompts import PromptTemplate

def get_prompt_v2():
    SYSTEM_PROMPT = """
    You are an expert product review analyst. Your task is to analyze the sentiment of a product review and predict the star rating (1-5) given by the user.

    Review Content:
    {review_text}

    Reactions:
    {reactions}

    Rules to follow if needed:
    - Predict the star rating from 1 to 5.
    - Provide a concise explanation for your prediction in the explanation field that justifies the star rating.

    Return only the predicted star rating and explanation.
    """

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions']
    )

    return prompt