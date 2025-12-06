from langchain_core.prompts import PromptTemplate


def get_prompt_v1():
    SYSTEM_PROMPT = """
    You are an expert sentiment analysis system. Your task is to predict the star rating (1-5) for a given review.

    Analyze the following review properties:
    Review Text: {review_text}
    Reactions: {reactions}

    Predict the star rating (1-5) for the given review and provide a concise explanation for your prediction in the explanation field that justifies the star rating.
    """

    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=['review_text', 'reactions']
    )
    return prompt
    
    