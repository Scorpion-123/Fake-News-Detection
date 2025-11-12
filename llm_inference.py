from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.chat_models import init_chat_model
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate

# ------------------------------------------------------
# 1️⃣ Define the Structured Output Schema
# ------------------------------------------------------
response_schemas = [
    ResponseSchema(
        name="news_type",
        description="Classify the news strictly as either 'Fake News' or 'Real News'."
    ),
    ResponseSchema(
        name="reason",
        description="Provide a brief, factual reason for the classification."
    )
]

# Create parser
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# ------------------------------------------------------
# 2️⃣ Initialize Top Tier model's from Groq & Gemini.
# ------------------------------------------------------
openai_model = init_chat_model(api_key="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", model="openai/gpt-oss-120b", model_provider="groq")

llama_model = init_chat_model(api_key="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", model="llama-3.3-70b-versatile", model_provider="groq")

gemini_model = init_chat_model(api_key="AIzaSyBhRxHqrOM9d0t6hl697O_DQczAJMxHO4M", model="gemini-2.5-flash", model_provider="google_genai")

# ------------------------------------------------------
# 3️⃣ Define Prompt Template
# ------------------------------------------------------
prompt = ChatPromptTemplate.from_template("""
                                        
You are an expert fact-checking and misinformation detection assistant trained to analyze news content. Your goal is to determine whether a given piece of text ("News Text") represents fake news (misleading, fabricated, or unverifiable) or real news (factual, credible, and verifiable).

You must make your decision based solely on linguistic, semantic, and stylistic cues — not by looking up external data. Evaluate the following aspects:

1. Factual Consistency: Does the text make verifiable or logically consistent claims?

2. Tone & Sensationalism: Does it use exaggerated, emotional, or alarmist language?

3. Source Credibility Indicators: Does it mention reliable sources, organizations, or evidence?

4. Writing Style: Is the writing coherent, objective, and grammatically sound (typical of professional journalism)?

5. Logical Validity: Do claims seem implausible or self-contradictory?

Based on your analysis, classify the text strictly as either:
FAKE NEWS — if the text is likely fabricated, exaggerated, misleading, or lacks factual support.
REAL NEWS — if the text appears factual, credible, and objective.

Then, provide a short, 3–4 line explanation summarizing your reasoning for the classification.
                                          
Here is the News Text:
"{news_text}"

You must respond in the following strict JSON format:
{format_instructions}
""")

# ------------------------------------------------------
# 4️⃣ Create a function to classify news
# ------------------------------------------------------
def classify_news_using_llm(model_name, news_text: str):

    chain = prompt | model_name | output_parser

    result = chain.invoke({"news_text": news_text, "format_instructions": format_instructions})

    return result