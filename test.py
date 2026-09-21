from typing import TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 1. ChatGroq ki jagah ChatOllama import karein
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END

# Setup Local Ollama Model
# Note: Iske liye background mein Ollama chal raha hona chahiye
model = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# JSON Output wale structured models setup karein
structured_model = ChatOllama(
    model="qwen2.5:3b",
    format="json",
    temperature=0
)

# Pydantic Schemas
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description='Sentiment of the review')

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

# Parsers for Structured JSON outputs
sentiment_parser = JsonOutputParser(pydantic_object=SentimentSchema)
diagnosis_parser = JsonOutputParser(pydantic_object=DiagnosisSchema)

class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# Node Functions
def find_sentiment(state: ReviewState):
    prompt = f"""For the following review find out the sentiment.
Return JSON strictly in this format: {{"sentiment": "positive"}} or {{"sentiment": "negative"}}

Review:
"{state["review"]}"
"""
    response = structured_model.invoke(prompt)
    parsed_res = sentiment_parser.parse(response.content)
    return {'sentiment': parsed_res['sentiment'].lower()}

def check_sentiment(state: ReviewState) -> Literal['positive_response', 'run_diagnosis']:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'

def positive_response(state: ReviewState):
    prompt = f'Write a warm thank-you message in response to this review:\n\n"{state["review"]}"\n Also, kindly ask the user to leave feedback on our website.'
    response = model.invoke(prompt).content
    return {'response': response}

def run_diagnosis(state: ReviewState):
    prompt = f"""Diagnose this negative review:
"{state["review"]}"

Return JSON strictly matching this schema:
- issue_type: one of ["UX", "Performance", "Bug", "Support", "Other"]
- tone: one of ["angry", "frustrated", "disappointed", "calm"]
- urgency: one of ["low", "medium", "high"]
"""
    response = structured_model.invoke(prompt)
    parsed_res = diagnosis_parser.parse(response.content)
    return {'diagnosis': parsed_res}

def nagetive_response(state: ReviewState):
    diagnosis = state['diagnosis']
    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content
    return {'response': response}

# Graph Setup
graph = StateGraph(ReviewState)

graph.add_node('find_sentiment', find_sentiment)
graph.add_node('positive_response', positive_response)
graph.add_node('run_diagnosis', run_diagnosis)
graph.add_node('nagetive_response', nagetive_response)

# Edges
graph.add_edge(START, 'find_sentiment')

# Conditional Routing
graph.add_conditional_edges('find_sentiment', check_sentiment)

# Connections
graph.add_edge('positive_response', END)
graph.add_edge('run_diagnosis', 'nagetive_response') 
graph.add_edge('nagetive_response', END)

# Workflow Compile
workflow = graph.compile()

# Execution Test
initial_state = {
    'review': "I’ve been trying to log in for over an hour now, and the app keeps freezing on the authentication screen. I even tried reinstalling it, but no luck. This kind of bug is unacceptable, especially when it affects basic functionality."
}

result = workflow.invoke(initial_state)

# Output Print Karein
print("--- RESULT ---")
print("Sentiment:", result.get('sentiment'))
print("Diagnosis:", result.get('diagnosis'))
print("\nResponse:\n", result.get('response'))