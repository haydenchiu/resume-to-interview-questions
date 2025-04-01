from pydantic import BaseModel
from typing import List
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

class InterviewQuestions(BaseModel):
    """Schema for structured interview questions output."""
    technical_questions: List[str]
    behavioral_questions: List[str]
    situational_questions: List[str]

ollama_model = OpenAIModel(
    model_name='llama3.3', provider=OpenAIProvider(base_url='http://localhost:11434/v1')
)

# Initialize the PydanticAI Agent
interview_agent = Agent(
    model=ollama_model,
    result_type=InterviewQuestions,  # Ensures output follows schema
    system_prompt=(
        "You are an AI hiring expert. Given a resume and job description, "
        "generate relevant interview questions categorized as:\n"
        "- Technical Questions\n"
        "- Behavioral Questions\n"
        "- Situational Questions"
    ),
)

def generate_questions_with_agent(resume: str, job_description: str) -> InterviewQuestions:
    """Generates structured interview questions using PydanticAI Agent."""
    
    input_text = f"Resume:\n{resume}\n\nJob Description:\n{job_description}"
    
    # Run the agent and validate structured output
    result = interview_agent.run_sync(input_text)
    
    return result.data  # Returns structured InterviewQuestions object
