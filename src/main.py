import argparse
from src.api_llm import generate_questions_with_api
from src.local_llm import generate_questions_locally
from src.llm_agent import generate_questions_with_agent
from src.utils import read_resume_or_job_description
import json

def main():
    parser = argparse.ArgumentParser(description="Generate interview questions from a resume and job description.")
    parser.add_argument("--resume", type=str, required=True, help="Path to resume file (txt, pdf, docx)")
    parser.add_argument("--job", type=str, required=True, help="Path to job description file (txt, pdf, docx)")
    parser.add_argument("--mode", type=str, choices=["api", "local", "agent"], required=True, help="Choose LLM mode")

    args = parser.parse_args()

    resume_text = read_resume_or_job_description(args.resume)
    job_text = read_resume_or_job_description(args.job)

    if args.mode == "api":
        result = generate_questions_with_api(resume_text, job_text)
    elif args.mode == "local":
        result = generate_questions_locally(resume_text, job_text)
    elif args.mode == "agent":
        result = generate_questions_with_agent(resume_text, job_text)  # Calls the PydanticAI-based structured agent

    # Print result based on mode
    print("\nGenerated Interview Questions:\n")

    if args.mode == "agent":
        # Convert structured Pydantic object to JSON-friendly format
        output_data = result.model_dump()
        print(json.dumps(output_data, indent=2))
    else:
        print(result)

if __name__ == "__main__":
    main()
