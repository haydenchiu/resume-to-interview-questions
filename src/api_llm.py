import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Store in an .env file

def generate_questions_with_api(resume, job_description):
    prompt = f"""
    You are an AI hiring expert. Given a resume and job description, generate **relevant interview questions**.

    **Resume:**  
    {resume}

    **Job Description:**  
    {job_description}

    ### **Output Format:**  
    **Technical Questions:**  
    1.  
    2.  
    3.  

    **Behavioral Questions:**  
    1.  
    2.  
    3.  

    **Situational Questions:**  
    1.  
    2.  
    3.  
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "system", "content": "You are a hiring expert."},
                  {"role": "user", "content": prompt}]
    )
    
    return response["choices"][0]["message"]["content"]
