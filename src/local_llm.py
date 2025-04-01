from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

def generate_questions_locally(resume, job_description):
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

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("mps")  
    output = model.generate(**inputs, max_new_tokens=1000)  

    return tokenizer.decode(output[0], skip_special_tokens=True)

def sliding_window_generate(input_text, model, tokenizer, max_chunk_size=512, stride=256, max_new_tokens=500):
    """Splits long input text into overlapping chunks, processes them, and concatenates output"""
    
    # Tokenize input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")[0]

    # Store generated responses
    generated_texts = []

    # Process in chunks with overlap
    for i in range(0, len(input_ids), stride):
        chunk = input_ids[i : i + max_chunk_size]  # Take a window of `max_chunk_size`
        if len(chunk) == 0:
            break  # Stop if no more tokens to process

        # Convert chunk back to tensor
        inputs = {"input_ids": chunk.unsqueeze(0).to("mps")}

        # Generate text for this chunk
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Decode and store
        generated_texts.append(tokenizer.decode(output[0], skip_special_tokens=True))

        # Stop if we've processed the whole input
        if i + max_chunk_size >= len(input_ids):
            break

    # Combine results
    return " ".join(generated_texts)

def generate_questions_with_sliding_window(resume, job_description):
    combined_text = f"Resume:\n{resume}\n\nJob Description:\n{job_description}"
    return sliding_window_generate(combined_text, model, tokenizer, max_chunk_size=1024, stride=512, max_new_tokens=300)


if __name__ == '__main__':

    print(generate_questions_locally(resume, job_description))