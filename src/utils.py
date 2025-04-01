import os
import pdfplumber
from docx import Document

def read_text_file(file_path):
    """Reads a plain text file and returns its content."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def read_pdf_file(file_path):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def read_docx_file(file_path):
    """Extracts text from a DOCX file."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()

def read_resume_or_job_description(file_path):
    """
    Reads a resume or job description from a text, PDF, or DOCX file.

    Args:
        file_path (str): Path to the file.
    
    Returns:
        str: Extracted text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = file_path.lower().split(".")[-1]

    if file_extension == "txt":
        return read_text_file(file_path)
    elif file_extension == "pdf":
        return read_pdf_file(file_path)
    elif file_extension == "docx":
        return read_docx_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: txt, pdf, docx.")
