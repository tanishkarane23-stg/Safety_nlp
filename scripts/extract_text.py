import fitz  # PyMuPDF
import os
import json
import re

# Define relative paths based on our directory structure
RAW_DIR = os.path.join("data", "raw_pdfs")
OUT_DIR = os.path.join("data", "processed_json")

def clean_text(text):
    # 1. Catch multiline disclaimers using re.DOTALL to allow dots to match newlines
    disclaimer_pattern = r"Provided for information purpose only\..*?avoid recurrence of such incidents\."
    text = re.sub(disclaimer_pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. Catch all private use unicode characters (like \uf0a7, \uf0b7, \uf0fc) and replace with a dash
    text = re.sub(r'[\uf000-\uf0ff]', '-', text)
    
    # 3. Clean up excessive line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def process_pdfs():
    # Ensure the output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Iterate through all PDF files in the raw directory
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".pdf"):
            continue
            
        filepath = os.path.join(RAW_DIR, filename)
        doc = fitz.open(filepath)
        
        full_text = ""
        # Extract text from each page
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") + "\n"
            
        cleaned_text = clean_text(full_text)
        
        # Structure the data as a dictionary
        report_data = {
            "source_file": filename,
            "raw_text": cleaned_text
        }
        
        # Save the dictionary as a JSON file
        out_filename = filename.replace(".pdf", ".json")
        out_filepath = os.path.join(OUT_DIR, out_filename)
        
        with open(out_filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
            
    print(f"Extraction complete! Check the {OUT_DIR} folder.")

if __name__ == "__main__":
    process_pdfs()