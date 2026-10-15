import json
import os
import glob
import pandas as pd
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Define the strict output schema
class IncidentReportData(BaseModel):
    is_sif_potential: bool = Field(description="True if the incident involves a fatality, serious injury, or high-energy near miss. False otherwise.")
    iogp_rule_violated: str = Field(description="The closest applicable IOGP Life-Saving Rule (e.g., 'Energy Isolation', 'Working at Height', 'Hot Work', 'Line of Fire'). If none, output 'None'.")
    activity: str = Field(description="The specific work task being performed (e.g., 'crane lifting', 'scaffolding erection', 'valve maintenance').")
    location: str = Field(description="The physical location of the incident (e.g., 'pump station', 'refinery crude unit', 'storage tank').")
    barrier_failure: str = Field(description="The specific safety mechanism, procedure, or physical barrier that failed (e.g., 'missing shoring', 'choked valve', 'lack of PPE').")

# 2. Set up the FREE local Ollama model (Make sure Ollama app is running in your background)
llm = ChatOllama(model="llama3.2", temperature=0)
structured_llm = llm.with_structured_output(IncidentReportData)

# (3 errors fixed) Create the prompt instruction for JSON output
prompt = PromptTemplate.from_template(
    """You are an expert Oil & Gas HSE investigator. Read the specific incident report below very carefully and extract its unique details. 
    DO NOT copy the placeholder examples. Ensure text fields contain descriptive text extracted strictly from the report.

    Example Output Format:
    - is_sif_potential: <boolean, true or false>
    - iogp_rule_violated: "<Identify the rule, e.g., Confined Space, Hot Work, etc.>"
    - activity: "<Summarize the exact task from the text>"
    - location: "<Extract the specific location from the text>"
    - barrier_failure: "<Describe what physically or procedurally failed based on the text>"

    Incident Report:
    {report_text}
    """
)

chain = prompt | structured_llm

def process_all_reports():
    results = []
    json_files = glob.glob("data/processed_json/*.json")
    
    if not json_files:
        print("No JSON files found in data/processed_json/. Please run extract_text.py first.")
        return

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        print(f"Processing locally with Llama 3.2: {data['source_file']}...")
        try:
            # Extract structured data using the local LLM
            extracted_data = chain.invoke({"report_text": data["raw_text"]})
            
            # Convert to dictionary and add the source filename
            row = extracted_data.model_dump()
            row["source_file"] = data["source_file"]
            results.append(row)
            
        except Exception as e:
            print(f"Error processing {data['source_file']}: {e}")
            
    # Save to a master CSV for the dashboard
    df = pd.DataFrame(results)
    os.makedirs("data", exist_ok=True)
    # clean the dataframe to ensure no broken or missing columns
    df = pd.DataFrame(results)
    
    # Sanitize data to fix local model type-slips
    if not df.empty:
        # If barrier_failure accidentally captured a boolean, replace it
        df['barrier_failure'] = df['barrier_failure'].astype(str)
        df.loc[df['barrier_failure'].isin(['True', 'False', 'true', 'false']), 'barrier_failure'] = 'Unspecified Barrier Failure'
        
        # Ensure iogp_rule_violated is a clean string
        df['iogp_rule_violated'] = df['iogp_rule_violated'].astype(str)
        df.loc[df['iogp_rule_violated'].isin(['True', 'False', 'true', 'false']), 'iogp_rule_violated'] = 'General Safety'

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/master_sif_data.csv", index=False)
    df.to_csv("data/master_sif_data.csv", index=False)
    print("Batch processing complete! Saved to data/master_sif_data.csv")

if __name__ == "__main__":
    process_all_reports()