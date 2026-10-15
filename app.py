import streamlit as st
import pandas as pd
import plotly.express as px
import pymupdf
import io
import os
from pydantic import BaseModel, Field
from typing import Literal
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# 1. Page Configuration (Strict Corporate Styling)
st.set_page_config(page_title="OIL SIF Engine", layout="wide")

# Custom CSS for polished command-center styling
st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #1C2833;
        border: 1px solid #2C3E50;
        padding: 5% 10% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

st.title("OIL: SIF Precursor Detection Engine")
st.markdown("AI/NLP Pipeline for analyzing Unsafe-Act, Unsafe-Condition, and Near-Miss Reports")

# 2. Define Strict LLM Architecture with Chain-of-Thought Pydantic Schema
class IncidentReportData(BaseModel):
    incident_analysis_reasoning: str = Field(description="Briefly summarize the root cause of the incident and explicitly state which IOGP rule applies and why.")
    is_sif_potential: bool = Field(description="True if the incident involves a fatality, serious injury, or high-energy near miss. False otherwise.")
    sif_risk_score: int = Field(description="Rate the potential severity/risk from 1 (lowest) to 10 (highest).")
    iogp_rule_violated: Literal[
        "Working at Height", 
        "Ground Disturbance", 
        "Energy Isolation", 
        "Confined Space", 
        "Hot Work", 
        "Work Authorization", 
        "Line of Fire", 
        "General Safety"
    ] = Field(description="Must exactly match the rule identified in the reasoning step.")
    activity: str = Field(description="The specific work task being performed.")
    location: str = Field(description="The physical location of the incident.")
    barrier_failure: str = Field(description="The specific safety mechanism, procedure, or physical barrier that failed.")
    incident_date: str = Field(description="Date of the incident in YYYY-MM-DD format. If not mentioned, output 'Unknown'.")

@st.cache_resource
def load_model():
    llm = ChatOllama(model="llama3.2", temperature=0)
    prompt = PromptTemplate.from_template(
        """You are an expert Oil & Gas HSE investigator. Read the incident report and follow these strict guidelines.
        
        STEP 1: Write a 1-2 sentence analysis explaining the root cause and matching it to an IOGP rule.
        STEP 2: Extract the remaining fields. 

        CRITICAL RULE 1: SIF POTENTIAL & RISK SCORE
        - is_sif_potential: Set to true IF the report mentions fatality, death, succumbed, major injury, explosion, or uncontrolled release. 
        - sif_risk_score: Assign a severity score from 1-10. Fatalities/explosions are 9-10. High-energy near-misses are 7-8. Minor trips/spills are 1-4.

        CRITICAL RULE 2: IOGP RULE DEFINITIONS
        - Working at Height: Falls from elevated surfaces, roofs, platforms, or ladders.
        - Ground Disturbance: Trenching, excavating, digging, or soil collapse/cave-ins.
        - Energy Isolation: Uncontrolled release of pressure, gas, liquid, or electrocution.
        - Confined Space: Incidents inside enclosed tanks, vessels, or pipes.
        - Hot Work: Welding, cutting, or sparks.
        - Line of Fire: Struck by moving machinery, collapsing structures (masts), or dropped objects.
        - Work Authorization: Proceeding without proper permits, lack of SOPs, or bypassing controls.
        - General Safety: ONLY use if absolutely none of the above apply.
        
        CRITICAL RULE 3: INCIDENT DATE
        Extract the date of the incident and format it strictly as YYYY-MM-DD. If no date is present, output 'Unknown'.

        Incident Report:
        {report_text}
        """
    )
    return prompt | llm.with_structured_output(IncidentReportData)

nlp_chain = load_model()

# Convert results DataFrame to CSV for bulk download
def convert_df_to_csv(df_to_convert):
    return df_to_convert.to_csv(index=False).encode('utf-8')

# 3. Create Navigation Tabs
tab1, tab2 = st.tabs(["Historical Dashboard", "Live Report Analysis"])

# --- TAB 1: DASHBOARD ---
with tab1:
    @st.cache_data(ttl=5)
    def load_data():
        try:
            return pd.read_csv("data/master_sif_data.csv")
        except FileNotFoundError:
            return pd.DataFrame()

    df = load_data()

    if not df.empty:
        st.subheader("Safety Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_reports = len(df)
        sif_potential_count = df['is_sif_potential'].sum() if 'is_sif_potential' in df.columns else 0
        avg_risk_score = round(df['sif_risk_score'].mean(), 1) if 'sif_risk_score' in df.columns else 0.0
        
        col1.metric("Total Reports Analyzed", total_reports)
        col2.metric("High SIF Potential", sif_potential_count)
        col3.metric("Review Required", total_reports - sif_potential_count)
        col4.metric("Avg SIF Risk Score", f"{avg_risk_score} / 10")

        # Trend Analysis Chart
        st.markdown("---")
        st.subheader("Trend Analysis: Incidents Over Time")
        if 'incident_date' in df.columns:
            trend_df = df[df['incident_date'] != 'Unknown'].copy()
            if not trend_df.empty:
                trend_df['incident_date'] = pd.to_datetime(trend_df['incident_date'], errors='coerce')
                trend_df = trend_df.dropna(subset=['incident_date'])
                if not trend_df.empty:
                    # Group by month-year
                    trend_counts = trend_df.groupby(trend_df['incident_date'].dt.to_period('M')).size().reset_index(name='Incident Count')
                    trend_counts['incident_date'] = trend_counts['incident_date'].astype(str)
                    fig_trend = px.line(trend_counts, x='incident_date', y='Incident Count', markers=True, color_discrete_sequence=['#3498DB'])
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.caption("Valid date formats not found for trend extraction.")
            else:
                st.caption("Insufficient date data available for trend analysis.")

        st.markdown("---")
        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.subheader("IOGP Life-Saving Rules Violated")
            if 'iogp_rule_violated' in df.columns:
                rule_counts = df['iogp_rule_violated'].value_counts().reset_index()
                rule_counts.columns = ['Rule', 'Count']
                fig_rules = px.bar(rule_counts, x='Count', y='Rule', orientation='h', color_discrete_sequence=['#F4D03F'])
                fig_rules.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_rules, use_container_width=True)

        with col_chart2:
            st.subheader("SIF Potential Distribution")
            if 'is_sif_potential' in df.columns:
                df['SIF Status'] = df['is_sif_potential'].map({True: 'SIF Potential', False: 'Non-SIF', 'true': 'SIF Potential', 'false': 'Non-SIF'})
                sif_counts = df['SIF Status'].value_counts().reset_index()
                sif_counts.columns = ['Status', 'Count']
                color_map = {'SIF Potential': '#E74C3C', 'Non-SIF': '#3498DB'}
                fig_sif = px.pie(sif_counts, names='Status', values='Count', hole=0.4, color='Status', color_discrete_map=color_map)
                st.plotly_chart(fig_sif, use_container_width=True)

        st.markdown("---")
        st.subheader("Top Risk Factors & Barrier Failures")
        if 'barrier_failure' in df.columns:
            barrier_counts = df['barrier_failure'].value_counts().reset_index()
            barrier_counts.columns = ['Barrier Failure Factor', 'Frequency']
            fig_barriers = px.bar(barrier_counts, x='Frequency', y='Barrier Failure Factor', orientation='h', color_discrete_sequence=['#E74C3C'])
            fig_barriers.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_barriers, use_container_width=True)

        st.markdown("---")
        st.subheader("Extracted Precursor Patterns")
        if 'sif_risk_score' in df.columns and 'incident_date' in df.columns:
            st.dataframe(df[['incident_date', 'source_file', 'sif_risk_score', 'activity', 'location', 'barrier_failure', 'is_sif_potential']], use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    else:
        st.info("No historical data found. Process reports in the Live Analysis tab to populate this dashboard.")

# --- TAB 2: LIVE ANALYSIS ---
with tab2:
    st.subheader("Process New Incident Reports")
    st.markdown("Upload unstructured safety reports (PDF) to extract SIF precursors, risk scores, and embedded mishap diagrams.")
    
    uploaded_files = st.file_uploader("Upload Safety Reports", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Analyze Batch with Llama 3.2", type="primary"):
            batch_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing ({i+1}/{total_files}): {uploaded_file.name}...")
                try:
                    pdf_bytes = uploaded_file.read()
                    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    
                    extracted_text = ""
                    for page in doc:
                        extracted_text += page.get_text()
                    
                    if not extracted_text.strip():
                        st.warning(f"Skipped {uploaded_file.name}: Could not extract text.")
                        continue
                    
                    # --- ADD THIS SAFEGUARD ---
                    # Limit to ~12,000 characters (roughly 3,000 tokens)
                    # This prevents CUDA buffer overruns and GPU OOM crashes on massive PDFs
                    if len(extracted_text) > 12000:
                        extracted_text = extracted_text[:12000] + "\n...[TEXT TRUNCATED FOR CONTEXT LIMIT]..."
                    
                    report_images = []
                    for page_idx, page in enumerate(doc):
                        image_list = page.get_images(full=True)
                        for img_idx, img in enumerate(image_list):
                            xref = img[0]
                            width = img[2]
                            height = img[3]
                            
                            aspect_ratio = width / height if height > 0 else 0
                            
                            if aspect_ratio > 2.5 or (width < 350 and height < 350):
                                continue
                            
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            if len(image_bytes) > 15000: 
                                report_images.append(image_bytes)

                    result = nlp_chain.invoke({"report_text": extracted_text})
                    
                    row = result.model_dump()
                    row["source_file"] = uploaded_file.name
                    batch_results.append(row)
                    
                    with st.expander(f"Report Results: {uploaded_file.name}", expanded=True):
                        st.markdown(f"**AI Reasoning:** {result.incident_analysis_reasoning}")
                        st.markdown("---")
                        
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.info(f"SIF Potential: {'High Risk' if result.is_sif_potential else 'Low Risk'} (Score: {result.sif_risk_score}/10)")
                            st.warning(f"IOGP Rule: {result.iogp_rule_violated}")
                        with r_col2:
                            st.write(f"Date: {result.incident_date}")
                            st.write(f"Location: {result.location}")
                            st.write(f"Activity: {result.activity}")
                            
                        st.error(f"Barrier Failure Risk Factor: {result.barrier_failure}")
                        
                        if report_images:
                            st.markdown("#### Extracted Visual Evidence")
                            img_cols = st.columns(min(len(report_images), 3))
                            for idx, img_b in enumerate(report_images[:3]):
                                with img_cols[idx % 3]:
                                    st.image(img_b, caption=f"Figure {idx+1} from {uploaded_file.name}", use_container_width=True)
                        else:
                            st.caption("No embedded diagrams detected.")
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text("Batch processing complete.")
            
            if batch_results:
                res_df = pd.DataFrame(batch_results)
                
                master_path = "data/master_sif_data.csv"
                os.makedirs("data", exist_ok=True)
                
                if os.path.exists(master_path):
                    existing_df = pd.read_csv(master_path)
                    updated_df = pd.concat([existing_df, res_df], ignore_index=True)
                    updated_df.drop_duplicates(subset=['source_file'], keep='last', inplace=True)
                    updated_df.to_csv(master_path, index=False)
                else:
                    res_df.to_csv(master_path, index=False)
                
                st.success("Reports added to Historical Database. Refresh Dashboard to see updated trends.")

                st.markdown("---")
                csv_data = convert_df_to_csv(res_df)
                st.download_button(
                    label="Download Batch Results (CSV)",
                    data=csv_data,
                    file_name="batch_sif_extractions.csv",
                    mime="text/csv",
                )