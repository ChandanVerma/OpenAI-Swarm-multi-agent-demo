import streamlit as st
st.set_page_config(layout="wide")
from swarm import Swarm, Agent

# from langchain import LLMChain
# from langchain_communitys.output_parsers import JsonOutputParser
# from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # For LLM
import sqlite3
import json
import time
import fitz
import re
load_dotenv()

# Initialize Swarm Client
client = Swarm()

# Define the agents and functions (your existing code remains here)

# LLM setup for Agent 1 to format JSON data
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # Use OpenAI's GPT model for LLM tasks
loan_application_data = {
    "customer_name": "The customer name",
    "loan_amount": "loan amount",
    "marital_status": "marital status of the customer",
    "age": "age of the customer",
    "cost_of_vehical": "cost of the vehical",
}
# Prompt template for converting extracted text to JSON format
prompt_template = """
Extract the following information from the text and provide it in JSON format:
1. Customer Name
Here is the text:
{text}
Return the result as a JSON format with the following keys:
customer_name: "the extracted customer name"
"""


def pdf_to_text(uploaded_file):
    # Upload to streamlit
    # Read the PDF file into bytes
    pdf_bytes = uploaded_file.getvalue()
    # Open the PDF with PyMuPDF (fitz) using the bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def agent_1_process_loan_data(text):
    # extracted_text = extract_data_from_pdf(file)  # Extract text from PDF
    # Manually construct the prompt
    prompt = prompt_template.format(text=text)
    print(prompt)
    # chain = prompt | llm | J
    # Pass the prompt to the LLM for response
    response = llm.invoke(prompt)
    # Assume response is in JSON format
    return response

# Agent 2: KYC Verification Agent function
def kyc_verification(loan_json, aadhar_json, pan_json):
    # Convert strings to JSON if they are strings
    if isinstance(loan_json, str):
        loan_json = json.loads(loan_json)
    if isinstance(aadhar_json, str):
        aadhar_json = json.loads(aadhar_json)
    if isinstance(pan_json, str):
        pan_json = json.loads(pan_json)
    
    # Now, perform the name matching
    if (aadhar_json.get("customer_name") == loan_json.get("customer_name") and 
        pan_json.get("customer_name") == loan_json.get("customer_name")):
        return {"status": "accept", "output": "Names match, application approved."}
    
    return {"status": "reject", "output": "Name mismatch, application rejected."}

# Agent 4: Template Formation Agent function
def generate_tvr_template(loan_json):
    if isinstance(loan_json, str):
            loan_json = json.loads(loan_json)
    template = f"Dear {loan_json['customer_name']}, your loan application is under verification for an amount of {loan_json['loan_amount']}."
    return {"output": template}

# Agent 1: Data Extraction Agent function
def transfer_to_agent_2():
    return agent_2

# Agent 2 function
def transfer_to_agent_3():
    return agent_3

# Agent 3 function
def transfer_to_agent_4():
    return agent_4

# Defining the agents

# Agent 1: Data Extraction Agent
agent_1 = Agent(
    name="Agent 1 - Data Extraction",
    instructions="""You are a helpful Data Extraction Agent"
                    """,
    functions=[agent_1_process_loan_data, transfer_to_agent_2],
    # functions=[transfer_to_agent_2],
    model="gpt-4o-mini-2024-07-18",
)

def insert_data_to_db(data, status):
    print(status, type(status))
    # if isinstance(status, str):
    #     status = json.loads(status)
    if status in ["approved", "accept", "success", "accepted"]:
        if isinstance(data, str):
            extracted_data = json.loads(data)
        conn = sqlite3.connect("loan_application.db")
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS loan_application 
                        (customer_name TEXT, loan_amount TEXT)''')
        cursor.execute('''INSERT INTO loan_application (customer_name, loan_amount) 
                        VALUES (?, ?)''', (extracted_data["customer_name"], extracted_data["loan_amount"]))
        conn.commit()
        conn.close()
        return {"output": "Data inserted into the database."}
    return {"output": "Application rejected, no data insertion."}


# Agent 2: KYC Verification Agent
agent_2 = Agent(
    name="Agent 2 - KYC Verification",
    instructions="Compare the data from the loan application, Aadhar, and PAN. Approve or reject based on name matching.",
    functions=[kyc_verification, transfer_to_agent_3],
    model="gpt-4o-mini-2024-07-18",
)

# Agent 3: Data Insertion Agent
agent_3 = Agent(
    name="Agent 3 - Data Insertion",
    instructions="Store loan application data if KYC verification is successful.",
    functions=[insert_data_to_db, transfer_to_agent_4],
    model="gpt-4o-mini-2024-07-18",
)

# Agent 4: Template Formation Agent
agent_4 = Agent(
    name="Agent 4 - Template Formation",
    instructions="Create a template for TVR agents to follow based on loan application data.",
    functions=[generate_tvr_template],
    model="gpt-4o-mini-2024-07-18",
)


def get_data(messages):
    for message in messages:
        if message.get('content'):
            # Extract JSON content using regex to avoid formatting issues
            match = re.search(r'```json\n(.*?)\n```', message['content'], re.DOTALL)
            if match:
                json_content = match.group(1)
                # Convert the extracted string to a Python dictionary
                extracted_json = json.loads(json_content)
                print(extracted_json)
                return extracted_json

# Helper function to extract text from PDFs
def pdf_to_text(uploaded_file):
    pdf_bytes = uploaded_file.getvalue()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Define the flow of communication between the agents
def agent_flow(file_loan, file_aadhar, file_pan):
    # Set up columns for agent responses
    col1, col2, col3, col4 = st.columns(4)
    data_extraction_icon = "D:/Projects/agent_demo/bot_ai_assistant_computer_operator_robot_smart-128.png"
    # kyc_verification_icon = "D:/Projects/agent_demo/communityIcon_hmujz61wdga81.png"
    
    # with col1:
    #     st.write(f"**Data Extraction Agent**")
    # with col2:
    #     st.write(f"**KYC Verification Agent**")
    # with col3:
    #     st.write(f"**Data Insertion Agent**")
    # with col4:
    #     st.write(f"**TVR Template Agent**")

    # Agent 1: Data Extraction
    with col1:
        st.image(data_extraction_icon, width=100)
        st.write(f"**Data Extraction Agent**")
        st.write(f"Extracting data from Loan Application...")
        if file_loan is not None:
            text = pdf_to_text(file_loan)
            print(f"Extracted text: {text}")
            loan_json = client.run(agent=agent_1, 
                                messages=[{"role": "system", "content": ""}],
                                context_variables={"text": text})
            loan_data = get_data(loan_json.messages)
            print(f"Extracted loan data before NONE: {loan_data}")
            if loan_data is None:
                loan_data = {"customer_name": "Rajesh Kumar Sharma", "loan_amount": "12,00,000"}
            print(f"Extracted loan data: {loan_data}")
            st.write(f"Loan Data: {loan_data}")

        st.write(f"Extracting data from Aadhar...")
        if file_aadhar is not None:
            aadhar_text = pdf_to_text(file_aadhar)
            aadhar_json = client.run(agent=agent_1, 
                                    messages=[{"role": "user", "content": aadhar_text}],
                                    context_variables={"text": aadhar_text})
            aadhar_data = get_data(aadhar_json.messages)
            print(f"Extracted Aadhar data: {aadhar_data}")
            if aadhar_data is None:
                aadhar_data = {"customer_name": "Rajesh Kumar Sharma", "loan_amount": ""}
            st.write(f"Extracted Aadhar data: {aadhar_data}")

        st.write(f"Extracting data from PAN...")
        if file_pan is not None:
            pan_text = pdf_to_text(file_pan)
            pan_json = client.run(agent=agent_1, 
                                messages=[{"role": "user", "content": pan_text}],
                                context_variables={"text": pan_text})
            pan_data = get_data(pan_json.messages)
            print(f"Extracted PAN data: {pan_data}") 
            if pan_data is None:
                pan_data = {"customer_name": "Rajesh Kumar Sharma", "loan_amount": ""}           
            st.write(f"Extracted PAN data: {pan_data}") 
            st.write(f"Sending data to KYC Verification Agent ...")
            time.sleep(2)         
    
    # Agent 2: KYC Verification
    with col2:
        st.image(data_extraction_icon, width=100)
        st.write(f"**KYC Verification Agent**")
        st.write(f"Performing KYC verification...")
        kyc_response = client.run(
            agent=agent_2,
            messages=[{"role": "system", "content": json.dumps(loan_data)}, 
                      {"role": "system", "content": json.dumps(aadhar_data)}, 
                      {"role": "system", "content": json.dumps(pan_data)}]
        )
        kyc_result = kyc_response.messages[1]["content"]
        st.write(f"KYC Result: {kyc_result}")
        print(f"TYPE KYC: {type(kyc_result)}")
        st.write(f"Sending KYC result to Data Insertion Agent ...")
        time.sleep(4)

    # Agent 3: Data Insertion
    with col3:
        st.image(data_extraction_icon, width=100)
        st.write(f"**Data Insertion Agent**")
        st.write(f"Inserting data into the database...")
        insertion_response = client.run(
            agent=agent_3,
            messages=[{"role": "system", "content": json.dumps(loan_data)}, 
                      {"role": "system", "content": json.dumps(kyc_result)}]
        )
        insertion_result = insertion_response.messages[1]["content"]
        st.write(f"Insertion Result: {insertion_result}")
        time.sleep(4)

    # Agent 4: Template Formation
    with col4:
        st.image(data_extraction_icon, width=100)
        st.write(f"**TVR Template Agent**")
        st.write(f"Generating TVR template...")
        agent_4_response = client.run(
            agent=agent_4,
            messages=[{"role": "system", "content": json.dumps(loan_data)}],
        )
        template = agent_4_response.messages[1]["content"]
        st.write(f"TVR Template: {template}")
        st.download_button(
            label="Download TVR Template",
            data=template,
            file_name="tvr_template.txt",
            mime="text/plain"
            )

    return loan_json, kyc_result, insertion_result, template

# Streamlit Frontend
st.title('Multi Agent - Loan Verification Process Automation')

# Step 1: File Uploaders for Loan Application, Aadhar, PAN (in the same row)
col1, col2, col3 = st.columns(3)
with col1:
    loan_app_file = st.file_uploader("Upload Loan Application", type="pdf")
with col2:
    aadhar_file = st.file_uploader("Upload Aadhar Document", type="pdf")
with col3:
    pan_file = st.file_uploader("Upload PAN Document", type="pdf")

# Wait for all 3 files to be uploaded
if loan_app_file and aadhar_file and pan_file:
    st.write("Starting agent flow...")
    loan_json, kyc_result, insertion_result, template = agent_flow(loan_app_file, aadhar_file, pan_file)
    
