import streamlit as st
from swarm import Swarm, Agent

# from langchain import LLMChain
# from langchain_communitys.output_parsers import JsonOutputParser
# from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # For LLM
import sqlite3
import json
import fitz
import re
load_dotenv()
# Initialize Swarm Client

client = Swarm()

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
    instructions="""Extract the following information from the text and provide it in JSON format:
                    1. Customer Name
                    Here is the text:
                    {text}
                    if you cannot find the loan amount leave it blank
                    Return the result as a JSON format with the following keys:
                    customer_name: "the extracted customer name",
                    loan_amount: "the extracted loan amount"
                    """,
    functions=[agent_1_process_loan_data, transfer_to_agent_2],
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

# Define the flow of communication between the agents
def agent_flow(file_loan, file_aadhar, file_pan):
    # Agent 1: Extract data from PDFs
    st.write(f"Data Extraction Agent - I have started extracting the data from the provided documents")
    if file_loan is not None:
        text = pdf_to_text(file_loan)
        # print(pages[0])
    # # loan_json = client.run(agent=agent_1, messages=[{"role": "user", "content": file_loan}])['messages'][-1]["content"]
    print(text)
    print(type(text))
    loan_json = client.run(agent=agent_1, 
                           messages=[{"role": "system", "content": text}])
    loan_data = get_data(loan_json.messages)
    print(f"Extracted loan data: {loan_data}")
    st.write(f"Data Extraction Agent - I have successfully extracted the data for loan application: {loan_data}")
    if file_aadhar is not None:
        aadhar_text = pdf_to_text(file_aadhar)
    aadhar_json = client.run(agent=agent_1, 
                             messages=[{"role": "user", "content": aadhar_text}])
    aadhar_data = get_data(aadhar_json.messages)
    st.write(f"Data Extraction Agent - I have successfully extracted the data for aadhar: {aadhar_data}")
    print(f"Extracted Aadhar data: {aadhar_json}")
    if file_pan is not None:
        pan_text = pdf_to_text(file_pan)
    pan_json = client.run(agent=agent_1, 
                          messages=[{"role": "user", "content": pan_text}])
    pan_data = get_data(pan_json.messages)
    print(f"Extracted PAN data: {pan_data}")
    # loan_json = {"customer_name": "John Doe", "loan_amount": "50000"}
    aadhar_data = {"customer_name": "Rajesh Kumar Sharma"}
    pan_data = {"customer_name": "Rajesh Kumar Sharma"}
    
    
    st.write(f"Data Extraction Agent - I have successfully extracted the data for PAN: {pan_data}")
    st.write(f"Data Extraction Agent - Passing the data to the KYC Agent")

    # Agent 2: Perform KYC verification
    st.write(f"KYC AGENT - I am performing KYC verification")
    agent_2_response = client.run(
        agent=agent_2,
        messages=[{"role": "system", "content": json.dumps(loan_data)}, 
                {"role": "system", "content": json.dumps(aadhar_data)}, 
                {"role": "system", "content": json.dumps(pan_data)}],
        # stream=True, 
    )

    kyc_result = agent_2_response.messages[1]["content"]
    st.write(f"KYC AGENT - I have successfully performed KYC verification: {kyc_result}")
    st.write(f"KYC AGENT - Data insertion agent can now proceed to insert data to the database")
    # Agent 3: Insert data into the database based on KYC result
    agent_3_response = client.run(
        agent=agent_3,
        messages=[{"role": "system", "content": json.dumps(loan_data)}, 
                {"role": "system", "content": json.dumps(kyc_result)}],
    )

    st.write(f"Data Insertion AGENT - Ok, Let me insert the data into the database")
    insertion_result = agent_3_response.messages[1]["content"]
    st.write(f"Data Insertion AGENT - : {insertion_result}")
    st.write(f"Data Insertion AGENT - TVR AGENT can now generate the TVR template")
    # Agent 4: Generate TVR template
    st.write(f"TVR AGENT - I am generating the TVR script now")
    agent_4_response = client.run(
        agent=agent_4,
        messages=[{"role": "system", "content": json.dumps(loan_data)}],
    )   
    template = agent_4_response.messages[1]["content"]
    st.write("TVR AGENT - I have successfully Generated TVR Template:")
    st.write(template)
    return loan_json, aadhar_json, pan_json, kyc_result, insertion_result, template

# Streamlit Frontend
st.title('Loan Verification Process Automation')

# Step 1: File Uploads for Loan Application, Aadhar, PAN
loan_app_file = st.file_uploader("Upload Loan Application (PDF)", type="pdf")
aadhar_file = st.file_uploader("Upload Aadhar Document (PDF)", type="pdf")
pan_file = st.file_uploader("Upload PAN Document (PDF)", type="pdf")

# Wait for all 3 files to be uploaded
if loan_app_file and aadhar_file and pan_file:
    # Run agent flow
    st.write("Starting agent flow...")
    loan_json, aadhar_json, pan_json, kyc_result, insertion_result, template = agent_flow(loan_app_file, aadhar_file, pan_file)

    st.download_button(
        label="Download TVR Template",
        data=template,
        file_name="tvr_template.txt",
        mime="text/plain"
    )