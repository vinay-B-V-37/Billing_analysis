import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_dataset(file_path):
    """
    Reads the dataset from the provided file path.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading the dataset: {e}")
        return None

def prompt_gpt_for_criteria(columns):
    """
    Prompts GPT-3.5-turbo for anomaly detection criteria.
    """
    prompt = f"""
    You are given a  billing dataset with the following columns: {', '.join(columns)}.
    Provide a detailed list of criteria to detect the following anomalies in the dataset:
    - Duplicate Billings
    - High or Low Billings
    - Invalid Service Types
    - Billing for Suspended Accounts

    For each anomaly, provide the exact steps to identify them based on the given columns.
    Provide a detailed report on the anomalies detected, including specific entries that are anomalous and explanations for why they were flagged.
    """

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in telecom billing anomaly detection."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def find_similar_column(columns, target):
    """
    Finds a similar column name in the dataset columns.
    """
    for column in columns:
        if column.lower() == target.lower():
            return column
    return None

def detect_anomalies(df, criteria):
    """
    Detects anomalies in the dataset based on the provided criteria.
    """
    columns = df.columns
    anomalies = {
        "Duplicate Billings": [],
        "High or Low Billings": [],
        "Invalid Service Types": [],
        "Billing for Suspended Accounts": []
    }

    # Find correct column names
    account_id_col = find_similar_column(columns, 'Customer ID')
    date_col = find_similar_column(columns, 'Date')
    billing_amount_col = find_similar_column(columns, 'Billing Amount')
    service_type_col = find_similar_column(columns, 'Service Type')
    data_usage_col = find_similar_column(columns, 'Data Usage (GB)')
    account_status_col = find_similar_column(columns, 'Account Status')
    plan_type_col = find_similar_column(columns, 'Plan Type')
    payment_status_col = find_similar_column(columns, 'Payment Status')

    # Duplicate Billings
    if account_id_col and date_col:
        duplicate_billings = df[df.duplicated(subset=[account_id_col, date_col], keep=False)]
        anomalies["Duplicate Billings"] = duplicate_billings.to_dict(orient='records')

    # High or Low Billings
    if data_usage_col and plan_type_col and billing_amount_col:
        rates = {'Basic': 0.20, 'Standard': 0.15, 'Ultra': 0.10}
        for index, row in df.iterrows():
            expected_amount = row[data_usage_col] * rates.get(row[plan_type_col].capitalize(), 0)
            if abs(row[billing_amount_col] - expected_amount) > 0.01:
                anomalies["High or Low Billings"].append(row.to_dict())

    # Invalid Service Types
    if service_type_col:
        valid_service_types = ['Data', 'Voice']
        unique_service_types = df[service_type_col].unique()
        invalid_service_types = [st for st in unique_service_types if st.lower() not in [vst.lower() for vst in valid_service_types]]
        invalid_entries = df[df[service_type_col].isin(invalid_service_types)]
        anomalies["Invalid Service Types"] = invalid_entries.to_dict(orient='records')

    # Billing for Suspended Accounts
    if account_status_col and payment_status_col:
        suspended_accounts = df[(df[account_status_col].str.lower() == 'suspended') & (df[payment_status_col].str.lower() == 'pending')]
        anomalies["Billing for Suspended Accounts"] = suspended_accounts.to_dict(orient='records')

    return anomalies

def generate_report(anomalies, output_file):
    """
    Generates a detailed report in JSON format.
    """
    report = {
        "summary": "The Telecom Billing Anomalies Detection system is designed to identify inconsistencies and errors in billing data.",
        "anomalies": anomalies
    }

    with open(output_file, 'w') as file:
        json.dump(report, file, indent=4)

    print(f"Report generated: {output_file}")

def main():
    # File path to the dataset
    dataset_path = input("Enter the path to the CSV file: ")
    
    # Read the dataset
    df = read_dataset(dataset_path)
    if df is None:
        return
    
    # Get columns from the dataset
    columns = df.columns.tolist()
    
    # Prompt GPT-3.5-turbo for anomaly detection criteria
    criteria = prompt_gpt_for_criteria(columns)
    print(f"Anomaly Detection Criteria:\n{criteria}")
    
    # Detect anomalies based on the criteria
    anomalies = detect_anomalies(df, criteria)
    
    # Generate the detailed report
    output_file = "anomalies_report.json"
    generate_report(anomalies, output_file)

if __name__ == "__main__":
    main()
