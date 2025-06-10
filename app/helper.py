import os
import re
import sqlite3
import chardet

import cloudinary
import pandas as pd

from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import cloudinary.uploader
import tempfile
import uuid

# from langchain.vectorstores import FAISS
# from langchain.embeddings import HuggingFaceEmbeddings

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.utilities import SQLDatabase
from langchain_ollama import OllamaLLM
from fuzzywuzzy import process
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import OUTPUT_FOLDER


cloudinary.config(
  cloud_name="dlb6c8ftf",
  api_key="636263816788567",
  api_secret="cMOW6mrEiAz_tqbXb494lAZpZl0"
)


# Initialize the LLM and embeddings
llm = OllamaLLM(model="llama3:8b-instruct-q4_0", base_url="http://localhost:11434")

# llm= Ollama(model='mistral:7b-instruct-v0.2-q8_0')
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

def rephrase_query(input_query, column_mapping):
    # Iterate over user terms and corresponding database columns
    for user_term, db_column in column_mapping.items():
        input_query = re.sub(rf'\b{re.escape(user_term)}\b', db_column, input_query, flags=re.IGNORECASE)
        print(f"Replacing '{user_term}' with '{db_column}' -> {input_query}")
    print("Final rephrased_query:", input_query)
    return input_query

def get_column_mapping(db,file_name):
    print("here is the db",db)
    table_names = db.get_usable_table_names()
    print("table namesss",table_names)
    schema_info = db.get_table_info(table_names=[file_name])
    
    # Initialize an empty dictionary to hold the mapping
    column_mapping = {}

    # Check if schema_info contains parentheses, indicating column definitions
    if "(" in schema_info and ")" in schema_info:
        # Extract only the part inside parentheses, which should list column definitions
        columns_section = schema_info.split("(", 1)[1].split(")", 1)[0]
        
        # Split this section by commas to get individual column definitions
        columns = columns_section.split(", ")
        
        # Iterate over each column definition
        for column_def in columns:
            # Each definition is in the format `"Column Name" DATA_TYPE`
            if '"' in column_def:
                column_name = column_def.split('"')[1]  # Get the column name without quotes
                # Map lowercase column names to original case for case-insensitive replacement
                column_mapping[column_name.lower()] = column_name
    
    # Debugging: Print the column mapping
    print("column_mapping", column_mapping)
    return column_mapping


def create_db_from_csv(df,file_name):
    sqlite_db_path = 'my_database.db'  # Desired SQLite database name
    conn = sqlite3.connect(sqlite_db_path)

    # Write the DataFrame to a new SQLite table
    # If the table already exists, you can choose 'replace', 'append', or 'fail'
    df.to_sql(file_name, conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    return True

def extract_prompt_chart_details(response):
    pattern = r"\*\*X-axis column:\*\*\s*(.*?)\n\*\*Y-axis column:\*\*\s*(.*?)\n\*\*Suggested chart type:\*\*\s*(.*?)\n\*\*Reasoning:\*\*\s*(.*?)\n"
    return re.findall(pattern, response)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df.dropna(how='all', inplace=True)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            elif df[col].dtype == object:
                mode = df[col].mode().iloc[0] if not df[col].mode().empty else "missing"
                df[col] = df[col].fillna(mode)
            else:
                df[col] = df[col].fillna("unknown")
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df.drop_duplicates(inplace=True)
    cols_to_drop = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=cols_to_drop, inplace=True)
    return df

def create_vector_db(file_path, chunk_size=1000, chunk_overlap=100):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        detected_encoding = result['encoding']

    df = pd.read_csv(file_path, encoding=detected_encoding)
    print(df.shape)
    df_clean = clean_dataframe(df)
    processed_file_path = os.path.join(OUTPUT_FOLDER, f'processed_{os.path.basename(file_path)}')
    # processed_file_path = f'{OUTPUT_FOLDER}_processed_{os.path.basename(file_path)}'
    df_clean.to_csv(processed_file_path, index=False, encoding='utf-8')
    text_data = df_clean.to_string(index=False)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.create_documents([text_data])

    # Create a FAISS vector database
    vectordb = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectordb_file_path = "vector_db"
    vectordb.save_local(vectordb_file_path)
    return vectordb_file_path


def get_qa_chain(vectordb_file_path):
    print("inside generate insight finction")
    vectordb = FAISS.load_local(vectordb_file_path, embeddings=embeddings, allow_dangerous_deserialization=True)
    retriever = vectordb.as_retriever(score_threshold=0.7)
    prompt_template = """
        Given the following context and a question, generate an answer based on this context only.

        CONTEXT: {context}

        QUESTION: {question}
        """
    print("vector db is loaded")
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PROMPT},
        verbose=False
    )
    return chain

def execute_query(conn: sqlite3.Connection, query: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(query, conn)
    except Exception as e:
        print("Error executing query:", e)
        return pd.DataFrame()

def extract_sql_query(agent_log: dict) -> str:
    
    if isinstance(agent_log, dict) and 'output' in agent_log:
        output_text = agent_log['output']
        matches = re.findall(r"```sql\n(.*?)```", output_text, re.DOTALL)
        return matches[-1].strip() if matches else None
    return None

def generate_summary(chain):
    print("inside generate insight function")
    summary_question = "Provide a brief summary of the complete dataset, including its purpose, structure."

    summary = chain.invoke({"query": summary_question})["result"]
    print(summary, "summary")

    return summary

def generate_insights(chain):
    print("inside generate insight function")

    insights_question = "List the column names and  observations from the complete  dataset."

    insights = chain.invoke({"query": insights_question})["result"]
    print(insights, "insights")

    return insights

def generate_prediction(chain):
    print("inside generate predictive function")

    predictive_questions = "Generate 5 predictive questions based on the complete dataset for analyzing outcomes, trends, or patterns."

    predictions = chain.invoke({"query": predictive_questions})["result"]
    print(predictions, "predictions")

    return predictions


def suggest_visualization(df: pd.DataFrame):
    llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
    prompt = f"""
    Given the following dataset columns: {list(df.columns)}, suggest relevant data visualizations.
    Identify relationships between columns and recommend suitable chart types (e.g., bar, pie, scatter, histogram) with reasons.
    """
    response = llm.invoke(prompt)
    return response

def find_best_match(target_col, dataset_columns, threshold=80):
    """
    Find the best match for a target column name within dataset columns using fuzzy matching.
    
    Parameters:
    - target_col: The column name extracted from AI response.
    - dataset_columns: List of actual dataset column names.
    - threshold: Minimum similarity score (0-100) to accept a match.
    
    Returns:
    - The best matching column name from dataset or None if no good match found.
    """
    match, score = process.extractOne(target_col, dataset_columns)
    print("MATCH---------",match)
    print(score)
    return match if score >= threshold else None

def plot_chart(df, x_col, y_col, chart_type):
    print("step - plot charts")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")  # Modern UI theme

    # Improved color palettes
    colors = sns.color_palette("pastel")

    try:
        if df[x_col].nunique() > 20:
            df = df.groupby(x_col, as_index=False)[y_col].sum().nlargest(20, y_col)
        if chart_type.lower() == "bar chart":
            sns.barplot(x=df[x_col], y=df[y_col], ax=ax, palette="coolwarm")
            ax.set_title(f"📊 Bar Chart: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "scatter plot":
            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax, color="dodgerblue", s=80, edgecolor="black", alpha=0.75)
            ax.set_title(f"🔵 Scatter Plot: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "line chart":
            sns.lineplot(x=df[x_col], y=df[y_col], ax=ax, marker="o", color="green", linewidth=2)
            ax.set_title(f"📈 Line Chart: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "stacked bar chart":
            df_grouped = df.groupby([x_col, y_col]).size().unstack()
            df_grouped.plot(kind="bar", stacked=True, figsize=(10, 6), ax=ax, colormap="viridis")
            ax.set_title(f"📊 Stacked Bar Chart: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "histogram":
            sns.histplot(df[x_col], bins=20, kde=True, ax=ax, color="coral")
            ax.set_title(f"📊 Histogram: {x_col} Distribution", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "box plot":
            sns.boxplot(x=df[x_col], y=df[y_col], ax=ax, palette="Set2")
            ax.set_title(f"📊 Box Plot: {x_col} vs {y_col}", fontsize=14, fontweight="bold")

        elif chart_type.lower() == "pie chart":
            plt.figure(figsize=(7, 7))
            df[x_col].value_counts().plot.pie(autopct="%1.1f%%", colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
            plt.title(f"🥧 Pie Chart: {x_col} Distribution", fontsize=14, fontweight="bold")
            return plt

        elif chart_type.lower() == "heatmap":
            correlation_matrix = df.corr(numeric_only=True)
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
            ax.set_title("🔥 Correlation Heatmap", fontsize=14, fontweight="bold")

        else:
            return None

        # Common UI improvements for all charts
        ax.set_xlabel(x_col.capitalize(), fontsize=12, fontweight="bold")
        ax.set_ylabel(y_col.capitalize(), fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        return fig

    except Exception as e:
        return None

def generate_ai_charts(df, ai_response):
    print("step - generate charts")
    chart_details = extract_prompt_chart_details(ai_response)
    print(chart_details, "chart_details")
    uploaded_images = []

    for details in chart_details:
        if len(details) == 3:
            x_col, y_col, chart_type = details
            reasoning = "No reasoning provided."
        else:
            x_col, y_col, chart_type, reasoning = details

        matched_x_col = find_best_match(x_col, df.columns)
        matched_y_col = find_best_match(y_col, df.columns)

        if matched_x_col and matched_y_col:
            fig = plot_chart(df, matched_x_col, matched_y_col, chart_type)
            if fig:
                # Save chart to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.savefig(tmpfile.name)
                    tmpfile.flush()

                    # Upload to Cloudinary
                    result = cloudinary.uploader.upload(tmpfile.name, public_id=f"ai_charts/{uuid.uuid4()}", overwrite=True)
                    image_url = result.get("secure_url")

                    uploaded_images.append({
                        "x_col": matched_x_col,
                        "y_col": matched_y_col,
                        "chart_type": chart_type,
                        "reasoning": reasoning,
                        "image_url": image_url
                    })

                # Close the figure to free memory
                plt.close(fig)
        else:
            uploaded_images.append({
                "chart_type": chart_type,
                "reasoning": f"⚠️ No suitable match found for '{x_col}' or '{y_col}' in dataset.",
                "image_url": None
            })

    return uploaded_images

def suggest_visualization_with_ai(df: pd.DataFrame):
    print("step - suggest_visualization_with_ai ")
    llm = OllamaLLM(model="llama3", base_url="http://localhost:11434")
    prompt_template = PromptTemplate(
        template="""
        You are a data visualization expert. Given the dataset columns: {columns}, 
        suggest suitable visualizations in the following format:

        - **X-axis column:** <column_name>
        - **Y-axis column:** <column_name>
        - **Suggested chart type:** <chart_name>
        - **Reasoning:** <explanation>

        Provide at least 3 different visualization suggestions.
        """,
        input_variables=["columns"]
    )

    prompt = prompt_template.format(columns=list(df.columns))
    response = llm.invoke(prompt)

    chart_urls = generate_ai_charts(df, ai_response=response)
    print("charts generated")
    return {
        "llm_response": response,
        "charts": chart_urls
    }


def generate_human_response(llm2, question: str, result_df: pd.DataFrame, sql_query: Optional[str] = None) -> str:
    # Convert result dataframe to a readable form
    if result_df.empty:
        result_str = "The result is empty."
    else:
        result_str = result_df.to_string(index=False)

    # Construct the prompt for LLM
    prompt = f"""
    🤖 You are a friendly and intelligent assistant who helps people understand data in simple, everyday language.

    Here’s what you need to know:

    📌 **User Question:**  
    - {question}

    🧾 **SQL Query (for reference):**  
    - {sql_query or '[Not provided]'}

    📊 **SQL Result:**  
    - {result_str}

    🎯 **Your Task:**  
    - Interpret the result above.
    - Explain it clearly and conversationally, like you're helping a non-technical person 

    ✨ Please write your answer now:
    """


    # Call your LLM (replace this with actual LLaMA/OpenAI API)
    response = llm.invoke(prompt)
    return response.strip()


def generate_plotly_code(llm, question: str, result_df: pd.DataFrame, sql_query: Optional[str] = None) -> str:
    # Convert data to dict for JS-friendly view
    data_dict = result_df.to_dict(orient="list")
    
    prompt = f"""
📊 You are a data visualization expert who helps users create clean, meaningful charts using Python and Plotly.


Given:
- 🤔 **User Question:** {question}
- 💾 **SQL Query:** {sql_query or '[Not provided]'}
- 📈 **Data Output:** {data_dict}

🎯 Your tasks:
1. Identify the **most appropriate chart type** (e.g., bar chart, pie chart, line chart, etc.) for visualizing this data.
2. Write the **Plotly.py code** (Python) to generate the chart.
3. Use clean variable names and ensure it's ready to run in a browser.
4. Do  explain the chart 

📝 Note:
- — just output working Plotly.py code.


Ready? Let’s go!
"""
    
    response = llm.invoke(prompt)
    return response.strip()

import re
from typing import Tuple

def extract_code_and_explanation(response_text: str) -> Tuple[str, str]:
    """
    Extract the Python Plotly code and explanation from the LLM response.
    Returns: (code_str, explanation_str)
    """
    # Extract code block between triple backticks
    code_match = re.search(r"```(?:python)?\s*(.*?)```", response_text, re.DOTALL)
    code = code_match.group(1).strip() if code_match else ""
    print("code ",code)

    # Remove the code block from original response to get explanation
    explanation = re.sub(r"```(?:python)?\s*.*?```", "", response_text, flags=re.DOTALL).strip()
    print("explanation",explanation)

    return code, explanation
