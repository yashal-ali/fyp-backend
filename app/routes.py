import os

import re
import sqlite3
import pandas as pd

import pandas as pd
from datetime import datetime, timedelta
from flask import request, jsonify, send_file
from werkzeug.utils import secure_filename
from collections import Counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from .config import OUTPUT_FOLDER
from .helper import create_vector_db, extract_code_and_explanation, get_qa_chain, generate_insights, suggest_visualization, suggest_visualization_with_ai, generate_summary, generate_prediction, create_db_from_csv, rephrase_query, get_column_mapping, extract_sql_query, execute_query, generate_human_response, generate_plotly_code

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from lida import Manager
from lida.datamodel import TextGenerationConfig
from .llama_ollama import LLaMAOllamaLLM
from langchain_community.llms import Ollama
from .ollama_text_generator import OllamaTextGenerator
from .goal_explorer import GoalExplorer
from PIL import Image
import base64
from io import BytesIO
import plotly.io as pio
text_gen = OllamaTextGenerator(model="llama3", temperature=0.7)
textgen_config = TextGenerationConfig(temperature=0.7, max_tokens=1024)
lida = Manager(text_gen=LLaMAOllamaLLM(model="llama3:8b-instruct-q4_0"))
goal_explorer = GoalExplorer()

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


llm = OllamaLLM(model="llama3:8b-instruct-q4_0", base_url="http://localhost:11434")
llm_= OllamaLLM(model='mistral:7b-instruct-v0.2-q8_0' ,base_url="http://localhost:11434")

def init_routes(app):

    @app.route("/api/file_upload", methods=["POST"])
    def file_upload(): 
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        filename = file.filename
        

        if filename == "":
            return jsonify({"error": "No selected file"}), 400

        if file:
            output_file_folder = OUTPUT_FOLDER

            if not os.path.exists(output_file_folder):
                os.makedirs(output_file_folder)

            file_path = os.path.join(output_file_folder, filename)

            # Check if the file already exists
            if os.path.exists(file_path):
                pass
            else:
                file.save(file_path)

            # Proceed with processing
            df = pd.read_csv(file_path)
            chunk_size = 2000  # Tune as needed
            chunk_overlap = 200  # Tune as needed

            vectordb_path = create_vector_db(file_path, chunk_size, chunk_overlap)
            if vectordb_path:
                return jsonify({
                    "status": True,
                    "file_name": file_path,
                    "remarks": "Vector DB created",
                    "vector_db_path": vectordb_path
                }), 200
        return jsonify({"status": False, "remarks": "File not found"}), 404
    

    @app.route("/api/generate_summary", methods=["POST"])
    def generate_summaries():  

        vectordb_file_path = request.json.get("vectordb_file_path", "")
        if vectordb_file_path:
            chain = get_qa_chain(vectordb_file_path=vectordb_file_path)
            print("chain is generated")
            summary =  generate_summary(chain)
            return jsonify({
                    "summary": summary
                }), 200

        return jsonify({"status": False, "remarks": "Error generating summary"}), 404
    

    @app.route("/api/generate_insight", methods=["POST"])
    def generate_insight():  

        vectordb_file_path = request.json.get("vectordb_file_path", "")
        if vectordb_file_path:
            chain = get_qa_chain(vectordb_file_path=vectordb_file_path)
            print("chain is generated")
            insights =  generate_insights(chain)
            return jsonify({
                    "insights": insights
                }), 200

        return jsonify({"status": False, "remarks": "Error generating insights"}), 404

    @app.route("/api/generate_prediction", methods=["POST"])
    def generate_prediction():  

        vectordb_file_path = request.json.get("vectordb_file_path", "")
        if vectordb_file_path:
            chain = get_qa_chain(vectordb_file_path=vectordb_file_path)
            print("chain is generated")
            predictions =  generate_prediction(chain)
            return jsonify({
                    "predictions": predictions
                }), 200

        return jsonify({"status": False, "remarks": "Error generating prediction"}), 404

    @app.route("/api/suggest_visual", methods=["POST"])
    def suggest_visual():  
        file_name = request.json.get("file_name", "")
        if file_name:
            file_path = os.path.join(OUTPUT_FOLDER, file_name)

            df = pd.read_csv(file_path)
            response = suggest_visualization(df)
            return jsonify({
                "status": True,
                "response": response
            }), 200
        else:
            return jsonify({"status": False, "remarks": "Error suggesting visualization"}), 404 
        

    @app.route("/api/suggest_visual_with_ai", methods=["POST"])
    def suggest_visual_with_ai():  
        file_name = request.json.get("file_name", "")
        if file_name:
            file_path = os.path.join(OUTPUT_FOLDER, file_name)
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip().str.lower()

            response_data = suggest_visualization_with_ai(df)

            return jsonify({
                "status": True,
                "data": response_data
            }), 200
        else:
            return jsonify({"status": False, "remarks": "Error suggesting visualization"}), 404
        
    
    @app.route("/api/chat_with_data", methods=["POST"])
    def chat_with_data():  
        question = request.json.get("question", "")
        vectordb_file_path = request.json.get("vectordb_file_path", "")
        if vectordb_file_path:
            chain = get_qa_chain(vectordb_file_path=vectordb_file_path)
            print("chain is generated")
            print("Processing your question...\n")
            result = chain.invoke({"query": question})["result"]
            return jsonify({
                "status": True,
                "result": result
            }), 200 
        else:
            return jsonify({"status": False, "remarks": "Error generating response"}), 404


    @app.route("/api/summarizing", methods=["POST"])
    def summarizing():
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "File is required"}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        file.save(filepath)

        summary = lida.summarize(filepath, summary_method="default", textgen_config=textgen_config)
        goals = goal_explorer.generate(summary, textgen_config=textgen_config, text_gen=text_gen, n=2)
    
        return jsonify({
            "summary": summary,
            "goals": goals,
            
        }), 200

    @app.route("/api/chat_with_real_data", methods=["POST"])
    def chat_with_real_data():  

        file_name = request.json.get("file_name", "")
        user_query = request.json.get("user_query", "")

        if file_name:
            file_path = os.path.join(OUTPUT_FOLDER, file_name)

            table_name = file_name.rsplit('.', 1)[0]  # 'sales_data'
            df = pd.read_csv(file_path)
            create_db_from_csv(df,file_name)

            db = SQLDatabase.from_uri("sqlite:///my_database.db")

            # Initialize the agent with parsing error handling
            agent_executor = create_sql_agent(
                llm,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
            )

            # Get the column mapping
            column_mapping = get_column_mapping(db,table_name)
            print("column_mapping:", column_mapping)

            # Rephrase the query based on the column mapping
            rephrased_query = rephrase_query(user_query, column_mapping)

            rephrased_query_with_context = (
            f"You are a SQL expert tasked with executing the following query accurately and concisely.\n"
            f"Database Table: {table_name}.\n"
            f"Columns: {', '.join(column_mapping.values())}.\n\n"
            f"User Query: {rephrased_query}.\n\n"
            f"Instructions:\n"
            f"1. Construct an SQL query only using columns from the {table_name} table.\n"
            f"2. Exclude rows with null values in calculations unless specified otherwise.\n"
            f"3. Use precise quoting for column names as needed, and avoid unnecessary commentary.\n\n"
            f"4. Avoid ambiguous column references to prevent errors like SQLAlchemy’s `E3Q8`. Ensure all column names clearly refer to the {table_name} table.\n"
            f"Return only the final result of the query with additional explanations or commentary."
        )


            response = agent_executor.invoke({"input": rephrased_query_with_context})
            print("Response:", response)
            return jsonify({
                    "status": True,
                    "result": response
                }), 200 
        else:
            return jsonify({"status": False, "remarks": "Error generating response"}), 404
        
    
    # user question input - 
    @app.route("/api/chat_with_human_data", methods=["POST"])
    def chat_with_data_chat():  
        try:
            file_name = request.json.get("file_name", "")
            table_name = file_name.rsplit('.', 1)[0]  # 'sales_data'

            user_query = request.json.get("question", "")
            db = SQLDatabase.from_uri("sqlite:///my_database.db")
            
            column_mapping = get_column_mapping(db,table_name)
            print("column_mapping:", column_mapping)

            # Rephrase the query based on the column mapping
            rephrased_query = rephrase_query(user_query, column_mapping)

            # Construct a general prompt with specific table and column context
            rephrased_query_with_context = (
            f"You are a SQL expert tasked with executing the following query accurately and concisely.\n"
            f"Database Table: {table_name}.\n"
            f"Columns: {', '.join(column_mapping.values())}.\n\n"
            f"User Query: {rephrased_query}.\n\n"
            f"Instructions:\n"
            f"1. Construct an SQL query only using columns from the {table_name} table.\n"
            f"2. Exclude rows with null values in calculations unless specified otherwise.\n"
            f"3. Use precise quoting for column names as needed, and avoid unnecessary commentary.\n\n"
            f"4. Avoid ambiguous column references to prevent errors like SQLAlchemy’s `E3Q8`. Ensure all column names clearly refer to the {table_name} table.\n"
            f"Return only the final result of the query with additional explanations or commentary."
        )
            # Initialize the agent with parsing error handling
            agent_executor = create_sql_agent(
                llm_,
                db=db,
                agent_type="zero-shot-react-description",
                verbose=True,
                handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
            )

            try:
                response = agent_executor.invoke({"input": rephrased_query_with_context})
                print("Agent response:", response)
            except Exception as e:
                print(f"SQL Agent Error: {e}")
                return jsonify({"error": f"SQL Agent failed: {e}"}), 500

            # extract sql 
            query = extract_sql_query(response)
            #query execute kaaha hogi? conn?
            sqlite_db_path = 'my_database.db'  # Desired SQLite database name
            conn = sqlite3.connect(sqlite_db_path)
            query_result_df = execute_query(conn,query)
            response_2 = generate_human_response(llm, user_query , query_result_df, sql_query=query)
            return jsonify({
                    "status": True,
                    "result": response_2
                }), 200 
          
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({"error": str(e)}), 500
    

    #  only extract sql 
    @app.route("/api/extract_sql", methods=["POST"])
    def extract_sql():  
        file_name = request.json.get("file_name", "")
        table_name = file_name.rsplit('.', 1)[0]  # 'sales_data'

        user_query = request.json.get("question", "")
        db = SQLDatabase.from_uri("sqlite:///my_database.db")
            
        column_mapping = get_column_mapping(db,table_name)
        
        print("column_mapping:", column_mapping)

        # Rephrase the query based on the column mapping
        rephrased_query = rephrase_query(user_query, column_mapping)

        # Construct a general prompt with specific table and column context
        rephrased_query_with_context = (
            f"You are a SQL expert tasked with executing the following query accurately and concisely.\n"
            f"Database Table: {table_name}.\n"
            f"Columns: {', '.join(column_mapping.values())}.\n\n"
            f"User Query: {rephrased_query}.\n\n"
            f"Instructions:\n"
            f"1. Construct an SQL query only using columns from the {table_name} table.\n"
            f"2. Exclude rows with null values in calculations unless specified otherwise.\n"
            f"3. Use precise quoting for column names as needed, and avoid unnecessary commentary.\n\n"
            f"4. Avoid ambiguous column references to prevent errors like SQLAlchemy’s `E3Q8`. Ensure all column names clearly refer to the {table_name} table.\n"
            f"Return only the final result of the query with additional explanations or commentary."
        )

        # Initialize the agent with parsing error handling
        agent_executor = create_sql_agent(
            llm_,
            db=db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
        )

        response = agent_executor.invoke({"input": rephrased_query_with_context})
        # extract sql 
        query = extract_sql_query(response)
        return jsonify({
                    "status": True,
                    "result": query
                }), 200 
    

    # user question input - 
    @app.route("/api/generate_plotly_chart", methods=["POST"])
    def generate_plotly_chart():  
        
        file_name = request.json.get("file_name", "")
        file_path = os.path.join(OUTPUT_FOLDER, file_name)

        table_name = file_name.rsplit('.', 1)[0]  # 'sales_data'
        df = pd.read_csv(file_path)
        create_db_from_csv(df,table_name)

        user_query = request.json.get("question", "")
        db = SQLDatabase.from_uri("sqlite:///my_database.db")
            
        column_mapping = get_column_mapping(db,table_name)
       
        print("column_mapping:", column_mapping)

        # Rephrase the query based on the column mapping
        rephrased_query = rephrase_query(user_query, column_mapping)

        # Construct a general prompt with specific table and column context
        rephrased_query_with_context = (
            f"You are a SQL expert tasked with executing the following query accurately and concisely.\n"
            f"Database Table: {table_name}.\n"
            f"Columns: {', '.join(column_mapping.values())}.\n\n"
            f"User Query: {rephrased_query}.\n\n"
            f"Instructions:\n"
            f"1. Construct an SQL query only using columns from the {table_name} table.\n"
            f"2. Exclude rows with null values in calculations unless specified otherwise.\n"
            f"3. Use precise quoting for column names as needed, and avoid unnecessary commentary.\n\n"
            f"4. Avoid ambiguous column references to prevent errors like SQLAlchemy’s `E3Q8`. Ensure all column names clearly refer to the {table_name} table.\n"
            f"Return only the final result of the query with additional explanations or commentary."
        )

        # Initialize the agent with parsing error handling
        agent_executor = create_sql_agent(
            llm_,
            db=db,
            agent_type="zero-shot-react-description",
            verbose=True,
            handle_parsing_errors="Check your output and make sure it conforms, use the Action/Action Input syntax",
        )

        response = agent_executor.invoke({"input": rephrased_query_with_context})
        # extract sql 
        query = extract_sql_query(response)
        #query execute kaaha hogi? conn?
        sqlite_db_path = 'my_database.db'  # Desired SQLite database name
        conn = sqlite3.connect(sqlite_db_path)
        query_result_df = execute_query(conn,query)
        
        # response_2 = generate_human_response(llm, user_query , query_result_df, sql_query=query)
        response_2 = generate_plotly_code(llm, user_query , query_result_df, query)
        code_str, explanation_str = extract_code_and_explanation( response_2)
        local_env = {"df": query_result_df}
        # try:
        #     exec(code_str, {}, local_env)
        #     fig = local_env.get("fig")
        #     fig_json = pio.to_json(fig)
        # except Exception as e:
        #     return jsonify({"status": False, "error": str(e)}), 500
        return jsonify({
            "status": True,
            "explanation": explanation_str,
            "chart": code_str,
        }), 200
  


    




        