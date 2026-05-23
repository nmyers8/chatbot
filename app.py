from flask import Flask, request, jsonify, render_template, Response
from openai import OpenAI, RateLimitError
from databricks import sql
import os
from dotenv import load_dotenv
from flask_cors import CORS
import faiss
import numpy as np
from pdf import ingest_pdfs
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

connect = sql.connect(
    server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path = os.getenv("DATABRICKS_HTTP_PATH"),
    access_token = os.getenv("DATABRICKS_TOKEN")
)

schema = {}
# load schema
with connect.cursor() as cursor:
    cursor.execute("SHOW TABLES IN chatbot")
    tables = [row[1] for row in cursor.fetchall()]

    for table in tables:
        cursor.execute(f"DESCRIBE chatbot.{table}") # every table in chatbot available
        columns = [row[0] for row in cursor.fetchall()]
        schema[f"chatbot.{table}"] = columns

def validate_query(query):
    query_lower = query.lower()

    if not query_lower.startswith("select"):
        return False, "Only SELECT queries allowed"

    # chatbot only
    if "chatbot." not in query_lower:
        return False, "Query must use chatbot schema"

    forbidden_words = ["insert", "update", "delete", "drop", "alter", "create"] # no modifying what is in the schema

    for word in forbidden_words:
        if word in query_lower:
            return False, f"{word.upper()} is not allowed"

    return True, None

pdf_index, pdf_chunks = ingest_pdfs()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_pdfs(question):
    query_vector = embed_model.encode([question], convert_to_numpy=True).astype("float32") # turn question into array
    distances, indices = pdf_index.search(query_vector, 5) # find 5 most related vector chunks to question

    results = []
    for dist, i in zip(distances[0], indices[0]):
        # similarity threshold, only include chunks if they meet the threshold
        if dist < 1.5:  # lower = more similar
            chunk = pdf_chunks[i]
            results.append(
                f"Source: {chunk['file']} (Page {chunk['page']})\n{chunk['text']}"
            )
    return "\n\n".join(results)

def generate_sql_query(user_prompt):
    schema_columns = ""
    for table, columns in schema.items():
        schema_columns += f"\nTable: {table}\nColumns:\n" + "\n".join(columns) + "\n"
 
    SYSTEM_PROMPT = f"""
You are a SQL generator.
 
Available tables:
{schema_columns}
 
Rules:
- ONLY generate SELECT statements
- Use exact column names
- Tables are in chatbot schema
- You may JOIN tables if needed
- Return ONLY SQL
- No explanation
- Do NOT use aliases (no AS, no table aliases like t1)
- Always reference columns directly without shorthand
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )
    sql_query = response.choices[0].message.content.strip()
    
    # remove any leading/unwanted text the LLM could have generated
    sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
    if sql_query.lower().startswith("sql"):
        sql_query = sql_query[3:].strip()
 
    is_valid, error_message = validate_query(sql_query)
    if not is_valid:
        return None, error_message
 
    with connect.cursor() as cursor:
        cursor.execute(sql_query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
 
    return [dict(zip(columns, row)) for row in result], None

def decide_output_format(question):
    prompt = """
Decide how the answer should be formatted.

Options:
TEXT: explanation
TABLE: structured table
CHART: visualization (trends, comparisons over time)

Return ONLY one word: TEXT, TABLE, or CHART
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip().upper()

# LLM guardrail
def validate_user_input(prompt):
    blocked = ["drop table", "delete", "hack", "bypass", "ignore previous instructions"]

    for word in blocked:
        if word in prompt.lower():
            return False, "Unsafe or invalid request"

    return True, None

@app.route("/")
def home(): 
    return render_template("index.html") # load chatbot at IP

@app.route("/chat", methods=["POST"]) # receive message from frontend
def chat():
    user_prompt = request.json["message"]
    
    is_valid, error = validate_user_input(user_prompt) # input guardrail
    if not is_valid:
        return jsonify({"error": error})

    try:
        output_format = decide_output_format(user_prompt)

        print(output_format)
        
        data = None
        pdf_context = None
        
        if output_format == "CHART":
            data, error_message = generate_sql_query(user_prompt)
            if error_message:
                return jsonify({"error": error_message})
            
            # generate code for creating a chart
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
You are a data visualization generator.

Return ONLY valid JSON.

Format:
{
  "type": "...",
  "labels": [...],
  "datasets": [
    {
      "label": "...",
      "data": [...]
    }
  ]
}

Rules:
- No explanation
- No markdown
- No HTML
- Ensure valid JSON
"""
                    },
                    {
                        "role": "user",
                        "content": f"""
User question:
{user_prompt}

Dataset results:
{data}
"""
                    }
                ]
            )
            chart_json = response.choices[0].message.content.strip()
            chart_json = chart_json.replace("```json", "").replace("```", "").strip()
            print(chart_json)
            return jsonify({
                "type": "CHART",
                "content": chart_json
            })
        elif output_format == "TABLE":
            data, error_message = generate_sql_query(user_prompt)
            if error_message:
                return jsonify({"error": error_message})
            
            # generate code for creating a table
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
You are a frontend UI generator.
Return ONLY raw HTML/structured data.

Do NOT wrap your response in ``` or markdown code blocks.

Rules:
- table: <table> using provided data
- Do NOT include <html>, <head>, or <body>
- Code must be insertable into a div
"""},
                    {"role": "user", "content": f"""
User question:
{user_prompt}

Dataset results:
{data}
"""}
                ]
            )
            code_output = response.choices[0].message.content.strip()
            print(code_output)
            return jsonify({
                "type": "TABLE", 
                "content": code_output
            }) 
        else: 
            PROMPT = """
You are a knowledgeable health data analyst.

Answer the user's question using only the provided context.
Adapt your response length and format to the question:
- For simple factual questions (a single number or name), answer in one sentence
- For explanatory questions, use markdown formatting:
  - Use **bold** for key terms and important values
  - Use bullet points for lists of effects, reasons, or items
  - Use headers (##) to separate major sections if the response covers multiple topics
  - Keep paragraphs short and scannable
- Never write a wall of unbroken text
- Do not start with "According to the dataset" or "Based on the data"
- If the information is not in the context, say you don't know
"""
            data, _ = generate_sql_query(user_prompt)  # ignore SQL errors, PDF could have answer
            pdf_context = search_pdfs(user_prompt)
 
            if not pdf_context.strip():
                pdf_context = "No relevant research found."
                
            # convert the response from a JSON object to human text
            human_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PROMPT},
                    {
                        "role": "user",
                        "content": f"""
User question:
{user_prompt}
 
Structured data results:
{data}
 
Research context:
{pdf_context}
"""
                    }
                ]
            )
            answer = human_response.choices[0].message.content.strip()
            return jsonify({ # send response back to front end
                "type": "TEXT",
                "content": answer
            })

    except RateLimitError:
        return jsonify({"error": "OpenAI quota exceeded"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run()

# def run_chatbot(user_prompt):    
#     is_valid, error = validate_user_input(user_prompt) # input guardrail
#     if not is_valid:
#         return {"error": error}

#     try:
#         tool = decide_tool(user_prompt)
#         #output_format = decide_output_format(user_prompt)
#         output_format = "TEXT"

#         print(tool)
#         print(output_format)
        
#         data = None
#         pdf_context = None
        
#         # store the column names so they don't all have to be typed out
#         schema_columns = ""
#         for table, columns in schema.items():
#             schema_columns += f"\nTable: {table}\nColumns:\n" + "\n".join(columns) + "\n"
            
#         if tool == "SQL":
#             SYSTEM_PROMPT = f"""
# You are a SQL generator.

# Available tables:
# {schema_columns}

# Rules:
# - ONLY generate SELECT statements
# - Use exact column names
# - Tables are in chatbot schema
# - You may JOIN tables if needed
# - Return ONLY SQL
# - No explanation
# """
#             response = client.chat.completions.create( # LLM decides what to query (what table(s) to choose)
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": SYSTEM_PROMPT},
#                     {"role": "user", "content": user_prompt}
#                 ]
#             )

#             sql_query = response.choices[0].message.content.strip()

#             # remove parts of the message returned that aren't sql queries
#             sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
#             if sql_query.lower().startswith("sql"):
#                 sql_query = sql_query[3:].strip()
#             is_valid, error_message = validate_query(sql_query)
#             if not is_valid:
#                 return {"error": error_message}

#             # run query
#             with connect.cursor() as cursor:
#                 cursor.execute(sql_query)
#                 result = cursor.fetchall()
#                 columns = [desc[0] for desc in cursor.description]

#             data = [dict(zip(columns, row)) for row in result]
#         elif tool == "PDF":
#             pdf_context = search_pdfs(user_prompt)

#             if not pdf_context.strip():
#                 pdf_context = "No relevant research found."
        
#         if output_format == "CHART":
#             # generate code for creating a chart
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": """
# You are a data visualization generator.

# Return ONLY valid JSON.

# Format:
# {
#   "type": "...",
#   "labels": [...],
#   "datasets": [
#     {
#       "label": "...",
#       "data": [...]
#     }
#   ]
# }

# Rules:
# - No explanation
# - No markdown
# - No HTML
# - Ensure valid JSON
# """
#                     },
#                     {
#                         "role": "user",
#                         "content": f"""
# User question:
# {user_prompt}

# Dataset results:
# {data}
# """
#                     }
#                 ]
#             )
#             chart_json = response.choices[0].message.content.strip()
#             chart_json = chart_json.replace("```json", "").replace("```", "").strip()
#             print(chart_json)
#             return {
#                 "type": "CHART",
#                 "content": chart_json
#             }
#         elif output_format == "TABLE":
#             # generate code for creating a table
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": """
# You are a frontend UI generator.
# Return ONLY raw HTML/structured data.

# Do NOT wrap your response in ``` or markdown code blocks.

# Rules:
# - table: <table> using provided data
# - Do NOT include <html>, <head>, or <body>
# - Code must be insertable into a div
# """},
#                     {"role": "user", "content": f"""
# User question:
# {user_prompt}

# Dataset results:
# {data}
# """}
#                 ]
#             )
#             code_output = response.choices[0].message.content.strip()
#             print(code_output)
#             return {
#                 "type": "TABLE", 
#                 "content": code_output
#             } 
#         else: 
#             if tool == "PDF":
#                 context = pdf_context
#                 PROMPT = """
# You are a public health analyst.

# Use ONLY the research context.
# If missing, say you don't know.
# """
#             else:
#                 context = data
#                 PROMPT = """
# You are a data analyst.

# Use ONLY dataset results.
# Be concise and factual.
# """
            
#             # convert the response from a JSON object to human text
#             human_response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=[
#                     {"role": "system", "content": PROMPT},
#                     {
#                         "role": "user",
#                         "content": f"""
# User question:
# {user_prompt}

# Context:
# {context}
# """
#                     }
#                 ]
#             )
#             answer = human_response.choices[0].message.content.strip()
#             return { # send response back to front end
#                 "type": "TEXT",
#                 "content": answer
#             }

#     except RateLimitError:
#         return {"error": "OpenAI quota exceeded"}
#     except Exception as e:
#         return {"error": str(e)}
    
# @app.route("/v1/chat/completions", methods=["POST"])
# def librechat_completion():
#     body = request.json
    
#     messages = body.get("messages", [])

#     user_prompt = ""

#     for msg in reversed(messages): # find the most recent user question
#         if msg.get("role") == "user":
#             content = msg.get("content", "")

#             if isinstance(content, str):
#                 user_prompt = content

#             elif isinstance(content, list): # if message returned in librechat content block format
#                 parts = []
#                 for item in content:
#                     if isinstance(item, dict) and item.get("type") == "text":
#                         parts.append(item.get("text", ""))

#                 user_prompt = " ".join(parts)

#             break # only want one message
        
#     if body.get("stream") is True: # if librechat requested streaming
#         result = run_chatbot(user_prompt)

#         if isinstance(result, dict):
#             content = result.get("content") or result.get("error") or ""
#         else:
#             content = str(result)

#         chunk = { # build streaming chunk
#             "id": f"chatcmpl-{uuid.uuid4().hex}",
#             "object": "chat.completion.chunk",
#             "created": int(time.time()),
#             "model": "custom-flask",
#             "choices": [
#                 {
#                     "index": 0,
#                     "delta": {
#                         "role": "assistant",
#                         "content": content
#                     },
#                     "finish_reason": "stop"
#                 }
#             ]
#         }

#         def generate():
#             yield f"data: {json.dumps(chunk)}\n\n"
#             yield "data: [DONE]\n\n"

#         return Response(generate(), mimetype="text/event-stream")

#     result = run_chatbot(user_prompt)

#     # extract content from chatbot's response
#     if isinstance(result, dict):
#         content = result.get("content") or result.get("error") or ""
#     else:
#         content = str(result)

#     # build open-AI compatible response
#     return jsonify({ 
#         "id": f"chatcmpl-{uuid.uuid4().hex}",
#         "object": "chat.completion",
#         "created": int(__import__("time").time()),
#         "model": "custom-flask",
#         "choices": [
#             {
#                 "index": 0,
#                 "message": {
#                     "role": "assistant",
#                     "content": content
#                 },
#                 "finish_reason": "stop"
#             }
#         ],
#         "usage": {
#             "prompt_tokens": 0,
#             "completion_tokens": 0,
#             "total_tokens": 0
#         }
#     })
    
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000)