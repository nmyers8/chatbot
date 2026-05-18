import camelot
import pandas as pd
from databricks import sql
import os
from dotenv import load_dotenv
import re

load_dotenv()

DATABRICKS_SERVER_HOSTNAME = os.getenv("DATABRICKS_SERVER_HOSTNAME")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

print("Reading PDF tables...")

tables = camelot.read_pdf(
    "test/table.pdf",
    pages="1",
    flavor="lattice"
)

print("Tables found:", tables.n)
for i, table in enumerate(tables):
    print(f"\nTable {i}")
    print(table.df)

dfs = []

for table in tables:
    df = table.df

    if df.empty:
        continue
    
    header_rows = []
    data_start = 0

    for i, row in df.iterrows():
        row_text = " ".join(row.astype(str))
        
        if re.search(r"\d", row_text):
            data_start = i
            break
        else:
            header_rows.append(row)

    # combine header rows
    header_df = pd.DataFrame(header_rows).fillna("")
    columns = header_df.apply(lambda col: " ".join(col).strip(), axis=0)

    df.columns = columns
    df = df.iloc[data_start:].reset_index(drop=True)
    
    df.columns = (
        df.columns
        .str.replace("\n", " ")
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("[^a-zA-Z0-9_]", "", regex=True)
    )

    # Remove completely empty rows
    df = df.dropna(how="all")

    dfs.append(df)

if not dfs:
    raise ValueError("No tables found in PDF")

table_df = pd.concat(dfs, ignore_index=True)

# Clean column names
table_df.columns = (
    table_df.columns
    .astype(str)
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("[^a-zA-Z0-9_]", "", regex=True)
)

# -------- Send to Databricks --------

with sql.connect(
    server_hostname=DATABRICKS_SERVER_HOSTNAME,
    http_path=DATABRICKS_HTTP_PATH,
    access_token=DATABRICKS_TOKEN
) as conn:

    with conn.cursor() as cursor:

        # Create SQL table
        columns_sql = ", ".join([f"{c} STRING" for c in table_df.columns])

        cursor.execute(f"""
        CREATE OR REPLACE TABLE hazards.testing (
            {columns_sql}
        )
        """)

        cols = ", ".join(table_df.columns)
        placeholders = ", ".join(["?"] * len(table_df.columns))

        insert_sql = f"""
        INSERT INTO hazards.testing ({cols})
        VALUES ({placeholders})
        """

        for _, row in table_df.iterrows():
            cursor.execute(insert_sql, tuple(row.astype(str)))

print("PDF extraction complete.")