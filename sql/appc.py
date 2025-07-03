import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import polars as pl
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from region_normalizer import normalize_region_name, normalize_region_list
from utils_helpers_model_context import rename_cols
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="Sales Data Assistant", layout="wide")

# Constants
DB_PATH = "sales.db"
PARQUET_PATH = "sales_rt.parquet"


# Initialize Groq client
# @st.cache_resource
# def get_llm():
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         st.error("GROQ_API_KEY not found in environment variables")
#         st.stop()
#     return ChatGroq(
#         api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0
#     )


@st.cache_resource
def get_llm():
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([api_key, azure_endpoint, api_version, azure_deployment]):
        st.error(
            "Azure OpenAI environment variables not found. Please set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_DEPLOYMENT_NAME."
        )
        st.stop()

    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version=api_version,
        azure_deployment=azure_deployment,
        temperature=0.2,
    )


# Load data and create/update SQLite database
@st.cache_data
def load_data_to_db():
    # Check if parquet file exists
    if not os.path.exists(PARQUET_PATH):
        st.error(
            f"File {PARQUET_PATH} not found. Please place it in the same directory as this script."
        )
        st.stop()

    # Load parquet file
    df_pandas = pd.read_parquet(PARQUET_PATH)

    # Convert pandas to polars for column renaming
    df_polars = pl.from_pandas(df_pandas)

    # Rename columns using helper function
    df_polars = rename_cols(df_polars)

    # Normalize region/state values
    if "customer_state" in df_polars.columns:
        df_polars = df_polars.with_columns(
            pl.col("customer_state")
            .map_elements(normalize_region_name)
            .alias("customer_state")
        )

    # Clean whitespace from brand column
    if "brand" in df_polars.columns:
        df_polars = df_polars.with_columns(
            pl.col("brand").str.strip_chars().alias("brand")
        )

    # Clean whitespace and normalize branch column
    if "branch" in df_polars.columns:
        df_polars = df_polars.with_columns(
            pl.col("branch").str.strip_chars().alias("branch")
        )
        df_polars = df_polars.with_columns(
            pl.col("branch")
            .map_elements(normalize_region_name, return_dtype=pl.String)
            .alias("branch")
        )

    # Convert back to pandas
    df = df_polars.to_pandas()

    # Create SQLite connection and save data
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("sales", conn, if_exists="replace", index=False)

    # Get schema information for future use
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(sales)")
    columns = cursor.fetchall()

    # Get column names and types
    schema_info = []
    for col in columns:
        schema_info.append({"name": col[1], "type": col[2]})

    # Get basic table stats
    cursor.execute("SELECT COUNT(*) FROM sales")
    row_count = cursor.fetchone()[0]

    # Get all brand names for better matching
    try:
        if "brand" in df.columns:
            all_brands = df["brand"].dropna().unique().tolist()
        else:
            all_brands = []
    except Exception as e:
        st.warning(f"Could not extract brand list: {str(e)}")
        all_brands = []

    conn.close()

    return {
        "df_sample": df.head(5),
        "schema": schema_info,
        "row_count": row_count,
        "columns": df.columns.tolist(),
        "all_brands": all_brands,  # Add the complete list of brands
    }


# Execute SQL query with better error handling
def execute_query(query):
    conn = sqlite3.connect(DB_PATH)
    try:
        # Sanitize query to help prevent SQL injection
        query = query.strip()
        df_result = pd.read_sql_query(query, conn)
        conn.close()
        return df_result
    except Exception as e:
        conn.close()
        st.error(f"SQL Error: {str(e)}")
        st.code(query, language="sql")
        raise e


# Add a function to find the closest brand match
def find_closest_brand(search_term, all_brands):
    """Find the closest matching brand name from the database"""
    if not search_term or not all_brands:
        return None

    # Convert search term to lowercase for case-insensitive matching
    search_lower = search_term.lower()

    # First try direct substring match (case insensitive)
    direct_matches = [brand for brand in all_brands if search_lower in brand.lower()]
    if direct_matches:
        # Sort by length to prefer shorter, more specific matches
        return sorted(direct_matches, key=len)[0]

    # If no direct match, look for partial matches
    partial_matches = []
    search_words = search_lower.split()
    for brand in all_brands:
        brand_lower = brand.lower()
        # Check if any word in the search term is in the brand name
        for word in search_words:
            if len(word) > 2 and word in brand_lower:
                partial_matches.append(brand)
                break

    if partial_matches:
        return partial_matches[0]

    # No matches found
    return None


# Enhanced SQL generation with better brand name handling
def generate_sql(question, db_info):
    llm = get_llm()

    # Format schema information with more details
    schema_str = "\n".join(
        [f"- {col['name']} ({col['type']})" for col in db_info["schema"]]
    )

    # Get sample values for key columns to help with query generation
    sample_values = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Find likely brand and date columns
        brand_columns = [
            col["name"]
            for col in db_info["schema"]
            if any(x in col["name"].lower() for x in ["brand", "product", "item"])
        ]

        date_columns = [
            col["name"]
            for col in db_info["schema"]
            if any(
                x in col["name"].lower()
                for x in ["date", "time", "day", "month", "year"]
            )
        ]

        # Get sample values for potential brand column
        if brand_columns:
            cursor.execute(f"SELECT DISTINCT {brand_columns[0]} FROM sales LIMIT 10")
            sample_values["brands"] = [row[0] for row in cursor.fetchall()]

        # Get date range for potential date column
        if date_columns:
            cursor.execute(
                f"SELECT MIN({date_columns[0]}), MAX({date_columns[0]}) FROM sales"
            )
            min_date, max_date = cursor.fetchone()
            sample_values["date_range"] = f"{min_date} to {max_date}"

        conn.close()
    except Exception as e:
        st.warning(f"Could not fetch sample values: {str(e)}")

    # Preprocess the question to replace potential brand mentions with actual brand names
    processed_question = question
    if "brand" in " ".join(db_info["columns"]).lower():
        # Extract potential brand names from the question
        words = question.split()
        for i, word in enumerate(words):
            # Skip common words
            if word.lower() in ["the", "for", "and", "to", "from"]:
                continue

            # Check if this might be a brand name (capitalized or known brand keyword)
            if len(word) > 3:
                # Clean the word
                clean_word = "".join(c for c in word if c.isalnum() or c.isspace())
                if clean_word:
                    # Try to find a matching brand
                    matched_brand = find_closest_brand(
                        clean_word, db_info.get("all_brands", [])
                    )
                    if matched_brand and matched_brand != clean_word:
                        # Replace in the question, preserving case and punctuation
                        processed_question = processed_question.replace(
                            word, f"{matched_brand}"
                        )

    # Improved prompt with more specific guidance and complete brand list
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert SQL query generator for a sales database.
        
Database information:
- Table name: sales
- Number of rows: {db_info["row_count"]}

Schema:
{schema_str}

Sample data:
{db_info["df_sample"].to_string()}

{f"Sample brands: {', '.join(sample_values.get('brands', []))}" if sample_values.get("brands") else ""}
{f"Date range: {sample_values.get('date_range')}" if sample_values.get("date_range") else ""}

Available brand names in the database:
{", ".join(db_info.get("all_brands", [])[:50])}
{f"... and {len(db_info.get('all_brands', [])) - 50} more brands" if len(db_info.get("all_brands", [])) > 50 else ""}

Your task is to convert the user's natural language question into a valid SQLite SQL query.

Guidelines:
1. Generate ONLY the SQL query, nothing else
2. Do not use backticks or markdown formatting
3. Ensure the query is executable SQLite syntax
4. Use appropriate aggregations, joins, and WHERE clauses as needed
5. For comparison questions, use proper aggregation and grouping to show both items side by side
6. For date filtering:
   - If dates are stored as TEXT: Use strftime() functions
   - If dates are stored as NUMERIC: Use appropriate comparison operators
7. Pay attention to capitalization in column names and string values
8. Only use brand names that exist in the database list provided above

For brand comparisons:
- Use proper OR conditions and GROUP BY clauses
- Consider using CASE statements for side-by-side comparisons
- If a brand name is misspelled or incomplete, use the closest match from the available brand list

For time period comparisons:
- Use date functions appropriate for the date format in the database
- Group by appropriate time periods (day, week, month)
""",
            ),
            ("human", processed_question),
        ]
    )

    # Format the prompt before invoking the LLM
    formatted_prompt = prompt.format_prompt(question=question)
    response = llm.invoke(formatted_prompt.to_messages())

    sql = response.content.strip()

    # Remove any code block markers
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql


# Add a debug function to show query details when needed
def debug_query(query, result_df):
    with st.expander("Debug Information"):
        st.code(query, language="sql")
        st.write(f"Query returned {len(result_df)} rows")
        if len(result_df) > 0:
            st.write("First few rows:")
            st.dataframe(result_df.head(3))
        else:
            st.warning("Query returned no results")


# Generate plot based on dataframe
def generate_plot(df, question):
    llm = get_llm()

    # Get column information
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert data visualization specialist.
        
I have a dataframe with the following columns:
{df.columns.tolist()}

Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}

Based on the dataframe and the user's question, generate Python code that creates an appropriate visualization.

Guidelines:
1. Use Plotly Express (import as px) or Plotly Graph Objects (import as go)
2. Create an appropriate chart (bar, line, scatter, pie, etc.) based on the data and question
3. Add a title, labels, and other formatting
4. End with st.plotly_chart(fig) to display the interactive chart in Streamlit
5. ONLY return the code, no explanations or comments
6. Do not forget to import pandas as pd

For brand comparison scenarios:
6. When comparing a specific brand with other brands:
   - Use grouped bar charts or multi-line charts for direct comparisons
   - Consider using a normalized scale (percentage) when comparing brands of different sizes
   - Highlight the main brand being analyzed with a distinct color
7. When comparing a brand across different branches:
   - Use faceted charts or grouped bar charts to show geographic distribution
   - Consider using maps if branch locations are available
   - Sort branches by performance for clearer insights
8. When comparing a brand across different time periods:
   - Use line charts with multiple lines for different time periods
   - Include trendlines to highlight growth patterns
   - For year-over-year comparisons, consider overlapping lines with different years

Sample data:
{df.head(5).to_string()}
""",
            ),
            ("human", f"Create a visualization for this question: {question}"),
        ]
    )

    # Format the prompt before invoking the LLM
    formatted_prompt = prompt.format_prompt()
    response = llm.invoke(formatted_prompt.to_messages())

    plot_code = response.content.strip()

    # Remove any code block markers
    plot_code = plot_code.replace("```python", "").replace("```", "").strip()

    # Fix common Plotly syntax issues
    plot_code = fix_plotly_code(plot_code)

    try:
        # Execute the plotting code
        exec_globals = {"px": px, "go": go, "df": df, "st": st}
        exec(plot_code, exec_globals)
        return True
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        st.code(plot_code, language="python")
        return False


# Function to fix common Plotly syntax issues
def fix_plotly_code(code):
    """Fix common syntax issues in Plotly code generated by LLMs"""

    # Replace incorrect 'marker_color' property in go.Bar
    if "go.Bar" in code and "marker_color" in code:
        # Replace direct marker_color with the correct marker=dict(color=...) syntax
        code = code.replace("marker_color=", "marker=dict(color=")

        # Fix any missing closing parentheses
        # Look for pattern like: marker=dict(color='blue'
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if (
                "marker=dict(color=" in line
                and ")" not in line.split("marker=dict(color=")[1]
            ):
                lines[i] = line + ")"
        code = "\n".join(lines)

    # Fix marker_colors for px.bar (should be color_discrete_sequence)
    if "px.bar" in code and "marker_colors" in code:
        code = code.replace("marker_colors=", "color_discrete_sequence=")

    # Fix other common Plotly syntax issues

    return code


# Improved generate_insights function to fix formatting issues
def generate_insights(df, question):
    llm = get_llm()

    # Format dataframe info
    if len(df) > 10:
        data_desc = f"{df.head(10).to_string()}\n... and {len(df) - 10} more rows"
    else:
        data_desc = df.to_string()

    # Add preprocessing to format numbers better
    if df.select_dtypes(include=["number"]).columns.size > 0:
        # Get numeric stats with better formatting
        numeric_summary = df.describe().round(2).to_string()
    else:
        numeric_summary = "No numeric columns for statistics"

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are an expert business analyst specialized in sales data.
        
Analyze the following data and provide meaningful insights related to the user's question.
Focus on trends, patterns, anomalies, and actionable insights.

Data (from query results):
{data_desc}

Additional statistics:
{numeric_summary}

Your response should:
1. Be concise and business-focused (3-5 paragraphs maximum)
2. Highlight key numbers and metrics using standard notation (no special formatting)
3. Show numbers in a readable format (e.g., "12.5 million" instead of "1.25e7")
4. Provide context and implications
5. Suggest potential actions or areas for further investigation
6. Use professional business language
7. DO NOT use any markdown formatting in the numbers

For data-driven analysis:
8. Focus on statistical observations and patterns in the actual data values
9. Identify key trends, correlations, and outliers based on numerical analysis
10. Provide comparative metrics when relevant (e.g., growth rates, market share)
11. For brand analysis:
   - Compare actual performance metrics between brands using the data
   - Calculate and interpret market share or performance distribution
   - Analyze performance across different segments (branches, time periods)
   - Highlight notable differences in metrics that reveal business insights
   - Support observations with specific numbers from the data
""",
            ),
            ("human", f"Provide data-driven insights for: {question}"),
        ]
    )

    formatted_prompt = prompt.format_prompt()
    response = llm.invoke(formatted_prompt.to_messages())
    return response.content.strip()


# Completely rewritten intent detection function
def detect_intent(question):
    llm = get_llm()

    # Initialize default intent values
    intent = {
        "requires_data": True,
        "requires_plot": True,
        "requires_insights": True,
        "plot_type": "bar",
    }

    # Function to ask a simple yes/no question
    def ask_yes_no(specific_question):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful assistant. Answer the question with ONLY 'yes' or 'no'. 
            Do not provide explanations or additional text.""",
                ),
                ("human", specific_question),
            ]
        )

        formatted_prompt = prompt.format_prompt()
        response = llm.invoke(formatted_prompt.to_messages())
        answer = response.content.lower().strip()
        return "yes" in answer

    # Function to determine best plot type
    def get_plot_type():
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a data visualization expert. Based on the question about sales data, 
            respond with ONLY ONE of these chart types that would best visualize the answer:
            bar, line, scatter, pie, area
            
            Just return the single word, nothing else.""",
                ),
                ("human", f"What is the best chart type for this question: {question}"),
            ]
        )

        formatted_prompt = prompt.format_prompt()
        response = llm.invoke(formatted_prompt.to_messages())
        plot_type = response.content.lower().strip()

        # Default to bar if response isn't one of the expected types
        valid_types = ["bar", "line", "scatter", "pie", "area"]
        return plot_type if plot_type in valid_types else "bar"

    try:
        # Handle common "top N" queries directly
        if any(
            x in question.lower() for x in ["top", "highest", "largest", "best"]
        ) and any(
            y in question.lower()
            for y in ["brand", "product", "category", "sales", "region"]
        ):
            # For "top N" type questions, we want both data and a bar chart
            intent["requires_data"] = True
            intent["requires_plot"] = True
            intent["requires_insights"] = False
            intent["plot_type"] = "bar"
        else:
            # For other queries, ask specific questions to determine intent
            intent["requires_data"] = ask_yes_no(
                f"Would showing a data table help answer this question about sales data: '{question}'?"
            )
            intent["requires_plot"] = ask_yes_no(
                f"Would a visualization or chart help answer this question about sales data: '{question}'?"
            )
            intent["requires_insights"] = ask_yes_no(
                f"Would business insights or analysis help answer this question about sales data: '{question}'?"
            )

            # Only determine plot type if we need a plot
            if intent["requires_plot"]:
                intent["plot_type"] = get_plot_type()

    except Exception as e:
        st.warning(
            f"Error detecting intent: {str(e)}. Defaulting to showing all outputs."
        )

    return intent


# Main Streamlit application
st.title("📊 Sales Data Assistant")

# Initialize database on first load
with st.spinner("Loading data..."):
    db_info = load_data_to_db()
    st.success(f"Loaded sales data ({db_info['row_count']} rows)")

# Simplified sidebar without mode selection
with st.sidebar:
    st.header("Options")

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query input (always Natural Language Query now)
query_input = st.chat_input("Ask a question about your sales data")
if query_input:
    st.session_state.messages.append({"role": "user", "content": query_input})

    with st.chat_message("user"):
        st.markdown(query_input)

    with st.chat_message("assistant"):
        # Initialize response_content variable to avoid NameError
        response_content = "I processed your query."

        with st.spinner("Analyzing your query..."):
            try:
                # Detect intent from the query
                intent = detect_intent(query_input)

                # Generate SQL from natural language
                sql_query = generate_sql(query_input, db_info)

                # Execute the query
                with st.spinner("Retrieving data..."):
                    try:
                        result_df = execute_query(sql_query)

                        # Create a list to collect response parts
                        response_parts = []

                        # Add debug information if query returns no results
                        if len(result_df) == 0:
                            debug_query(sql_query, result_df)
                            st.warning(
                                "No data matched your filters. Try checking different brand names or spellings."
                            )

                            # Try to help with debugging
                            simple_query = "SELECT COUNT(*) FROM sales"
                            total = execute_query(simple_query).iloc[0, 0]
                            st.write(f"Total rows in database: {total}")

                            response_content = (
                                "I couldn't find any data matching your criteria."
                            )
                        else:
                            # Show data result if intent requires it
                            if intent.get("requires_data", True) and len(result_df) > 0:
                                # Always show the full data table directly without preview option
                                st.subheader("Data Results")
                                st.dataframe(result_df, use_container_width=True)
                                response_parts.append("Here's the data you requested.")

                            # Generate plot if intent requires it
                            if intent.get("requires_plot", True) and len(result_df) > 0:
                                st.subheader("Visualization")
                                # Pass plot type to generate more specific visualizations
                                generate_plot(result_df, query_input)
                                response_parts.append(
                                    "I've created a visualization to help answer your question."
                                )

                            # Generate insights if intent requires it
                            if (
                                intent.get("requires_insights", True)
                                and len(result_df) > 0
                            ):
                                st.subheader("Analysis")
                                # Always use the full result dataframe for insights
                                insights = generate_insights(result_df, query_input)
                                st.markdown(insights)
                                response_parts.append(insights)

                            # Set content for chat history
                            if len(response_parts) > 1:
                                # Use only the first 2 items plus insights for the response content
                                # to avoid overly long messages in chat history
                                if (
                                    len(response_parts) > 2
                                    and len(response_parts[-1]) > 100
                                ):
                                    response_content = "\n\n".join(response_parts[:2])
                                else:
                                    response_content = "\n\n".join(response_parts)
                            else:
                                response_content = (
                                    response_parts[0]
                                    if response_parts
                                    else "Query processed successfully."
                                )

                    except Exception as e:
                        st.error(f"Error executing query: {str(e)}")
                        st.code(sql_query, language="sql")
                        response_content = (
                            f"I encountered an error executing the query: {str(e)}"
                        )

            except Exception as e:
                st.error(f"Error: {str(e)}")
                response_content = f"I encountered an error: {str(e)}"

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_content})
