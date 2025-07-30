import os
import json
import pandas as pd
import re
import json

import google.generativeai as genai  # ✅ Using Gemini instead of Ollama 
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter

from tools import tools_for_llm, available_functions_map

# === CONFIGURE GEMINI ===
genai.configure(api_key="AIzaSyBDKS7IegBhInbUQcx0VsmUGuUoZrClbxs")  # Replace or load safely

llm = genai.GenerativeModel('gemini-1.5-pro')  # Or 'gemini-1.5-flash'

# === RAG SETUP ===
persist_directory = "chroma_db"
embedding_model = "all-MiniLM-L6-v2"

def create_vectorstore_from_df(df: pd.DataFrame):
    loader = DataFrameLoader(df)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_directory)
    vectordb.persist()

def get_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(persist_directory=persist_directory, embedding_function=embedding)

def retrieve_context(query: str, k: int = 3):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)
    return "\n".join([d.page_content for d in docs])

# === LOAD DATA ===
try:
    df = pd.read_csv("test_data.csv", parse_dates=["date_added", "date_modified"])
    if not os.path.exists(persist_directory):
        create_vectorstore_from_df(df)
except Exception as e:
    print(f"❌ Failed to load dataset: {e}")
    exit(1)

# === MAIN INTERACTION ===
def call_llm_with_tools(user_input: str):
    print(f"\n🔍 You asked: {user_input}")

    use_rag = any(k in user_input.lower() for k in ["product", "date", "price", "recent", "added"])
    context = retrieve_context(user_input) if use_rag else ""

    # We’ll ask Gemini: What tool should I run and with what arguments?
    tool_prompt = f"""
You are a Product Data Assistant.
Your ONLY job is to SELECT THE CORRECT TOOL from the list below and RETURN STRICTLY VALID JSON to call that tool — nothing else.

❌ NO explanations  
❌ NO greetings  
❌ NO Markdown or comments  
✅ Output ONE valid JSON object only

=== 🔧 AVAILABLE TOOLS ===

1️⃣ **filter_items**  
- Filter the product dataset by column values, ranges, or conditions.  
- Handles numeric, categorical, and date filters.

2️⃣ **sort_columns**  
- Sort the dataset by one or more columns.  
- Can sort ascending/descending, control null order, and reset index.

---

=== ✅ filter_items: Allowed Arguments ===

Inside \"filters\" you must wrap all conditions in one or more top-level logic blocks: \"and\", \"or\", \"not\".

Valid filter keys:
- `price_gt`: float — price > value
- `price_gte`: float — price ≥ value
- `price_lt`: float — price < value
- `price_lte`: float — price ≤ value
- `date_after`: str (YYYY-MM-DD)
- `date_before`: str (YYYY-MM-DD)
- `date_from`: str (YYYY-MM-DD)
- `date_to`: str (YYYY-MM-DD)
- `date_added_min`: str (YYYY-MM-DD)
- `date_added_max`: str (YYYY-MM-DD)
- `date_modified_min`: str (YYYY-MM-DD)
- `date_modified_max`: str (YYYY-MM-DD)
- `brand`: str
- `category_id`: int
- `color`: str
- `industry_id`: int
- `country_of_origin`: str
- `oem_number`: float
- `product_number`: str
- `name`: str
- `description`: str
- `hs_code`: str
- `unique_id`: int

✅ ALL filter keys MUST appear inside \"filters\" → \"and\"/\"or\"/\"not\" blocks.

⚙️ Filter Structure Rules

✅ Single-value filters:  
- Can be placed directly inside an `and` block.  
Example:  
{{
  "filters": {{
    "and": [
      {{ "country_of_origin": "Turkey" }},
      {{ "price_gt": 100 }}
    ]
  }}
}}

✅ Multi-value filters:  
- If a filter key must match one of multiple possible values, it MUST go inside an `or` (or `not`) block.  
- Never put multiple values for the same key directly inside `and`.
- ❗ **IMPORTANT:** You cannot split possible values for the same key between `and` and `or` — keep them together inside the same `or` (or `not`). For example, don’t put `country_of_origin = Turkey` in `and` and `country_of_origin = Japan` in `or`.

✔️ Correct usage with `or`:  
{{
  "filters": {{
    "or": [
      {{ "country_of_origin": "Turkey" }},
      {{ "country_of_origin": "Japan" }}
    ]
  }}
}}

✔️ Correct usage inside an `and` block with nested `or`:  
{{
  "filters": {{
    "and": [
      {{ "status": "active" }},
      {{
        "or": [
          {{ "country_of_origin": "Turkey" }},
          {{ "country_of_origin": "Japan" }}
        ]
      }}
    ]
  }}
}}

❌ Incorrect usage:  
{{
  "filters": {{
    "and": [
      {{ "country_of_origin": "Turkey" }},
      {{
        "or": [
          {{ "country_of_origin": "Japan" }}
        ]
      }}
    ]
  }}
}}  // ❌ INVALID: same key split across `and` and `or`

👉 Always wrap **all** possible values for the same key in a single `or` or `not`.

---

⚙️ TOOL PROMPT

You are a product data assistant.  
Your job is to SELECT THE CORRECT TOOL from the list below and GENERATE VALID, EXECUTABLE JSON to call that tool — nothing else.  
IMPORTANT: Your response MUST be strictly valid JSON, with no extra text, no explanations, no greetings.

---

🗂️ **Filter Logic Rules**

1️⃣ **Use `and`, `or`, and `not` ONLY at the top level.**  
   - There must be **NO nesting** of `or` inside `and`, or `not` inside `or`, etc.
   - All logical operators must be **siblings** at the same level under `filters`.

2️⃣ **Single-value filters** must go inside an `and` block (or whatever block makes sense).  
   Example:  
   {{
     "filters": {{
       "and": [
         {{ "brand": "Furuno" }},
         {{ "price_gt": 100 }}
       ]
     }}
   }}

3️⃣ **Multi-value filters** for the **same key** must go in an `or` or `not` block — never in `and`.  
   - You cannot split values for the same key across different blocks.
   - Example (**Correct**):  
   {{
     "filters": {{
       "or": [
         {{ "country_of_origin": "Japan" }},
         {{ "country_of_origin": "Turkey" }}
       ]
     }}
   }}

   ❌ Example (**Incorrect** — DO NOT DO THIS!):  
   {{
     "filters": {{
       "and": [
         {{ "country_of_origin": "Japan" }}
       ],
       "or": [
         {{ "country_of_origin": "Turkey" }}
       ]
     }}
   }}

4️⃣ **Combining conditions**:  
   If you need to combine `and`, `or`, and `not` conditions, use multiple sibling blocks.  
   Example:  
   {{
     "filters": {{
       "and": [
         {{ "date_after": "2024-12-01" }}
       ],
       "or": [
         {{ "country_of_origin": "Japan" }},
         {{ "country_of_origin": "Turkey" }}
       ],
       "not": [
         {{ "brand": "Furuno" }}
       ]
     }}
   }}

5️⃣ **NEVER nest blocks inside each other.**  
   ✔️ Correct: `and`, `or`, `not` are siblings.  
   ❌ Wrong: `and` contains an `or`.

👉 Always check that your final filter JSON keeps all logical blocks **flat** at the same level.

---

Use these rules every time you generate filter JSON.

🛡️ **ABSOLUTE RULES**

✅ ✅ ✅ **RULE 1 — `filters` MUST BE INSIDE `args`**

- The top-level JSON MUST always look like:
{{
  "tool_name": "filter_items",
  "args": {{
    "filters": {{ ... }}
  }}
}}

- `filters` MUST **never** appear directly at the top level — it must ONLY appear inside `"args"`.

✔️ **Correct:**
{{
  "tool_name": "filter_items",
  "args": {{
    "filters": {{
      "and": [ {{ "brand": "Furuno" }} ]
    }}
  }}
}}

❌ **Incorrect:**
{{
  "tool_name": "filter_items",
  "filters": {{
    "and": [ {{ "brand": "Furuno" }} ]
  }}
}}

---

✅ ✅ ✅ **RULE 2 — `and`, `or`, `not` MUST BE 1000% SIBLINGS**

- If you use `and`, `or`, or `not` they must always be direct siblings INSIDE `filters`.
- They must NEVER be nested inside each other.
- They must NEVER appear at multiple nesting levels.
- They must ALWAYS be at the same level under `filters`.

✔️ **Correct:**
{{
  "args": {{
    "filters": {{
      "and": [ {{ "date_after": "2024-12-01" }} ],
      "or": [
        {{ "country_of_origin": "Japan" }},
        {{ "country_of_origin": "Turkey" }}
      ],
      "not": [
        {{ "brand": "Furuno" }}
      ]
    }}
  }}
}}

❌ **Incorrect — nesting is NOT ALLOWED:**
{{
  "args": {{
    "filters": {{
      "and": [
        {{
          "or": [
            {{ "country_of_origin": "Japan" }},
            {{ "country_of_origin": "Turkey" }}
          ]
        }},
        {{ "date_after": "2024-12-01" }}
      ]
    }}
  }}
}}

❌ **Incorrect — `not` inside `or` is NOT ALLOWED:**
{{
  "args": {{
    "filters": {{
      "or": [
        {{
          "not": [
            {{ "brand": "Furuno" }}
          ]
        }}
      ]
    }}
  }}
}}

---

✅ ✅ ✅ **RULE 3 — SAME KEY MULTI-VALUE**

- If you have multiple values for the SAME KEY (e.g. `country_of_origin` is “Japan” OR “Turkey”), they MUST go under `or` or `not` only.
- They must NEVER be split between `and` and `or`.  
- They must NEVER be mixed into `and` directly.

✔️ **Correct:**
{{
  "args": {{
    "filters": {{
      "or": [
        {{ "country_of_origin": "Japan" }},
        {{ "country_of_origin": "Turkey" }}
      ]
    }}
  }}
}}

❌ **Incorrect — values split across `and` and `or`:**
{{
  "args": {{
    "filters": {{
      "and": [ {{ "country_of_origin": "Japan" }} ],
      "or": [ {{ "country_of_origin": "Turkey" }} ]
    }}
  }}
}}

---

✅ ✅ ✅ **RULE 4 — NO COMMENTS, NO TEXT, NO EXTRAS**

- The final output must ONLY contain the valid JSON object.  
- No explanations.  
- No markdown.  
- No extra words.  
- Only `{{` and `}}` JSON syntax.

---

✅ ✅ ✅ **RULE 5 — EXAMPLE OUTPUT (FINAL)**

A correct final output looks like:
{{
  "tool_name": "filter_items",
  "args": {{
    "filters": {{
      "and": [
        {{ "date_after": "2024-12-01" }},
        {{ "price_gt": 100 }}
      ],
      "or": [
        {{ "country_of_origin": "Japan" }},
        {{ "country_of_origin": "Turkey" }}
      ],
      "not": [
        {{ "brand": "Furuno" }}
      ]
    }}
  }}
}}

---

🔥 You must ALWAYS follow these rules 1000%.  
🔥 There is NEVER an exception.  
🔥 There must NEVER be any nesting of `and` inside `or`, or `or` inside `and`, or `not` inside `or` — ALL must be siblings directly under `filters`.  
🔥 The filter block must ALWAYS be wrapped inside `args`.  
🔥 The final JSON must ALWAYS match the tool’s expected schema exactly.

---

✔️ Follow these rules every single time.  
❌ Do not ever break them.


=== ✅ sort_columns: Allowed Arguments ===

These keys go directly inside \"args\" — do NOT wrap in \"filters\":
- `columns`: str or list[str] — Must be valid column names:  
  \"unique_id\", \"name\", \"description\", \"oem_number\", \"product_number\", \"brand\", \"country_of_origin\", \"hs_code\", \"date_added\", \"date_modified\", \"category_id\", \"industry_id\", \"price\"
- `ascending`: bool or list[bool] — True = ascending, False = descending. List must match columns.
- `na_position`: \"first\" or \"last\"
- `ignore_index`: bool

---

=== 🚫 LOGICAL STRUCTURE RULES ===

✔️ `and`, `or`, `not` must be direct top-level keys inside \"filters\".  
✔️ NEVER nest logic blocks inside each other.

✅ GOOD:
{{
  \"filters\": {{
    \"and\": [ {{...}} ],
    \"or\": [ {{...}} ],
    \"not\": [ {{...}} ]
  }}
}}

🚫 BAD:
{{
  \"filters\": {{
    \"and\": [ {{ \"or\": [ ... ] }} ]
  }}
}}

✔️ If filtering the same field for multiple possible values, use `or`.  
❌ Do NOT list the same key multiple times inside `and`.

Example GOOD:
{{
  \"filters\": {{
    \"and\": [
      {{ \"price_gt\": 100 }},
      {{ \"date_after\": \"2024-12-01\" }}
    ],
    \"or\": [
      {{ \"country_of_origin\": \"Japan\" }},
      {{ \"country_of_origin\": \"Turkey\" }}
    ]
  }}
}}

---
########################################
Function: analyze_data
########################################

\"\"\"
Analyze a pandas DataFrame with flexible grouping and aggregation options.

This function allows data analysis through column selection, optional group-by logic,
value counting, and aggregation. If grouping is enabled but no aggregation or value counting
is specified, it defaults to a plain group-by view. It is designed to be modular and easily
integrated into LLM tools for structured tabular insights.

Parameters
----------
df : pd.DataFrame
    The input DataFrame to be analyzed.

group_by_column : Optional[Union[str, List[str]]]
    A column or list of columns to group the data by. If not provided, no grouping is done.

columns : Optional[List[str]]
    Specific columns to include in the output. If None, all columns may be included.

agg_dict : Optional[Dict[str, Union[str, List[str]]]]
    Dictionary mapping column names to one or more aggregation functions.
    E.g., {{ "sales": "sum", "quantity": ["mean", "max"] }}

value_counts : Optional[List[str]]
    List of column names for which to compute value counts.

dropna_group_keys : bool
    Whether to drop rows with null values in the group_by_column(s). Defaults to True.

flatten_column_names : bool
    Whether to flatten multi-index columns (resulting from multiple aggs). Defaults to True.

Returns
-------
pd.DataFrame
    A DataFrame with applied aggregations, value counts, or group-by structure.
\"\"\"

LLM Tool Prompt for `analyze_data`:
{{
  "name": "analyze_data",
  "description": "Analyze a DataFrame by optionally grouping, aggregating, and/or counting values for specified columns.",
  "parameters": {{
    "type": "object",
    "properties": {{
      "group_by_column": {{
        "type": ["string", "array"],
        "description": "One or more column names to group by. Required for grouping-based summaries."
      }},
      "columns": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "List of columns to include in analysis. If omitted, defaults to all available columns."
      }},
      "agg_dict": {{
        "type": "object",
        "description": "Dictionary mapping column names to aggregation functions (e.g., {{ 'price': 'mean' }} or {{ 'price': ['min', 'max'] }})."
      }},
      "value_counts": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "List of column names for which to return value counts instead of aggregation."
      }},
      "dropna_group_keys": {{
        "type": "boolean",
        "description": "Whether to drop NA values in group_by_column(s) before analysis. Defaults to true."
      }},
      "flatten_column_names": {{
        "type": "boolean",
        "description": "Whether to flatten the resulting DataFrame columns if multiple aggregations are used. Defaults to true."
      }}
    }},
    "required": []
  }}
}}

########################################
Function: summarize_by_period
########################################

\"\"\"
Adds a derived period-based column to the DataFrame (based on a datetime column)
and applies `analyze_data` grouped by that period.

This is useful for time-based summaries such as:
- Daily breakdowns
- Weekly trends
- Monthly aggregations
- Quarterly or yearly reporting

It converts the provided `date_col` to datetime, derives the appropriate label column
based on `freq`, and calls `analyze_data` grouped by that derived column.

Parameters
----------
df : pd.DataFrame
    The DataFrame containing the date column.

date_col : str
    The name of the datetime column to use for period extraction (e.g., 'date_added').

freq : str
    Frequency of the time grouping. One of:
    - 'day': Groups by weekday name
    - 'week': Groups by ISO week number (year + week label)
    - 'month': Groups by calendar month (year + month)
    - 'quarter': Groups by calendar quarter (e.g., '2025Q1')
    - 'year': Groups by year

**kwargs : Any
    All other keyword arguments are forwarded to the `analyze_data` function, such as
    agg_dict, columns, value_counts, etc.

Returns
-------
pd.DataFrame
    A grouped and analyzed DataFrame using the derived time period as the group_by column.
\"\"\"

LLM Tool Prompt for `summarize_by_period`:
{{
  "name": "summarize_by_period",
  "description": "Summarize a DataFrame by deriving time-based periods from a date column and grouping data accordingly.",
  "parameters": {{
    "type": "object",
    "properties": {{
      "date_col": {{
        "type": "string",
        "description": "The column containing datetime values to use for deriving periods (e.g., 'date_added')."
      }},
      "freq": {{
        "type": "string",
        "enum": ["day", "week", "month", "quarter", "year"],
        "description": "The time period granularity for grouping. One of: 'day', 'week', 'month', 'quarter', 'year'."
      }},
      "group_by_column": {{
        "type": ["string", "array"],
        "description": "Additional column(s) to group by alongside the time period, if desired."
      }},
      "columns": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "List of columns to include in analysis."
      }},
      "agg_dict": {{
        "type": "object",
        "description": "Dictionary mapping columns to aggregation functions."
      }},
      "value_counts": {{
        "type": "array",
        "items": {{
          "type": "string"
        }},
        "description": "List of column names for which to compute value counts."
      }},
      "dropna_group_keys": {{
        "type": "boolean",
        "description": "Drop nulls in group-by columns. Defaults to true."
      }},
      "flatten_column_names": {{
        "type": "boolean",
        "description": "Whether to flatten resulting column names after aggregation. Defaults to true."
      }}
    }},
    "required": ["date_col", "freq"]
  }}
}}


=== 🚫 ABSOLUTE RULES ===

✔️ Output must be **one JSON object**:  
{{
  \"tool_name\": \"<tool_name>\",
  \"args\": {{ ... }}
}}

✔️ For `filter_items`, wrap all filter conditions inside \"filters\" → \"and\", \"or\", \"not\"  
✔️ For `sort_columns`, do NOT include \"filters\".

---

=== FINAL ===

✅ You must output **ONLY valid JSON** — no extra characters.  
✅ Always use `{{` and `}}` properly when needed.  
✅ Never output free text or comments.

---

=== CONTEXT ===  
{{{{ f"{context}" }}}}

=== USER QUERY ===  
{{{{ f"{user_input}" }}}}
"""



    import re
    import json

    response = llm.generate_content(tool_prompt).text.strip()

    # ✅ Remove ```json at start and ``` at end — robust version:
    response_clean = re.sub(r"^```json", "", response.strip(), flags=re.IGNORECASE).strip()
    response_clean = re.sub(r"^```", "", response_clean).strip()
    response_clean = re.sub(r"```$", "", response_clean).strip()

    # ✅ Optional: fix trailing commas
    response_clean = re.sub(r",\s*}", "}", response_clean)
    response_clean = re.sub(r",\s*]", "]", response_clean)

    print("📝 Cleaned LLM output:", response_clean)

    # ✅ Parse
    try:
        parsed = json.loads(response_clean)
        tool_name = parsed["tool_name"]
        tool_args = parsed["args"]
    except Exception as e:
        print(f"❌ Could not parse LLM output: {e}\nRaw cleaned output:\n{response_clean}")
        raise SystemExit

    print(f"\n🛠️ Tool requested: {tool_name}")
    print(f"🧾 Arguments: {json.dumps(tool_args, indent=2)}")

    # Import your tools
    from tools import available_functions_map

    # === Run tool ===
    if tool_name == "filter_items":
        confirmation = available_functions_map[tool_name](filters=tool_args["filters"])
        print(confirmation)
        result = available_functions_map["apply_filter_items"](
            df,
            filters=tool_args["filters"]
        )

    elif tool_name == "sort_columns":
        confirmation = available_functions_map[tool_name](
            columns=tool_args["columns"],
            ascending=tool_args.get("ascending", True),
            na_position=tool_args.get("na_position", "last"),
            ignore_index=tool_args.get("ignore_index", False)
        )
        print(confirmation)
        result = available_functions_map["apply_sort_columns"](
            df,
            columns=tool_args["columns"],
            ascending=tool_args.get("ascending", True),
            na_position=tool_args.get("na_position", "last"),
            ignore_index=tool_args.get("ignore_index", False)
        )

    elif tool_name == "analyze_data":
        confirmation = available_functions_map[tool_name](
            group_by_column=tool_args.get("group_by_column"),
            agg_dict=tool_args.get("agg_dict"),
            count_column=tool_args.get("count_column"),
            include_group_size=tool_args.get("include_group_size", False),
            convert_numeric=tool_args.get("convert_numeric", False)
        )
        print(confirmation)
        result = available_functions_map["apply_analyze_data"](
            df,
            group_by_column=tool_args.get("group_by_column"),
            agg_dict=tool_args.get("agg_dict"),
            count_column=tool_args.get("count_column"),
            include_group_size=tool_args.get("include_group_size", False),
            convert_numeric=tool_args.get("convert_numeric", False)
        )

    elif tool_name == "get_unique_values":
        confirmation = available_functions_map[tool_name](
            columns=tool_args.get("columns"),
            dropna=tool_args.get("dropna", True),
            as_list=tool_args.get("as_list", False)
        )
        print(confirmation)
        result = available_functions_map["apply_get_unique_values"](
            df,
            columns=tool_args.get("columns"),
            dropna=tool_args.get("dropna", True),
            as_list=tool_args.get("as_list", False)
        )

    elif tool_name in available_functions_map:
        result = available_functions_map[tool_name](df, **tool_args)

    elif tool_name == "find_missing_values":
        result = df.isnull().sum().to_frame("missing_count")

    else:
        print(f"❌ Tool '{tool_name}' not found.")
        return

# === Format results ===
    if isinstance(result, pd.DataFrame) and not result.empty:
        rows = []
        for i, row in result.iterrows():
            row_str = f"{i+1}. " + ", ".join([f"{col}: {row[col]}" for col in result.columns])
            rows.append(row_str)
        result_preview = "\n".join(rows)

    if isinstance(result, pd.DataFrame):
        result_preview = result.head(10).to_string(index=False) if not result.empty else "⚠️ No matching rows found."
    elif isinstance(result, dict):
        result_preview = json.dumps(result, indent=2)
    else:
        result_preview = str(result)

    # === Call Gemini again for summary ===
    summary_prompt = f"Here is the result:\n{result_preview}\nSummarize this for the user."
    summary = llm.generate_content(summary_prompt).text

    numbered_data = apply_get_final_product_data(result)

    print(f"\n✅ Tool executed successfully.")
    print(f"\nThe below is the output table")
    print(f"{numbered_data}")
    print(f"\n🧠 Summary:\n{summary}")



# === TOOL HELPERS ===

from typing import Optional
import pandas as pd



def apply_get_final_product_data(df):
    """
    Format the entire DataFrame as a single plain-text table.

    This function converts the full DataFrame into a classic
    row-and-column table representation using pandas' built-in
    string conversion.

    Example output:
        name      price   feature
        Widget A  19.99   Compact
        Widget B  29.99   Durable

    This is useful for showing the raw product data in a clean,
    readable format that can be easily included in a prompt
    or displayed to the user.

    Args:
        df (pd.DataFrame): The input DataFrame containing product data.

    Returns:
        str: The entire DataFrame formatted as a plain-text table.
    """
    return df.to_string(index=True)


# === CLI ===
if __name__ == "__main__":
    print("🤖 Local Product Data Assistant ready. Type your query below. (Type 'exit' to quit.)")
    while True:
        try:
            user_input = input("\n🗣️  You: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("👋 Goodbye!")
                break
            elif user_input == "":
                continue
            else:
                call_llm_with_tools(user_input)
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Exiting.")
            break
