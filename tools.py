# tools.py
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from functions import apply_filter_items,apply_sort_columns,apply_analyze_data,apply_get_unique_values


# === FILTER ITEMS ===

from typing import Optional, Dict, List, Any

def filter_items(filters: Optional[Dict[str, Any]] = None) -> str:
    """
    Defines a structured, logic-based filter expression for selecting products from a dataset.

    This tool enables the language model to construct complex logical filtering criteria
    by combining multiple conditions using explicit or implicit logical operators: AND, OR, and NOT.

    Parameters
    ----------
    filters : dict, optional
        A dictionary describing the filter logic. It may contain up to three top-level keys: "and", "or", and "not".

        - "and": A list of atomic filter conditions. All conditions in this list must be true for a row to be included.
          ❗ IMPORTANT: Each condition key in an "and" block must be unique. 
          Duplicate keys with different values in the same "and" block are **not allowed** and are considered illogical.
        
        - "or": A list of atomic filter conditions. At least one condition in this list must be true for a row to be included.
          ✅ The same condition key can appear multiple times with different values in an "or" block.
          For example, you can have multiple {"country_of_origin": "..."} conditions to check for multiple countries.
        
        - "not": A list of atomic filter conditions. Any row matching **any** condition in this list will be excluded.
          ✅ The same condition key can appear multiple times with different values in a "not" block.

        ✅ Example using explicit logical operators:
        --------------------------------------------
        {
            "and": [
                {"price_gte": 500},
                {"brand": "Sony"}
            ],
            "or": [
                {"country_of_origin": "Japan"},
                {"country_of_origin": "Turkey"}
            ],
            "not": [
                {"country_of_origin": "China"}
            ]
        }

        ✅ Example using implicit AND (default):
        ----------------------------------------
        If the dictionary does NOT contain "and", "or", or "not" as keys,
        then all top-level key-value pairs are automatically treated as an AND block.
        For example:
        {
            "brand": "Sony",
            "price_gte": 500
        }
        is interpreted as:
        {
            "and": [
                {"brand": "Sony"},
                {"price_gte": 500}
            ]
        }

        This ensures simple single-condition or multi-condition filters do not break
        if the logic block is omitted.

        Each atomic filter must be a dictionary with exactly one key-value pair.

        ✅ Supported atomic filter keys:
        --------------------------------
        - "price_gt" : float — Price must be greater than this value.
        - "price_gte" : float — Price must be greater than or equal to this value.
        - "price_lt" : float — Price must be less than this value.
        - "price_lte" : float — Price must be less than or equal to this value.
        - "date_added_min" : str — Include rows where 'date_added' is on or after this date (YYYY-MM-DD).
        - "date_added_max" : str — Include rows where 'date_added' is on or before this date (YYYY-MM-DD).
        - "date_modified_min" : str — Include rows where 'date_modified' is on or after this date (YYYY-MM-DD).
        - "date_modified_max" : str — Include rows where 'date_modified' is on or before this date (YYYY-MM-DD).
        - "brand" : str — Partial, case-insensitive match on brand.
        - "category_id" : int — Exact match on the category ID.
        - "industry_id" : int — Exact match on the industry ID.
        - "country_of_origin" : str — Exact, case-insensitive match on country of origin.
        - "oem_number" : float or int — Exact match on OEM number.
        - "product_number" : str — Exact match on product number.
        - "name" : str — Partial, case-insensitive match on product name.
        - "description" : str — Partial, case-insensitive match on description.
        - "hs_code" : str — Exact match on HS code.
        - "unique_id" : int — Exact match on unique product ID.

    Notes
    -----
    - If both explicit logic blocks ("and", "or", "not") and implicit top-level keys are provided,
      only the explicit logic blocks will be used.
    - Filters are evaluated in this order: AND → OR → NOT.
    - Unknown or unsupported filter keys will be ignored by the system.
    - Dates must be in ISO format (YYYY-MM-DD) and parsable by `pd.to_datetime`.
    - ❗ For `"and"` blocks: each condition key must appear only once.
    - ✅ For `"or"` and `"not"` blocks: repeating the same key with different values is allowed and valid.

    Returns
    -------
    str
        A confirmation message that filter parameters have been received.
        The actual filtering logic is applied elsewhere in the data pipeline.

    Examples
    --------
    1. Single filter (implicit AND):
    >>> filter_items(filters={"brand": "Sony"})
    Interpreted as:
    {"and": [{"brand": "Sony"}]}

    2. Multiple filters (implicit AND):
    >>> filter_items(filters={"price_gte": 500, "brand": "Sony"})

    3. Explicit AND + OR with same key repeated in OR:
    >>> filter_items(filters={
    ...     "and": [{"price_gte": 500}],
    ...     "or": [{"country_of_origin": "Japan"}, {"country_of_origin": "Turkey"}]
    ... })

    4. Exclude using NOT with same key repeated:
    >>> filter_items(filters={
    ...     "not": [{"country_of_origin": "China"}, {"country_of_origin": "Russia"}]
    ... })
    """
    return "Filter parameters received by the tool."





# === ANALYZE DATA ===
def analyze_data(
    group_by_column: Optional[Union[str, List[str]]] = None,
    agg_dict: Optional[Dict[str, Union[str, List[str]]]] = None,
    count_column: Optional[str] = None,
    include_group_size: bool = False,
    convert_numeric: bool = False
) -> str:
    """
    Prepare instructions for analyzing a dataset using grouping, aggregation, or value counting.

    This function defines the parameters for performing data analysis on a tabular dataset. It supports grouping by
    one or more columns, applying aggregation functions to specified columns, and counting values within a column.
    Only one of `agg_dict` or `count_column` should be provided at a time.

    Parameters
    ----------
    group_by_column : str, list of str, or None, optional
        The column or list of columns to group the data by before applying aggregation or counting.
        If None, operations will be applied without grouping.

    agg_dict : dict or None, optional
        A dictionary that defines which aggregation operations to apply on which columns.
        Keys should be column names, and values should be either a string (e.g., "mean", "sum")
        or a list of such strings.
        Example: {"price": "mean"} or {"sales": ["sum", "max"]}
        This parameter is mutually exclusive with `count_column`.

    count_column : str or None, optional
        The name of a column for which to compute value counts (frequencies), either globally or per group.
        Cannot be used together with `agg_dict`.

    include_group_size : bool, optional (default=False)
        If True and grouping is used, include a column named "group_size" in the result to indicate the
        number of records in each group.

    convert_numeric : bool, optional (default=False)
        If True, attempts to automatically convert string columns containing mostly numeric values
        into numeric types before analysis. Useful when data comes from CSVs or spreadsheets
        with numbers stored as strings.

    Returns
    -------
    str
        A message confirming that the analysis parameters were received and are ready to be processed.

    Constraints
    -----------
    - `agg_dict` and `count_column` cannot be used together.
    - All columns mentioned in `group_by_column`, `agg_dict`, or `count_column` must exist in the dataset.
    - This function only defines the analysis intent; actual computation happens elsewhere.

    Example
    -------
    >>> analyze_data(
            group_by_column="region",
            agg_dict={"revenue": ["sum", "mean"]},
            include_group_size=True,
            convert_numeric=True
        )
    >>> analyze_data(count_column="product_category")
    """
    return "Analysis parameters received."



# === GET UNIQUE VALUES ===
def get_unique_values(
    columns: Union[str, List[str]],
    dropna: bool = True,
    as_list: bool = False
) -> str:
    """
    Extract the unique values from one or more specified columns in the dataset.

    This function serves as a tool schema interface (stub) for downstream agents,
    enabling them to request unique value extraction from a DataFrame. It accepts
    column names and parameters to control whether NaN values should be dropped,
    and whether the result should be grouped as Python lists or shown row-wise.

    Parameters
    ----------
    columns : str or list of str
        The name of one or more columns for which to extract unique values.
        - If a single string is passed, only that column is processed.
        - If a list of strings is passed, each column is processed independently.

    dropna : bool, optional (default = True)
        Whether to exclude missing (NaN) values from the unique values:
        - True → Missing values will be removed.
        - False → Missing values will be retained if present.

    as_list : bool, optional (default = False)
        Whether to group each column’s unique values into a list:
        - True → Output will contain a single row with list objects in each column.
        - False → Output will show one unique value per row, per column.

    Returns
    -------
    str
        A confirmation string indicating that the parameters were received and
        validated. This stub does not return the actual result, but is used for
        validation before executing the real logic via `apply_get_unique_values`.

    Usage Examples
    --------------
    # Get all unique brands, excluding NaNs, row-wise
    get_unique_values(columns="brand", dropna=True, as_list=False)

    # Get unique categories and regions, as list form
    get_unique_values(columns=["category", "region"], as_list=True)

    Notes
    -----
    - This is a **tool schema stub function**; it validates inputs before the actual
      logic is applied via a separate execution function (`apply_get_unique_values`).
    - Always returns a string confirmation; it does **not** return the actual unique values.
    - The true result is produced by the corresponding backend implementation.
    """
    return "Unique columns parameters received."



# === SORT COLUMNS ===
from typing import Union, List

def sort_columns(
    columns: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    na_position: str = "last",
    ignore_index: bool = False
) -> str:
    """
    Tool Schema Function — Register sorting parameters for tabular product data.

    This function is used in an LLM-driven tool orchestration system to define 
    how a dataset should be sorted — without performing the actual sort. It 
    registers the user's sort intent (e.g. sort by price descending), which is 
    later executed by a separate downstream function.

    Parameters
    ----------
    columns : str or list of str
        Column or list of columns to sort by. These must be valid fields from 
        the dataset. Supported columns include:
            - 'unique_id'
            - 'name'
            - 'description'
            - 'oem_number'
            - 'product_number'
            - 'brand'
            - 'country_of_origin'
            - 'hs_code'
            - 'date_added'
            - 'date_modified'
            - 'category_id'
            - 'industry_id'
            - 'price'

        Examples:
        - `"price"`: sort by price
        - `["brand", "price"]`: sort by brand first, then price

    ascending : bool or list of bool, default True
        Defines the sort order:
        - `True`: ascending (low to high or A–Z)
        - `False`: descending (high to low or Z–A)
        - If a list, must match the number of columns.

        Examples:
        - `True`: sort all columns ascending
        - `[True, False]`: first ascending, second descending

    na_position : {'first', 'last'}, default 'last'
        Controls placement of NaN/missing values in the sorted result.
        - `'first'`: place NaNs at the beginning
        - `'last'`: place NaNs at the end

    ignore_index : bool, default False
        If True, the result of the eventual sort should reset the index to a 
        sequential 0-based range. This affects downstream display and storage.

    Returns
    -------
    str
        A message confirming that sort parameters were received. This message 
        can be logged or used to trigger the actual sort operation.

    Notes
    -----
    - This function is part of a tool schema; it does **not** sort the data.
    - Sorting is expected to be executed later by a dedicated function such as `apply_sort`.
    - This layer helps separate intent capture (via LLMs or user input) from execution.

    Example
    -------
    >>> sort_columns(columns="price", ascending=False)
    "Sort parameters received."

    >>> sort_columns(columns=["brand", "price"], ascending=[True, False])
    "Sort parameters received."
    """
    return "Sort parameters received."


# === FIND MISSING VALUES ===
def find_missing_values() -> str:
    """
    Find missing values in the dataset.
    """
    return "Find missing values tool called."


# === SUMMARIZE BY PERIOD ===
def summarize_by_period(
    date_col: str,
    freq: str
) -> str:
    """
    Group and summarize data by time period.
    """
    return "Summarize period parameters received."


tools_for_llm = [
    apply_filter_items,
    filter_items,
    analyze_data,
    get_unique_values,
    sort_columns,
    apply_sort_columns,
    find_missing_values,
    summarize_by_period,
]

available_functions_map = {
    "apply_filter_items": apply_filter_items,
    "apply_sort_columns":apply_sort_columns,
    "filter_items":filter_items,
    "apply_analyze_data": apply_analyze_data,
    "analyze_data":analyze_data,
    "get_unique_values": get_unique_values,
    "sort_columns": sort_columns,
    "find_missing_values": find_missing_values,
    "summarize_by_period": summarize_by_period,
     "apply_get_unique_values": apply_get_unique_values,
}