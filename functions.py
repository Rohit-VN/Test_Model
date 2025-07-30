

import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union, List, Dict, Any, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import re
from typing import Union, Optional, List, Dict, Any, Tuple, Set

all_columns = [
    "price_gt", "price_gte", "price_lt", "price_lte",
    "date_from", "date_to", "date_after", "date_before",
    "date_added_min", "date_added_max",
    "date_modified_min", "date_modified_max",
    "brand", "category_id", "industry_id",
    "country_of_origin", "oem_number", "product_number",
    "name", "description", "hs_code", "unique_id"
]


df = pd.read_csv(
    'test_data.csv',
    parse_dates=['date_added', 'date_modified']
)




# In[239]:


def apply_price_range_filter(df: pd.DataFrame, price_col: str, **price_kwargs) -> pd.DataFrame:

    if price_col not in df.columns:
        print(f"Warning: Price column '{price_col}' not found!")
        return df.copy()

    mask = pd.Series(True, index=df.index)

    if 'price_gt' in price_kwargs:
        mask &= df[price_col] > price_kwargs['price_gt']
        print(f"Applied price_gt filter: > {price_kwargs['price_gt']}")

    if 'price_gte' in price_kwargs:
        mask &= df[price_col] >= price_kwargs['price_gte']
        print(f"Applied price_gte filter: >= {price_kwargs['price_gte']}")

    if 'price_lt' in price_kwargs:
        mask &= df[price_col] < price_kwargs['price_lt']
        print(f"Applied price_lt filter: < {price_kwargs['price_lt']}")

    if 'price_lte' in price_kwargs:
        mask &= df[price_col] <= price_kwargs['price_lte']
        print(f"Applied price_lte filter: <= {price_kwargs['price_lte']}")

    if 'price_between' in price_kwargs:
        min_val, max_val = price_kwargs['price_between']
        mask &= (df[price_col] >= min_val) & (df[price_col] <= max_val)
        print(f"Applied price_between filter: {min_val} <= price <= {max_val}")

    return mask


# In[ ]:


from typing import Optional, Tuple
import pandas as pd
from datetime import timedelta

def apply_date_range_filter(
    df: pd.DataFrame,
    date_col: str,
    handle_nulls: str = 'exclude',
    timezone: Optional[str] = None,
    date_format: Optional[str] = None,
    verbose: bool = False,
    *,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    date_after: Optional[str] = None,
    date_before: Optional[str] = None,
    date_between: Optional[Tuple[str, str]] = None,
    days_ago: Optional[int] = None,
    date_today: Optional[bool] = None
) -> pd.Series:

    # --- Example simplified implementation ---

    # Defensive checks (you can expand this)
    if date_col not in df.columns:
        raise KeyError(f"Column '{date_col}' not found in DataFrame.")

    date_series = df[date_col]

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(date_series):
        date_series = pd.to_datetime(date_series, format=date_format, errors='coerce')

    # Handle nulls
    if handle_nulls == 'exclude':
        mask = ~date_series.isna()
    elif handle_nulls == 'include':
        mask = pd.Series(True, index=df.index)
    elif handle_nulls == 'error' and date_series.isna().any():
        raise ValueError(f"Null values found in '{date_col}'")

    else:
        mask = pd.Series(True, index=df.index)

    # Apply filters one by one (only if they exist)
    if date_from:
        mask &= date_series >= pd.to_datetime(date_from)
    if date_to:
        mask &= date_series <= pd.to_datetime(date_to)
    if date_after:
        mask &= date_series > pd.to_datetime(date_after)
    if date_before:
        mask &= date_series < pd.to_datetime(date_before)
    if date_between:
        start, end = date_between
        mask &= (date_series >= pd.to_datetime(start)) & (date_series <= pd.to_datetime(end))
    if days_ago is not None:
        cutoff = pd.Timestamp.now(tz=timezone) - timedelta(days=days_ago)
        mask &= date_series >= cutoff
    if date_today:
        today = pd.Timestamp.now(tz=timezone).normalize()
        tomorrow = today + timedelta(days=1)
        mask &= (date_series >= today) & (date_series < tomorrow)

    return mask


# In[243]:


def detect_range_parameters(kwargs: dict) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Separate range parameters from regular filter parameters.

    Args:
        kwargs: Dictionary of filter parameters

    Returns:
        Tuple containing:
        - price_params: Dictionary of price range parameters
        - date_params: Dictionary of date range parameters  
        - regular_params: Dictionary of regular filter parameters

    Possible Outputs and Scenarios:

    1. ALL EMPTY DICTIONARIES:
       - Input: {} or None values only
       - Output: ({}, {}, {})

    2. ONLY PRICE PARAMETERS:
       - Input: {'price_min': 1000, 'price_max': 5000}
       - Output: ({'price_min': 1000, 'price_max': 5000}, {}, {})

    3. ONLY DATE PARAMETERS:
       - Input: {'date_from': '2025-01-01', 'date_added_min': '2025-02-01'}
       - Output: ({}, {'date_from': '2025-01-01', 'date_added_min': '2025-02-01'}, {})

    4. ONLY REGULAR PARAMETERS:
       - Input: {'brand': 'Teledyne', 'category_id': 5}
       - Output: ({}, {}, {'brand': 'Teledyne', 'category_id': 5})

    5. MIXED PARAMETERS:
       - Input: {'price_min': 1000, 'date_from': '2025-01-01', 'brand': 'Teledyne'}
       - Output: ({'price_min': 1000}, {'date_from': '2025-01-01'}, {'brand': 'Teledyne'})

    6. EDGE CASES:
       - None values: Preserved in their respective categories
       - Empty strings: Preserved in their respective categories
       - Case sensitivity: Exact match required
    """

    # Handle edge case: None input
    if kwargs is None:
        return {}, {}, {}

    # Handle edge case: empty dict
    if not kwargs:
        return {}, {}, {}

    price_prefixes = {
        'price_min', 'price_max', 'price_gt', 'price_gte', 
        'price_lt', 'price_lte', 'price_between'
    }

    date_prefixes = {
        'date_from', 'date_to', 'date_after', 'date_before', 
        'date_between', 'days_ago', 'date_today'
    }

    # Column-specific date range patterns
    date_column_patterns = ['date_added_', 'date_modified_']

    price_params = {}
    date_params = {}
    regular_params = {}

    for key, value in kwargs.items():
        # Convert key to string to handle potential non-string keys
        key_str = str(key)

        if key_str in price_prefixes:
            price_params[key] = value
        elif key_str in date_prefixes:
            date_params[key] = value
        elif any(key_str.startswith(pattern) for pattern in date_column_patterns):
            # Handle column-specific date ranges like date_added_from, date_modified_to
            date_params[key] = value
        else:
            regular_params[key] = value

    return price_params, date_params, regular_params


# In[244]:


# Include the filter implementation here
def apply_regular_filters(df: pd.DataFrame, **kwargs) -> pd.Series:
    """
    Apply regular filtering logic with improved error handling and edge case management.

    Args:
        df: Input DataFrame
        **kwargs: Filter conditions where key=column_name, value=filter_value

    Returns:
        pd.Series: Boolean mask for filtering

    Features:
        - Case-insensitive string matching for object columns
        - Partial matching for strings
        - Multiple value support (OR logic)
        - Exact matching for numeric columns
        - Proper handling of None/NaN values
        - No modification of original DataFrame
    """
    if df.empty:
        print("Warning: DataFrame is empty")
        return pd.Series(dtype=bool, index=df.index, name='empty_mask')

    if not kwargs:
        print("No filter conditions provided")
        return pd.Series(True, index=df.index, name='no_filter_mask')

    condition_masks = []
    for col, value in kwargs.items():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
            continue

        if df[col].isnull().all():
            print(f"Warning: Column '{col}' contains only NaN values. Skipping condition.")
            continue

        condition_mask = _process_single_condition(df, col, value)
        if condition_mask is not None:
            condition_masks.append(condition_mask)

    if not condition_masks:
        print("Warning: No valid filter conditions found. Returning empty result.")
        return pd.Series(False, index=df.index, name='no_valid_conditions')

    final_mask = condition_masks[0]
    for mask in condition_masks[1:]:
        final_mask = final_mask & mask

    matches = final_mask.sum()
    print(f"Filter applied: {matches} out of {len(df)} rows match all conditions")
    return final_mask

def _process_single_condition(df: pd.DataFrame, col: str, value: Any) -> Union[pd.Series, None]:
    """Process a single filter condition."""
    col_dtype = df[col].dtype

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return df[col].isna()

    if isinstance(value, (list, tuple, set)):
        return _handle_multiple_values(df, col, value, col_dtype)

    return _handle_single_value(df, col, value, col_dtype)

def _handle_multiple_values(df: pd.DataFrame, col: str, values: Union[List, Tuple, Set], col_dtype) -> Union[pd.Series, None]:
    """Handle multiple values with OR logic."""
    non_none_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    none_values = [v for v in values if v is None or (isinstance(v, float) and pd.isna(v))]

    masks = []

    if non_none_values:
        if pd.api.types.is_object_dtype(col_dtype):
            value_masks = [_handle_single_value(df, col, v, col_dtype) for v in non_none_values]
            value_masks = [m for m in value_masks if m is not None]
            if value_masks:
                combined_mask = value_masks[0]
                for mask in value_masks[1:]:
                    combined_mask = combined_mask | mask
                masks.append(combined_mask)
        else:
            masks.append(df[col].isin(non_none_values))

    if none_values:
        masks.append(df[col].isna())

    if not masks:
        print(f"Warning: No valid values found in list for column '{col}'")
        return None

    final_mask = masks[0]
    for mask in masks[1:]:
        final_mask = final_mask | mask

    return final_mask

def _handle_single_value(df: pd.DataFrame, col: str, value: Any, col_dtype) -> Union[pd.Series, None]:
    """Handle a single value."""
    if pd.api.types.is_object_dtype(col_dtype):
        working_series = df[col].fillna('').astype(str).str.lower()
        escaped_value = re.escape(str(value).lower())
        mask = working_series.str.contains(escaped_value, na=False, regex=True)
        if not mask.any():
            print(f"Warning: Value '{value}' not found in column '{col}'. No matches.")
            return pd.Series(False, index=df.index)
        return mask
    else:
        try:
            if pd.api.types.is_numeric_dtype(col_dtype):
                converted_value = pd.to_numeric(value, errors='coerce')
                if pd.isna(converted_value):
                    print(f"Warning: Value '{value}' cannot be converted to numeric for column '{col}'")
                    return pd.Series(False, index=df.index)
                mask = df[col] == converted_value
            else:
                mask = df[col] == value

            if not mask.any():
                print(f"Warning: Value '{value}' not found in column '{col}'. No matches.")
                return pd.Series(False, index=df.index)

            return mask
        except Exception as e:
            print(f"Warning: Error processing value '{value}' for column '{col}': {str(e)}")
            return pd.Series(False, index=df.index)



# In[246]:


import pandas as pd
from typing import Any, Dict, List

def apply_filter_items(df: pd.DataFrame, *, filters: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> pd.DataFrame:
    """
    Filters a DataFrame of product information based on a structured, logic-based filtering schema.

    This function allows flexible logical filtering using a structured `filters` dictionary that supports
    combinations of `AND`, `OR`, and `NOT` conditions. By default, all conditions are treated as `AND`
    conditions unless `or` or `not` blocks are explicitly specified.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing product data. It must have the following columns:
        - "price" (float)
        - "date_added" (datetime or string parsable by `pd.to_datetime`)
        - "date_modified" (datetime or string parsable by `pd.to_datetime`)
        - "brand", "name", "description", "country_of_origin" (str)
        - "category_id", "industry_id", "unique_id" (int)
        - "oem_number" (float or int)
        - "product_number", "hs_code" (str)

    filters : dict, optional
        A dictionary defining the logical filter structure.

        Structure:
        ----------
        By default, all top-level keys are treated as `AND` conditions.
        To combine conditions differently, use `"and"`, `"or"`, or `"not"` keys explicitly.

        Example structures:
        -------------------
        # Implicit AND (default)
        filters = {
            "price_gte": 500,
            "brand": "sony"
        }

        # Explicit AND/OR/NOT
        filters = {
            "and": [<condition_dict_1>, <condition_dict_2>, ...],
            "or":  [<condition_dict_3>, ...],
            "not": [<condition_dict_4>, ...]
        }

        Multiple values for the same key:
        ---------------------------------
        ✅ Allowed for `"or"` and `"not"` blocks.  
        This allows queries like:
            - country_of_origin is either "Japan" OR "Turkey"
            - NOT country_of_origin "China" OR "India"

        ❌ NOT allowed for `"and"` block:
            - For `"and"` each condition must be unique. Multiple same-key checks in `"and"` would conflict.

        Example using multiple same keys in OR:
        ---------------------------------------
        filters = {
            "or": [
                {"country_of_origin": "Japan"},
                {"country_of_origin": "Turkey"}
            ],
            "and": [
                {"date_after": "2025-01-01"}
            ]
        }

        Logical meaning:
        ----------------
        - `AND` (default): ALL listed conditions must be true for a row to be included.
        - `OR`: At least ONE of these conditions must be true for a row to be included.
        - `NOT`: Rows matching ANY of these conditions will be excluded.

        Each atomic condition must be a dictionary with exactly one key-value pair,
        or a top-level key in the implicit AND style.

    Supported Filter Keys (Atomic Conditions)
    -----------------------------------------
    - "price_gt": float — keep rows where price > value
    - "price_gte": float — price >= value
    - "price_lt": float — price < value
    - "price_lte": float — price <= value
    - "date_added_min": str — date_added >= this date
    - "date_added_max": str — date_added <= this date
    - "date_modified_min": str — date_modified >= this date
    - "date_modified_max": str — date_modified <= this date
    - "date_from": str — same as date_added >= this date
    - "date_to": str — same as date_added <= this date
    - "date_after": str — same as date_added > this date
    - "date_before": str — same as date_added < this date
    - "brand": str — partial, case-insensitive match on brand name
    - "category_id": int — exact match
    - "industry_id": int — exact match
    - "country_of_origin": str — exact, case-insensitive match
    - "oem_number": float/int — exact match
    - "product_number": str — exact match
    - "name": str — partial, case-insensitive match
    - "description": str — partial, case-insensitive match
    - "hs_code": str — exact match
    - "unique_id": int — exact match

    Notes
    -----
    - Strings like "name", "brand", "description" use `.str.contains(..., case=False)` (partial match).
    - "country_of_origin" uses `.str.lower() == value.lower()` (exact match, case-insensitive).
    - Dates must be ISO format (YYYY-MM-DD) and parsable by `pd.to_datetime()`.
    - Unknown or unsupported filter keys will be ignored with a debug warning.
    - Logical order: AND is always first, then OR is added, then NOT removes rows.
    - Multiple same-key conditions are valid ONLY inside OR or NOT.

    Returns
    -------
    pd.DataFrame
        A new filtered DataFrame containing only rows that match the logical criteria.
    """

    if not filters:
        return df  # Nothing to filter

    # If no explicit logical blocks, treat all top-level keys as AND.
    if not any(k in filters for k in ("and", "or", "not")):
        filters = {"and": [{k: v} for k, v in filters.items()]}

    df_copy = df.copy()

    def apply_atomic_filter(df: pd.DataFrame, cond: Dict[str, Any]) -> pd.DataFrame:
        key, value = next(iter(cond.items()))

        if key == "price_gt":
            return df[df["price"] > value]
        elif key == "price_gte":
            return df[df["price"] >= value]
        elif key == "price_lt":
            return df[df["price"] < value]
        elif key == "price_lte":
            return df[df["price"] <= value]
        elif key in ("date_added_min", "date_from"):
            return df[df["date_added"] >= pd.to_datetime(value)]
        elif key in ("date_added_max", "date_to"):
            return df[df["date_added"] <= pd.to_datetime(value)]
        elif key == "date_after":
            return df[df["date_added"] > pd.to_datetime(value)]
        elif key == "date_before":
            return df[df["date_added"] < pd.to_datetime(value)]
        elif key == "date_modified_min":
            return df[df["date_modified"] >= pd.to_datetime(value)]
        elif key == "date_modified_max":
            return df[df["date_modified"] <= pd.to_datetime(value)]
        elif key == "brand":
            return df[df["brand"].str.contains(value, case=False, na=False)]
        elif key == "category_id":
            return df[df["category_id"] == value]
        elif key == "industry_id":
            return df[df["industry_id"] == value]
        elif key == "country_of_origin":
            return df[df["country_of_origin"].str.lower() == value.lower()]
        elif key == "oem_number":
            return df[df["oem_number"] == value]
        elif key == "product_number":
            return df[df["product_number"] == value]
        elif key == "name":
            return df[df["name"].str.contains(value, case=False, na=False)]
        elif key == "description":
            return df[df["description"].str.contains(value, case=False, na=False)]
        elif key == "hs_code":
            return df[df["hs_code"] == value]
        elif key == "unique_id":
            return df[df["unique_id"] == value]
        else:
            print(f"[DEBUG] Unknown filter key: {key}")
            return df

    # Apply AND block — each condition must hold
    if "and" in filters:
        for cond in filters["and"]:
            df_copy = apply_atomic_filter(df_copy, cond)
        print(f"[DEBUG] After AND filters: {len(df_copy)} rows")

    # Apply OR block — at least one must hold
    if "or" in filters and filters["or"]:
        or_dfs = [apply_atomic_filter(df.copy(), cond) for cond in filters["or"]]
        or_combined = pd.concat(or_dfs).drop_duplicates()
        df_copy = df_copy[df_copy.index.isin(or_combined.index)]
        print(f"[DEBUG] After OR filters: {len(df_copy)} rows")

    # Apply NOT block — remove any matches
    if "not" in filters and filters["not"]:
        not_dfs = [apply_atomic_filter(df.copy(), cond) for cond in filters["not"]]
        not_combined = pd.concat(not_dfs).drop_duplicates()
        df_copy = df_copy[~df_copy.index.isin(not_combined.index)]
        print(f"[DEBUG] After NOT filters: {len(df_copy)} rows")

    if df_copy.empty:
        print("[DEBUG] No rows matched the filter conditions.")
    else:
        print(f"[DEBUG] Final result: {len(df_copy)} rows matched the filters.")

    return df_copy





# In[254]:
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Union, List, Dict, Callable, Any


def apply_analyze_data(
    df: pd.DataFrame,
    group_by_column: Optional[Union[str, List[str]]] = None,
    agg_dict: Optional[Dict[str, Union[str, List[Union[str, Callable]]]]] = None,
    count_column: Optional[Union[str, List[str]]] = None,
    include_group_size: bool = False,
    convert_numeric: bool = True,
    dropna_group_keys: bool = False,
    flatten_column_names: bool = True
) -> pd.DataFrame:
    """
    Analyze and summarize a DataFrame by performing flexible aggregations, value counts,
    and group-based summaries in a unified and configurable way.

    This function serves as a core engine for data summarization. It supports:
    - Standard aggregations on specific columns (via `agg_dict`)
    - Value counts for one or more categorical columns (via `count_column`)
    - Group-based aggregation if `group_by_column` is provided
    - Raw frequency counts (group size) when no aggregations are specified
    - Flattening of MultiIndex column names for readability

    ----------
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to analyze.

    group_by_column : str, optional
        If provided, the DataFrame is grouped by this column before applying aggregations or counts.
        This can be a datetime-derived column like "month_label" or a categorical field like "category".
        If None, all operations are done on the entire DataFrame without grouping.

    columns : List[str], optional
        The list of columns to include for aggregation if `agg_dict` is not provided.
        If both `columns` and `agg_dict` are None, but `group_by_column` is provided,
        the function defaults to returning group sizes.

    agg_dict : dict, optional
        A dictionary mapping column names to aggregation functions or list of functions.
        Example: {'price': 'mean', 'quantity': ['sum', 'max']}
        If this is given, aggregation is performed using pandas `.agg()`.

    count_column : str or List[str], optional
        One or more column names for which value counts are computed *within each group* (if grouped),
        or across the entire DataFrame (if not grouped).
        For multiple columns, separate value count DataFrames will be computed and merged with
        other results if applicable.

    include_group_size : bool, default False
        If True and `group_by_column` is provided, an additional column `'group_size'` is added
        showing the count of rows per group (i.e., group size via `.size()`).

    convert_numeric : bool, default False
        If True, all numeric-like string columns in the DataFrame are auto-converted to numeric.
        This is useful before applying aggregations to ensure correct type inference.

    dropna_group_keys : bool, default True
        If True, groups with NaN values in the `group_by_column` are dropped during grouping.
        If False, NaN is treated as a valid group key.

    flatten_column_names : bool, default True
        If True, any MultiIndex columns resulting from `.agg()` or `.value_counts()` are flattened
        to single-level strings using an underscore convention (e.g., 'price_mean').
        If False, column names may remain as tuples or MultiIndex objects.

    ----------
    Returns
    ----------
    pandas.DataFrame
        A summary DataFrame based on the input configuration. The structure varies:
        - If `agg_dict` is provided, returns aggregated metrics.
        - If `count_column` is provided, returns value counts per group or globally.
        - If both are given, returns a merged result with aggregations and value counts.
        - If neither is given but `group_by_column` is specified, returns basic group size summary.
        - If no `group_by_column`, returns either global aggregations or counts.

    ----------
    Raises
    ----------
    ValueError
        - If both `agg_dict` and `columns` are None and no `group_by_column` is given.
        - If a specified `count_column` or aggregation column does not exist in `df`.

    Notes
    -----
    - The order of operations is: optional numeric conversion → grouping → aggregation and/or counting.
    - Value counts are computed using `.value_counts()` internally per group or globally.
    - Aggregated results and value counts are joined on the group key if both are enabled.
    - When flattening column names, the pattern used is `"{column}_{agg_func}"` for aggregations.

    Examples
    --------
    >>> analyze_data(df, group_by_column='region', agg_dict={'sales': 'sum', 'profit': 'mean'})

    >>> analyze_data(df, group_by_column='category', count_column='sub_category')

    >>> analyze_data(df, group_by_column='month_label', columns=['price'], include_group_size=True)

    >>> analyze_data(df, count_column=['status', 'payment_method'])

    >>> analyze_data(df, agg_dict={'score': ['mean', 'max']}, flatten_column_names=False)
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        warnings.warn("Input DataFrame is empty.")
        return pd.DataFrame()

    if agg_dict is not None and count_column is not None:
        raise ValueError("Cannot specify both 'agg_dict' and 'count_column'. Choose one.")

    if convert_numeric:
        df = _convert_numeric_strings(df)

    if group_by_column is not None:
        return _grouped_analysis(
            df,
            group_by_column,
            agg_dict,
            count_column,
            include_group_size,
            dropna_group_keys,
            flatten_column_names
        )
    else:
        return _ungrouped_analysis(df, agg_dict, count_column)


def _grouped_analysis(
    df: pd.DataFrame,
    group_by_column: Union[str, List[str]],
    agg_dict: Optional[Dict[str, Union[str, List[Union[str, Callable]]]]],
    count_column: Optional[Union[str, List[str]]],
    include_group_size: bool,
    dropna_group_keys: bool,
    flatten_column_names: bool
) -> pd.DataFrame:
    if isinstance(group_by_column, str):
        group_by_column = [group_by_column]

    for col in group_by_column:
        if col not in df.columns:
            raise ValueError(f"Grouping column '{col}' not found in DataFrame.")

    if df[group_by_column].isnull().any().any():
        if dropna_group_keys:
            df = df.dropna(subset=group_by_column)
            if df.empty:
                warnings.warn("No data left after dropping missing group values.")
                return pd.DataFrame()
        else:
            warnings.warn("Grouping columns contain missing values. These will appear as NaN group keys.")

    grouped = df.groupby(group_by_column, dropna=not dropna_group_keys)

    if agg_dict:
        result = _apply_aggregations(grouped, agg_dict)
        if flatten_column_names:
            result = _flatten_columns(result)
    elif count_column:
        result = _apply_value_counts(grouped, count_column, group_by_column)
    else:
        return grouped.head(100)  # Fallback: return first rows of groups

    if include_group_size and not result.empty:
        sizes = grouped.size().reset_index(name='group_size')
        result = pd.merge(result, sizes, on=group_by_column, how='left')

    return result


def _ungrouped_analysis(df: pd.DataFrame,
                        agg_dict: Optional[Dict[str, Union[str, List[Union[str, Callable]]]]],
                        count_column: Optional[Union[str, List[str]]]) -> pd.DataFrame:
    if agg_dict:
        return _apply_ungrouped_aggregations(df, agg_dict)

    elif count_column:
        return _apply_ungrouped_value_counts(df, count_column)

    else:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            warnings.warn("No numeric columns found for descriptive statistics.")
            return pd.DataFrame({'message': ['No numeric columns available.']})
        return numeric.describe().T.reset_index().rename(columns={'index': 'column'})


def _convert_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        try:
            stripped = df[col].astype(str).str.strip()
            converted = pd.to_numeric(stripped, errors='coerce')
            if converted.notna().sum() > len(df) * 0.5:
                df[col] = converted
                warnings.warn(f"Converted column '{col}' to numeric.")
        except Exception:
            continue
    return df


def _apply_aggregations(grouped, agg_dict: Dict[str, Union[str, List[Union[str, Callable]]]]) -> pd.DataFrame:
    _validate_agg_dict(agg_dict)
    return grouped.agg(agg_dict).reset_index()


def _apply_ungrouped_aggregations(df: pd.DataFrame,
                                   agg_dict: Dict[str, Union[str, List[Union[str, Callable]]]]) -> pd.DataFrame:
    _validate_agg_dict(agg_dict)
    result = {}
    for col, funcs in agg_dict.items():
        if not isinstance(funcs, list):
            funcs = [funcs]
        for func in funcs:
            try:
                label = f"{col}_{func.__name__}" if callable(func) else f"{col}_{func}"
                result[label] = getattr(df[col], func)() if isinstance(func, str) else func(df[col])
            except Exception:
                warnings.warn(f"Failed to apply aggregation '{func}' to column '{col}'.")
    return pd.DataFrame([result])


def _apply_value_counts(grouped, count_column: Union[str, List[str]], group_by_column: List[str]) -> pd.DataFrame:
    count_column = [count_column] if isinstance(count_column, str) else count_column
    frames = []
    for col in count_column:
        if col not in grouped.obj.columns:
            raise ValueError(f"Count column '{col}' not found in DataFrame.")
        counts = grouped[col].value_counts().reset_index(name='count')
        counts.columns = group_by_column + [col, f'{col}_count']
        frames.append(counts)
    return _merge_multiple_value_counts(frames, group_by_column)


def _apply_ungrouped_value_counts(df: pd.DataFrame, count_column: Union[str, List[str]]) -> pd.DataFrame:
    count_column = [count_column] if isinstance(count_column, str) else count_column
    results = []
    for col in count_column:
        if col not in df.columns:
            raise ValueError(f"Count column '{col}' not found in DataFrame.")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, 'count']
        counts['source_column'] = col
        results.append(counts)
    return pd.concat(results, ignore_index=True)


def _merge_multiple_value_counts(frames: List[pd.DataFrame], group_by_column: List[str]) -> pd.DataFrame:
    from functools import reduce
    def merge_two(left, right):
        return pd.merge(left, right, on=group_by_column, how='outer')
    return reduce(merge_two, frames)


def _validate_agg_dict(agg_dict: Dict[str, Any]) -> None:
    valid_funcs = {'mean', 'median', 'sum', 'count', 'std', 'var', 'min', 'max', 'first', 'last', 'nunique'}
    for col, funcs in agg_dict.items():
        if isinstance(funcs, str):
            funcs = [funcs]
        elif not isinstance(funcs, list):
            funcs = [funcs]
        for func in funcs:
            if isinstance(func, str) and func not in valid_funcs:
                raise ValueError(f"Invalid aggregation function: '{func}'.")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns if needed by joining levels with underscore."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns.values]
    return df





# In[255]:


import pandas as pd
from typing import Union, List

def apply_get_unique_values(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    dropna: bool = False,
    as_list: bool = False
) -> pd.DataFrame:
    """
    Retrieve the unique values from one or more columns in a pandas DataFrame
    and return them in a structured, uniform DataFrame format suitable for
    human review, programmatic access, and LLM-based processing.

    This function ensures consistent tabular formatting regardless of how many
    columns are analyzed. It also provides optional features like dropping NaN values
    and packaging the results as Python lists.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataset to process. Must be a valid, non-empty pandas DataFrame.
        Each specified column must exist in this DataFrame.

    columns : str or list of str
        The column or columns from which to extract unique values.
        - If a string is provided, it is interpreted as one column name.
        - If a list of strings is provided, each will be processed independently.

    dropna : bool, optional (default=False)
        Whether to exclude missing values (NaNs) from the list of unique values.
        - True: NaNs are removed from the output.
        - False: NaNs are preserved if present.

    as_list : bool, optional (default=False)
        Controls the format of the returned DataFrame.
        - If True: Each column in the result contains a single list of unique values.
        - If False: Each column has one unique value per row, aligned row-wise across columns.
          Shorter columns are padded with NaNs to ensure alignment.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the unique values from each requested column.
        The structure depends on `as_list`:
        - If `as_list=True`: Each column will contain a single Python list as its only value.
        - If `as_list=False`: The DataFrame will contain one row per unique value. Columns with
          fewer values are padded with NaN to match the longest.

    Raises
    ------
    TypeError
        - If `df` is not a pandas DataFrame.
        - If `columns` is not a string or a list of strings.

    ValueError
        - If `df` is empty.
        - If any specified column is not present in the DataFrame.

    Examples
    --------
    >>> get_unique_values(df, "brand")
          brand
    0     Sony
    1       LG
    2  Panasonic

    >>> get_unique_values(df, ["brand", "country_of_origin"], as_list=True)
                           brand     country_of_origin
    0     [Sony, LG, Panasonic]   [Japan, Korea, China]

    >>> get_unique_values(df, ["brand"], dropna=True, as_list=False)
          brand
    0     Sony
    1       LG
    2  Panasonic

    Notes
    -----
    - The output is always a DataFrame regardless of how many columns are processed.
    - This function is commonly used in reporting pipelines, UI dropdown construction,
      faceted search filters, or metadata introspection tasks.
    - Padding with NaN (when `as_list=False`) ensures all output columns align correctly.
    - Does not sort unique values; preserves the order from the original DataFrame.
    """


    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise TypeError("'columns' must be a string or list of strings.")

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    result_dict = {}

    for col in columns:
        col = col.strip()
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        uniques = pd.Series(df[col].unique())

        if dropna:
            uniques = uniques.dropna()

        result_dict[col] = [uniques.tolist()] if as_list else uniques.reset_index(drop=True)

    # Assemble the final DataFrame
    if as_list:
        return pd.DataFrame(result_dict)
    else:
        # Pad shorter columns with NaNs so they align row-wise
        max_len = max(len(col) for col in result_dict.values())
        aligned = {
            col: pd.Series(vals if isinstance(vals, list) else vals).reindex(range(max_len))
            for col, vals in result_dict.items()
        }
        return pd.DataFrame(aligned)






# In[256]:


import pandas as pd
from typing import Union, List

def apply_sort_columns(
    df: pd.DataFrame,
    columns: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
    na_position: str = 'last',
    ignore_index: bool = False
) -> pd.DataFrame:
    """
    Sort a pandas DataFrame by one or more columns and return the sorted DataFrame.

    This function takes a DataFrame and returns a new DataFrame sorted by one or more
    specified columns. It does **not** modify the original DataFrame. The user must 
    assign the return value to capture the sorted result.

    This is designed for use in LLM pipelines, data transformation utilities,
    or interactive analysis sessions where predictable return types and clear logic 
    are essential.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sorted. It is not modified.

    columns : str or List[str]
        One or more column names by which to sort the DataFrame. Accepts:
        - A single string: e.g. `'age'`
        - A list of strings: e.g. `['city', 'income']`

    ascending : bool or List[bool], default True
        Sort order:
        - If a single `bool`, applies to all columns (e.g. `True` = ascending).
        - If a list, must match the number of columns, specifying the sort order 
          for each corresponding column.

    na_position : {'first', 'last'}, default 'last'
        Where to position NaN values during the sort:
        - `'first'`: NaNs appear at the top.
        - `'last'`: NaNs appear at the bottom.

    ignore_index : bool, default False
        If True, the index of the resulting DataFrame will be reset to a new 
        integer range index (0, 1, ..., n-1). If False, the original index is preserved.

    Returns
    -------
    pd.DataFrame
        A new DataFrame sorted by the specified column(s), with NaNs positioned as configured,
        and the index optionally reset.

    Raises
    ------
    TypeError
        - If `df` is not a pandas DataFrame.
        - If `columns` is not a string or list of strings.
        - If `ascending` is not a bool or list of bools.
        - If a list is provided for `ascending` and its length does not match `columns`.

    ValueError
        - If any column specified in `columns` is not present in the DataFrame.

    Examples
    --------
    >>> df = pd.DataFrame({'age': [25, 30, 22], 'name': ['Alice', 'Bob', 'Charlie']})
    >>> sort_columns(df, columns='age', ascending=True)

    >>> sort_columns(df, columns=['name', 'age'], ascending=[True, False])

    Notes
    -----
    - This function always returns a new DataFrame. The original `df` remains unchanged.
    - Column names are stripped of whitespace internally to avoid errors due to formatting.
    - Designed for use in both human and LLM-based workflows that require clarity,
      predictability, and safe handling of tabular data.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("'df' must be a pandas DataFrame.")

    if isinstance(columns, str):
        columns = [columns]
    elif not isinstance(columns, list) or not all(isinstance(col, str) for col in columns):
        raise TypeError("'columns' must be a string or list of strings.")

    for col in columns:
        if col.strip() not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    if isinstance(ascending, bool):
        ascending = [ascending] * len(columns)
    elif isinstance(ascending, list):
        if len(ascending) != len(columns):
            raise ValueError("'ascending' list must match the number of columns.")
        if not all(isinstance(a, bool) for a in ascending):
            raise TypeError("All elements in 'ascending' list must be boolean.")
    else:
        raise TypeError("'ascending' must be a boolean or a list of booleans.")

    return df.sort_values(
        by=[col.strip() for col in columns],
        ascending=ascending,
        na_position=na_position,
        ignore_index=ignore_index
    )

# In[257]:


def find_missing_values(df):
    """
    Comprehensive function to detect missing values in a DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame

    Returns:
    dict: Dictionary containing:
        - 'summary': DataFrame with missing value statistics per column
        - 'locations': DataFrame with row/column locations of missing values
        - 'total_missing': Total count of missing values
        - 'missing_percentage': Overall percentage of missing values
    """

    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if df.empty:
        return {
            'summary': pd.DataFrame(),
            'locations': pd.DataFrame(),
            'total_missing': 0,
            'missing_percentage': 0.0
        }

    # Create a copy to avoid modifying original
    df_copy = df.copy()

    # Replace common placeholder patterns with NaN
    placeholder_patterns = [
        '#########',  # Your specific pattern
        '######### ',
        '##############',  # Longer pattern for dates
        'N/A',
        'n/a',
        'NA',
        'na',
        'NULL',
        'null',
        'Null',
        'None',
        'none',
        'NONE',
        '',
        ' ',
        '  ',
        '   '
    ]

    for pattern in placeholder_patterns:
        df_copy = df_copy.replace(pattern, np.nan)

    # Also handle whitespace-only strings
    df_copy = df_copy.replace(r'^\s*$', np.nan, regex=True)

    # Calculate missing values per column
    missing_counts = df_copy.isnull().sum()
    missing_percentages = (missing_counts / len(df_copy)) * 100

    # Create summary DataFrame
    summary = pd.DataFrame({
        'Column': df_copy.columns,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentages.values,
        'Total_Rows': len(df_copy),
        'Non_Missing_Count': len(df_copy) - missing_counts.values
    })

    # Sort by missing count (descending)
    summary = summary.sort_values('Missing_Count', ascending=False)

    # Find locations of missing values
    locations = []
    for col in df_copy.columns:
        missing_indices = df_copy[df_copy[col].isnull()].index.tolist()
        for idx in missing_indices:
            locations.append({
                'Row_Index': idx,
                'Column': col,
                'Original_Value': df.iloc[idx][col] if idx < len(df) else None
            })

    locations_df = pd.DataFrame(locations)

    # Calculate overall statistics
    total_cells = df_copy.shape[0] * df_copy.shape[1]
    total_missing = missing_counts.sum()
    overall_missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0

    return {
        'summary': summary,
        'locations': locations_df,
        'total_missing': int(total_missing),
        'missing_percentage': round(overall_missing_percentage, 2),
        'total_cells': total_cells,
        'columns_with_missing': summary[summary['Missing_Count'] > 0]['Column'].tolist(),
        'columns_without_missing': summary[summary['Missing_Count'] == 0]['Column'].tolist()
    }

def print_missing_report(missing_info):
    """
    Print a formatted report of missing values.

    Parameters:
    missing_info (dict): Output from find_missing_values function
    """

    print("="*60)
    print("MISSING VALUES ANALYSIS REPORT")
    print("="*60)

    print(f"\nOVERALL STATISTICS:")
    print(f"Total cells: {missing_info['total_cells']:,}")
    print(f"Total missing values: {missing_info['total_missing']:,}")
    print(f"Overall missing percentage: {missing_info['missing_percentage']}%")

    print(f"\nCOLUMNS WITH MISSING VALUES: {len(missing_info['columns_with_missing'])}")
    print(f"COLUMNS WITHOUT MISSING VALUES: {len(missing_info['columns_without_missing'])}")

    if not missing_info['summary'].empty:
        print(f"\nMISSING VALUES BY COLUMN:")
        print(missing_info['summary'].to_string(index=False))

    if not missing_info['locations'].empty:
        print(f"\nFIRST 10 MISSING VALUE LOCATIONS:")
        print(missing_info['locations'].head(10).to_string(index=False))

    print("="*60)


# In[258]:


def summarize_by_period(
    df: pd.DataFrame,
    date_col: str,
    freq: str,
    **kwargs
) -> pd.DataFrame:
    """
    Adds a derived period-based column to the DataFrame (based on a datetime column)
    and applies `apply_analyze_data` grouped by that period.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the date column.
    date_col : str
        The name of the datetime column to use for period extraction.
    freq : str
        One of 'day', 'week', 'month', 'quarter', 'year'.
    **kwargs
        All other keyword arguments are passed to `apply_analyze_data`.

    Returns
    -------
    pd.DataFrame
        The result of `apply_analyze_data` grouped by the extracted time period.
    """
    df = df.copy()
    if date_col not in df.columns:
        raise ValueError(f"'{date_col}' not found in DataFrame.")

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    if freq == 'day':
        df['day_name'] = df[date_col].dt.day_name()
        group_col = 'day_name'

    elif freq == 'week':
        df['week_number'] = df[date_col].dt.isocalendar().week
        df['year'] = df[date_col].dt.isocalendar().year
        df['week_label'] = df['year'].astype(str) + '-W' + df['week_number'].astype(str)
        group_col = 'week_label'

    elif freq == 'month':
        df['month'] = df[date_col].dt.month_name()
        df['year'] = df[date_col].dt.year
        df['month_label'] = df['year'].astype(str) + '-' + df['month']
        group_col = 'month_label'

    elif freq == 'quarter':
        df['quarter_label'] = df[date_col].dt.to_period('Q').astype(str)
        group_col = 'quarter_label'

    elif freq == 'year':
        df['year_label'] = df[date_col].dt.year
        group_col = 'year_label'

    else:
        raise ValueError("Frequency must be one of: 'day', 'week', 'month', 'quarter', 'year'")

    return apply_analyze_data(df, group_by_column=group_col, **kwargs)
