"""Functions for aggregating and partitioning SOEP data based on YAML configuration."""

import pandas as pd


def aggregate_and_partition_soep(df, agg_config, partition_config=None):
    """Main function to aggregate and partition SOEP data.

    Args:
        df: Input DataFrame (cleaned SOEP data)
        agg_config: Aggregation configuration dictionary
        partition_config: Partition configuration dictionary (optional)

    Returns:
        pd.DataFrame: Aggregated (and optionally partitioned) DataFrame
    """
    # Apply pre-aggregation filters
    pre_filters = agg_config.get("pre_aggregation_filters", {})
    filtered_df = apply_pre_aggregation_filters(df, pre_filters)
    # Perform aggregation
    aggregated_df = perform_aggregation(filtered_df, agg_config)

    # Add derived columns
    derived_cols = agg_config.get("derived_columns", {})
    aggregated_df = add_derived_columns(aggregated_df, derived_cols)

    # Compute standard deviations from variances
    aggregated_df = compute_std_from_var(aggregated_df)

    # Apply partition filters if config provided
    if partition_config:
        final_df = apply_partition_filters(aggregated_df, partition_config)
    else:
        final_df = aggregated_df
    return final_df


def apply_pre_aggregation_filters(df, filters):
    """Apply filters before aggregation.

    Args:
        df: Input DataFrame
        filters: Dictionary of filter specifications from config

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()

    # Year range filter
    if "syear" in filters:
        if "min" in filters["syear"] and filters["syear"]["min"] is not None:
            filtered_df = filtered_df[filtered_df["syear"] >= filters["syear"]["min"]]
        if "max" in filters["syear"] and filters["syear"]["max"] is not None:
            filtered_df = filtered_df[filtered_df["syear"] <= filters["syear"]["max"]]

    # Exclude specific values
    if "exclude_values" in filters:
        for column, values_to_exclude in filters["exclude_values"].items():
            if column in filtered_df.columns:
                filtered_df = filtered_df[~filtered_df[column].isin(values_to_exclude)]

    # Require at least one non-null value among specified columns
    if "require_any_not_null" in filters:
        columns = filters["require_any_not_null"]
        mask = pd.Series(data=False, index=filtered_df.index)
        for col in columns:
            if col in filtered_df.columns:
                mask |= filtered_df[col].notna()
        filtered_df = filtered_df[mask]

    return filtered_df


def perform_aggregation(df, config):
    """Perform groupby aggregation based on config.

    Args:
        df: Input DataFrame
        config: Configuration dictionary

    Returns:
        pd.DataFrame: Aggregated DataFrame
    """
    group_by = config["group_by"]
    aggregations = config["aggregations"]
    count_column = config["count_column"]
    options = config.get("options", {})

    # Build aggregation dictionary
    agg_dict = {}
    for variable, functions in aggregations.items():
        if variable not in df.columns:
            continue
        for func in functions:
            agg_dict[f"{variable}_{func}"] = (variable, func)

    # Add count
    agg_dict["n_obs"] = (count_column, "count")

    # Perform groupby
    observed_only = options.get("observed_only", True)
    result_df = df.groupby(group_by, observed=observed_only).agg(**agg_dict)

    # Reset index if specified
    if options.get("reset_index", True):
        result_df = result_df.reset_index()

    return result_df


def add_derived_columns(df, derived_columns):
    """Add computed columns after aggregation.

    Args:
        df: Aggregated DataFrame
        derived_columns: Dictionary of derived column specifications

    Returns:
        pd.DataFrame: DataFrame with derived columns added
    """
    if not derived_columns:
        return df

    result_df = df.copy()

    for col_name, col_spec in derived_columns.items():
        formula = col_spec["formula"]
        result_df[col_name] = result_df.eval(formula)

    return result_df


def compute_std_from_var(df):
    """Compute standard deviation columns from variance columns.

    Args:
        df: DataFrame with variance columns

    Returns:
        pd.DataFrame: DataFrame with added std columns
    """
    result_df = df.copy()

    var_columns = [col for col in df.columns if col.endswith("_var")]
    for var_col in var_columns:
        std_col = var_col.replace("_var", "_std")
        result_df[std_col] = result_df[var_col].pow(0.5)

    return result_df


def apply_partition_filters(df, partition_config):
    """Apply partition filters to aggregated data.

    Args:
        df: Aggregated DataFrame
        partition_config: Partition configuration dictionary

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()
    filters = partition_config.get("filters", {})

    # Apply column value filters
    for column, values in filters.items():
        # Skip if values is None or empty
        if values is None:
            continue

        if column == "n_obs_min":
            # Minimum observation threshold
            filtered_df = filtered_df[filtered_df["n_obs"] >= values]
        elif column == "n_obs_max":
            # Maximum observation threshold
            filtered_df = filtered_df[filtered_df["n_obs"] <= values]
        elif (
            isinstance(values, list)
            and len(values) > 0
            and column in filtered_df.columns
        ):
            # Only filter if list is not empty
            filtered_df = filtered_df[filtered_df[column].isin(values)]

    # Sort if specified
    sort_by = partition_config.get("sort_by")
    if sort_by and sort_by is not None:
        filtered_df = filtered_df.sort_values(by=sort_by)

    # Limit rows if specified
    max_rows = partition_config.get("max_rows")
    if max_rows is not None:
        filtered_df = filtered_df.head(max_rows)

    # Reset index if specified
    if partition_config.get("reset_index", True):
        filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df
