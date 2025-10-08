from typing import Any, Dict, List

import pandas as pd


class MarkdownUtil:
    @staticmethod
    def table_list(dict_list: List[Dict[str, str]]) -> str:
        """
        Creates a markdown table from a list of dictionaries.

        Args:
            dict_list: List of dictionaries where each dictionary has the same keys.
                       The keys of the first dictionary will be used as table headers.

        Returns:
            A string containing the markdown table representation.
        """
        if not dict_list:
            return ""

        # Extract headers from the first dictionary
        headers = list(dict_list[0].keys())

        if not headers:
            return ""

        # Create the header row
        header_row = "| " + " | ".join(headers) + " |"

        # Create the separator row
        separator_row = "| " + " | ".join(["---" for _ in headers]) + " |"

        # Create the data rows
        data_rows = []
        for data_dict in dict_list:
            row = (
                "| "
                + " | ".join([str(data_dict.get(header, "")) for header in headers])
                + " |"
            )
            data_rows.append(row)

        # Combine all rows to form the table
        table = "\n".join([header_row, separator_row] + data_rows)

        return table

    @staticmethod
    def table_dict(
        data_dict: Dict[Any, Any],
        key_heading: str = "Key",
        value_heading: str = "Value",
    ) -> str:
        """
        Creates a markdown table from a dictionary.

        Args:
            data_dict: Dictionary to convert to a table.
            key_heading: The heading for the keys column (default: "Key").
            value_heading: The heading for the values column (default: "Value").

        Returns:
            A string containing the markdown table representation.
        """
        if not data_dict:
            return ""

        # Create the header row
        header_row = f"| {key_heading} | {value_heading} |"

        # Create the separator row
        separator_row = "| --- | --- |"

        # Create the data rows
        data_rows = []
        for key, value in data_dict.items():
            row = f"| {str(key)} | {str(value)} |"
            data_rows.append(row)

        # Combine all rows to form the table
        table = "\n".join([header_row, separator_row] + data_rows)

        return table

    @staticmethod
    def table_df(df: pd.DataFrame) -> str:
        """
        Creates a markdown table from a pandas DataFrame.

        Args:
            df: The pandas DataFrame to convert to a markdown table.

        Returns:
            A string containing the markdown table representation.
        """
        if df.empty:
            return ""

        # Create the header row
        headers = df.columns.tolist()
        header_row = "| " + " | ".join(str(header) for header in headers) + " |"

        # Create the separator row
        separator_row = "| " + " | ".join(["---" for _ in headers]) + " |"

        # Create the data rows
        data_rows = []
        for _, row in df.iterrows():
            data_row = "| " + " | ".join(str(value) for value in row.values) + " |"
            data_rows.append(data_row)

        # Combine all rows to form the table
        table = "\n".join([header_row, separator_row] + data_rows)

        return table
