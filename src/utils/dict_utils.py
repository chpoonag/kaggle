from tabulate import tabulate
from prettytable import PrettyTable
from copy import deepcopy

def print_dict_in_table(data, max_line_length=100):
    """
    Print a dictionary as a table with PrettyTable.

    Parameters:
    data (dict): The dictionary to print.
    max_line_length (int): The maximum line length for values.

    Returns:
    PrettyTable: The PrettyTable object with the dictionary data.
    """
    # Create a PrettyTable object
    table = PrettyTable(["Key", "Value"])

    # Add rows to the table
    for key, value in data.items():
        if isinstance(value, list):
            value_str = ', '.join(str(v) for v in value)
            value_lines = [value_str[i:i+max_line_length] for i in range(0, len(value_str), max_line_length)]
            value = '\n'.join(value_lines)
        elif len(str(value)) > max_line_length:
            value = '\n'.join(str(value)[i:i+max_line_length] for i in range(0, len(str(value)), max_line_length))
        table.add_row([key, value])

    # Set alignment of columns
    table.align["Key"] = "r"  # Left align the 'Key' column
    table.align["Value"] = "l"  # Left align the 'Value' column
    print(table)
    return table


# def print_dict_as_table(data, headers=None):
#     """
#     Prints a dictionary (from df.to_dict()) as a pretty table.

#     Parameters:
#         data (dict): The dictionary to print (output of df.to_dict()).
#         headers (list): Column headers for the table. If None, headers are inferred.
#     """
#     # Determine the orientation of the dictionary
#     if isinstance(data, dict) and all(isinstance(value, dict) for value in data.values()):
#         # Case 1: df.to_dict(orient='dict') or df.to_dict(orient='index')
#         table_data = [[key] + list(value.values()) for key, value in data.items()]
#         if headers is None:
#             headers = ["Index"] + list(next(iter(data.values())).keys())
#     elif isinstance(data, dict) and all(isinstance(value, list) for value in data.values()):
#         # Case 2: df.to_dict(orient='list')
#         table_data = list(zip(*data.values()))
#         if headers is None:
#             headers = list(data.keys())
#     elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
#         # Case 3: df.to_dict(orient='records')
#         table_data = [[value for value in item.values()] for item in data]
#         if headers is None:
#             headers = list(data[0].keys())
#     else:
#         raise ValueError("Unsupported dictionary format. Use df.to_dict() with a valid orient parameter.")

#     # Print the table
#     print(tabulate(table_data, headers=headers, tablefmt="pretty"))


def flatten_dict(d, parent_key='', sep='.', flat_dict=None):
    """
    Flatten a hierarchical dictionary to a single level while checking for overlapping keys.

    Parameters:
    d (dict): The input hierarchical dictionary.
    parent_key (str): The concatenated key from the parent dictionary.
    sep (str): The separator to use between keys.
    flat_dict (dict): The flattened dictionary (used for recursion).

    Returns:
    dict: The flattened dictionary.

    Raises:
    ValueError: If overlapping keys are encountered.
    """
    if flat_dict is None:
        flat_dict = {}

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if new_key in flat_dict:
            raise ValueError(f"Overlapping key found: {new_key}")

        if isinstance(v, dict):
            DictUtils.flatten_dict(v, new_key, sep, flat_dict)
        else:
            flat_dict[new_key] = v

    return flat_dict


def iterate_nested_dict(d, parent_key='', sep='.', res=None, ignore_parent_key=False):
    """
    Iterate through a nested dictionary and flatten it.

    Parameters:
    d (dict): The input nested dictionary.
    parent_key (str): The concatenated key from the parent dictionary.
    sep (str): The separator to use between keys.
    res (dict): The resulting flattened dictionary (used for recursion).
    ignore_parent_key (bool): Whether to ignore the parent key in the new key.

    Returns:
    dict: The flattened dictionary.

    Raises:
    ValueError: If overlapping keys are encountered.
    """
    if res is None:
        res = {}
    
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if new_key in res:
            raise ValueError(f"Overlapping key found: {new_key}")
        if isinstance(v, dict):
            DictUtils.iterate_nested_dict(
                deepcopy(v), 
                new_key if not(ignore_parent_key) else '', 
                sep, 
                res,
                ignore_parent_key
            )
        else:
            res[new_key] = v
    return res


def unflatten_dict(flat_dict, sep='.'):
    """
    Unflatten a dictionary that was flattened using a specific separator.

    Parameters:
    flat_dict (dict): The flattened dictionary.
    sep (str): The separator used between keys.

    Returns:
    dict: The unflattened dictionary.
    """
    unflat_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(sep)
        current_dict = unflat_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return unflat_dict
