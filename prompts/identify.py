"""
  Camera ready vs submission version:
  This module is newly added since we think tables from data lakes with no metadata
  available is a more natural setting.
"""

def identify_prompt():
    identify = """

Now you will be given a list of tables and columns, each one with the following format:
Analysis for Table <name of the table>:
  Column: <name of the column 1>
    Max: <max value of the column>
    Min: <min value of the column>
    Mode: <mode value of the column>
    Sampled Values: <list of sampled values>, for example, ['value1', 'value2', 'value3']
  Column: <name of the column 2>
    Max: <max value of the column>
    Min: <min value of the column>
    Mode: <mode value of the column>
    Sampled Values: <list of sampled values>, for example, ['value1', 'value2', 'value3']
...

You should identify the data type of each column. The data types you can choose from are:
['float', 'category', 'datetime', 'text', 'multi_category']
float: The column is probably a float-type embedding tensor. There should be (nearly) no redundant values.
category: The column is probably a categorical column.
datetime: The column is probably a datetime column. Only full datetime values should be considered, some columns presenting only year or month or day should be better considerd as category.
text: The column is probably a text column. There should be a lot of unique values. Otherwise it will probably be a category column. Moreover, we should expect texts with natural semantics, otherwise it's probably a category column.
multi_category: The column is probably a multi-category column. Usually this means the column value is a list. 
It should be noted that if the column is probably an embedding type, then directly put it to the float type.
Then, you should output a discription of the column, for example:
"This column is probably representing the ID from 1 to n of users in the system, as it has a lot of unique values."
Output the results with the following format:
{
    "<name of the table>": {
        "<name of the column 1>": ("<data type of the column 1>", "<description of the column 1>"),
        "<name of the column 2>": ("<data type of the column 2>", "<description of the column 2>")
    },
    ...
}

In description, if you see two columns are very similar and may represent the same thing, you should mention it.

    """

    return identify


def reflect():
    REFLECT = "Please double check to eliminate errors"

    return REFLECT