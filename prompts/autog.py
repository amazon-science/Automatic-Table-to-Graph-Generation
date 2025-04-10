"""
    First version of the prompt. 
    Not used in the paper.
"""

from typing import List, Dict
import json


def pack_example_str(examples: List[str]):
    example_str = "<example>\n"
    for i, e in enumerate(examples):
        this_str = f"<example {i}>\n {e} \n</example {i}>"
        example_str += this_str
    return example_str + "\n</example>\n"



def single_round_prompt(examples: List[str], input_data_schema: Dict, downstream_task_instruction: str = "", 
                        dataset_stats: str = "", dataset_meta_info: str = ""):
    """
        examples: a list of examples, each example is a string,
        input_data_schema: a dictionary representing the input data schema
        downstream_task_instruction: a string representing the downstream task instruction
        dataset_stats: statistics of different columns in the table
        dataset_meta_info: meta information of the table, like the meaning of each column
    """
    packed_example_str = pack_example_str(examples)
    json_str = json.dumps(input_data_schema, indent=2)
    AUGMENTATION_PROMPT = f"""
    Imagine you are an expert graph data scientist, and now you are expected to construct graph schema based on the original
    inputs. You will be given an original schema represented in the dictionary format:
    <data>
        You are given a series of tabular data, each of them containing some columns and represents
        one table. You will be given an original dataset schema with a dictionary format, which will contains the following information:
        1. dataset_name: name of the dataset 
        2. tables: meta data for list of tables, each one will present following attributes
            1. name: table name
            2. source: source of the data, can either be a numpy .npz file or a parquet file
            3. columns: list of columns, each column will have following attributes
                1. name: column name
                2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category.
                primary_key and foreign_key are two special types of categorical columns, which presents a structural
                relationship with other tables. Multi_category means this column is of list type, and each cell main contains
                a list of categorical values. After a column is set as primary_key or foreign_key, you shouldn't change
                the dtype of this column.
                3. link_to (optional): if this column is a foreign key, point to which primary key from which table
    </data>
    You can think of we are using a tabular-graph unified language, where finally each table should represent one node type with
    proper columns set as the corresponding features (this is not satisfied for the original input). The updated foreign key/primary key relationships will be viewed as the edge type,
    and finally we can easily transform this dictionary into a graph (not your task).     

    Then, your task is to identify the potential graph structures within these tabular data. You just need to do the 
    augmentations with the original dictionary format. 
    <task>
        Given the input tabular data schema, you need to identify the potential graph structures, they can be of 
        the following possibilities:
        1. The most obvious graph structure can be the primary key/foreign key relationship between two tables.
        For example, the primary key/foreign key relationship between 'View' and 'Product' tables. In this case, you 
        don't need to do change anything. 
        2. For columns with categorical values (or more generally, there are many duplications in column values), you can also consider constructing a relation based on the values
        inside those columns. For example, if there's a table with columns [{{"name": "ProductID", "dtype": "primary_key" 
        }}, {{"name": "name", "dtype": "text"}}, {{"name": "UserID", "dtype": "category"}}], in this case, the "UserID" column
        should be an independent type and you can create a dummy table for it, which results in [{{"name": "ProductID", "dtype": "primary_key" 
        }}, {{"name": "name", "dtype": "text"}}, {{"name": "UserID", "dtype": "foreign_key", "link_to": "User.ID"}}]. Of course, 
        not all categorical columns should be converted to a dummy table, and you should do decisions wisely. **When you create a dummy table with 
        only one primary_key column, don't add it to the output dictionary object, just declare it in the original table.**
        3. For columns with multi-cateogry values, you can also consider constructing a new tables based on the values.
        4. You need to identify missing foreign key/primary key relationships, for example, for two columns "User" and "UserID" 
        with both "category" dtypes, you should probably build a PK/FK relationship between them since they both represent
        the same entity.
        
        Criteria for graph construction:
        1. The constructed graph should be a valid graph structure (no grammar errors).
        2. The constructed graph should be helpful for the downstream tasks, which means you should construct a graph better
        reflecting proper relationships between proper node types. You should select proper columns as features and augment 
        new relations properly.
        3. Some common rule-of-thumbs when creating a new table is to consider the statistics of columns. For example, if a categorical
        column contains only two or three different values, then you should not create a new table out of it since it will introduce a supernode
        which is not helpful. On the other hand, if the relation is too sparse, you may also consider keeping it as original.
        4. When you generate the name of the new table, you should follow the naming convention. For example, when you try to 
        build a new table for a "multi-category" column "Keywords", the name of the augmented table should be "HasKeywords" to 
        indicate that it's a relationship table between "Paper" and "Keywords". For multi-category columns, if you want to expand them, 
        you should create a new table with one foreign_key pointing to the primary key of the original table, and a new column with categorical 
        values. 
        5. For simplicitly, you can assume there are at most two foreign keys in a table without primary key. For all tables, there are be 
        at most one primary key.
        
        After augmenting the schema, you should select on of the following algorithms to turn dict-format table schema to graphs, 
        the options are ["Row2Node", "Row2Node/Edge"]. 
        * Row2Node means each row will be converted to a node type
        * Row2Node/Edge means each row will be converted to either a node type or an edge type, if a table has no 
        primary key and two foreign keys, then it will be converted to an edge type.
        
    </task>

    {packed_example_str}

    You can assume that the input data is always valid and follows the format described above.
    <input> {json_str} </input>
    {f'''<downstream> 
        {downstream_task_instruction}
    </downstream>''' if downstream_task_instruction != "" else ""}
    {f'''<dataset_stats>
        {dataset_stats}
    </dataset_stats>''' if dataset_stats != "" else ""}
    {f'''<dataset_meta_info>
        {dataset_meta_info}
    </dataset_meta_info>''' if dataset_meta_info != "" else ""}
    
    You should structure your output with the following format
    <output>

    <schema>
        {{The output graph schema, which should be a python dictionary object as input.}}
    </schema>
    
    <algorithm>
        {{Row2Node or Row2Node/Edge}}
    </algorithm>

    <reason>
        {{Your reason for choosing this schema over other possible schemas. Also, why you choose Row2Node or Row2Node-Edge?}}
    </reason>

    </output>
    """
    
    return AUGMENTATION_PROMPT





EXAMPLE_INPUT_1="""
An example will be as follows:
    <input>
    <dataset_stats>
        Table: View
{
  "Column": "UserID",
  "data type": "category",
  "Number of unique values": 8932,
  "Number of nan values": 0,
  "Number of total values": 97422,
  "Mode values": 414,
  "5 sampled values": [
    329,
    414,
    378,
    421,
    521
  ]
}
{
  "Column": "ItemID",
  "data type": "foreign_key"
}
Table: Purchase
{
  "Column": "UserID",
  "data type": "category",
  "Number of unique values": 10245,
  "Number of nan values": 0,
  "Number of total values": 137422,
  "Mode values": 414,
  "5 sampled values": [
    329,
    414,
    378,
    421,
    521
  ]
}
{
  "Column": "ItemID",
  "data type": "foreign_key"
}
Table: Product
{
  "Column": "ItemID",
  "data type": "primary_key"
}
{
    "Column": "Price",
    "data type": "float",
}
{
    "Column": "Category",
    "data type": "category",
    "Number of unique values": 10,
  "Number of nan values": 0,
  "Number of total values": 128564,
  "Mode values": 3,
  "5 sampled values": [
    3,
    4,
    1,
    6,
    9
  ]
    
}

    </dataset_stats>
    <schema>
    {
        "dataset_name": "Sales",
        "tables": [
            {
                "name": "View",
                "source": "data/view.npz",
                "columns": [
                    {"name": "UserID", "dtype": "category"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            }, 
            {
                "name": "Purchase",
                "source": "data/purchase.npz",
                "columns": [
                    {"name": "UserID", "dtype": "category"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            },
            {
                "name": "Product",
                "source": "data/product.parquet",
                "columns": [
                    {"name": "ItemID", "dtype": "primary_key"},
                    {"name": "Price", "dtype": "float"},
                    {"name": "Category", "dtype": "category"}
                ]
            }
        ]
    }
    </schema>
    <tasks>
    Now I want to train a model which can predict the category of a product based on the information in the product.
    </tasks>
    </input>
    <output>
    <schema>
    {
        "dataset_name": "Sales",
        "tables": [
            {
                "name": "View",
                "source": "data/view.npz",
                "columns": [
                    {"name": "UserID", "dtype": "foreign_key", "link_to": "User.UserID"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            }, 
            {
                "name": "Purchase",
                "source": "data/purchase.npz",
                "columns": [
                    {"name": "UserID", "dtype": "foreign_key", "link_to": "User.UserID"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            },
            {
                "name": "Product",
                "source": "data/product.parquet",
                "columns": [
                    {"name": "ItemID", "dtype": "primary_key"},
                    {"name": "Price", "dtype": "float"},
                    {"name": "Category", "dtype": "category"}
                ]
            },
            {
                "name": "User",
                "source": "data/user.npz",
                "columns": [
                    {"name": "UserID", "dtype": "primary_key"}
                ]
            }
        ]
    }
    </schema>
    <algorithm>
    Row2Node/Edge
    </algorithm>
    <reason>
    1. "UserID" between Table "View" and "Purchase" is a missing foreign key/primary key relationship. At the same time, 
    UserID should be encoded in a separate table "User". From the table statistics, we can see that the ratio of unique values
    are appropirate, so we should form a new table for "User". As a result, we construct a dummy table user and establish 
    corresponding relationships between "View" and "Purchase" and "User". Since it's a dummy table, we don't need to explicitly add a new element to the "tables" list, 
    but only declare it in the "link_to" of the original table. 
    2. "Price" column is apparantly a feature column, so we don't need to generate a new table for it.
    3. "Category" in the "Product" table is the target column, we should not generate a new table for it.
    4. I choose to use the Row2Node/Edge algorithm because new purchase table represents a edge relationship.
    </reason>
    </output>
"""

STATS_FREE_EXAMPLE_INPUT_1 = """
An example will be as follows:
    <input>
    <schema>
    {
        "dataset_name": "Sales",
        "tables": [
            {
                "name": "View",
                "source": "data/view.npz",
                "columns": [
                    {"name": "UserID", "dtype": "category"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            }, 
            {
                "name": "Purchase",
                "source": "data/purchase.npz",
                "columns": [
                    {"name": "UserID", "dtype": "category"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            },
            {
                "name": "Product",
                "source": "data/product.parquet",
                "columns": [
                    {"name": "ItemID", "dtype": "primary_key"},
                    {"name": "Price", "dtype": "float"},
                    {"name": "Category", "dtype": "category"}
                ]
            }
        ]
    }
    </schema>
    <tasks>
    Now I want to train a model which can predict the category of a product based on the information in the product.
    </tasks>
    </input>
    <output>
    <schema>
    {
        "dataset_name": "Sales",
        "tables": [
            {
                "name": "View",
                "source": "data/view.npz",
                "columns": [
                    {"name": "UserID", "dtype": "foreign_key", "link_to": "User.UserID"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            }, 
            {
                "name": "Purchase",
                "source": "data/purchase.npz",
                "columns": [
                    {"name": "UserID", "dtype": "foreign_key", "link_to": "User.UserID"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            },
            {
                "name": "Product",
                "source": "data/product.parquet",
                "columns": [
                    {"name": "ItemID", "dtype": "primary_key"},
                    {"name": "Price", "dtype": "float"},
                    {"name": "Category", "dtype": "category"}
                ]
            },
            {
                "name": "User",
                "source": "data/user.npz",
                "columns": [
                    {"name": "UserID", "dtype": "primary_key"}
                ]
            }
        ]
    }
    </schema>
    <algorithm>
    Row2Node/Edge
    </algorithm>
    <reason>
    1. "UserID" between Table "View" and "Purchase" is a missing foreign key/primary key relationship. At the same time, 
    UserID should be encoded in a separate table "User". As a result, we construct a dummy table user and establish 
    corresponding relationships between "View" and "Purchase" and "User". Since it's a dummy table, we don't need to explicitly add a new element to the "tables" list, 
    but only declare it in the "link_to" of the original table. 
    2. "Price" column is apparantly a feature column, so we don't need to generate a new table for it.
    3. "Category" in the "Product" table is the target column, we should not generate a new table for it.
    4. I choose to use the Row2Node/Edge algorithm because new purchase table represents a edge relationship.
    </reason>
    </output>
"""


STATS_FREE_EXAMPLE_INPUT_2 = """
An example will be as follows:
<input>
<tasks>
Now I want to train a model which can predict the category of a paper based on the information in the paper.
</tasks>
<schema>
{
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.npz",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Authors", "dtype": "multi_category"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "category"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "Journal",
                "source": "data/journal.npz",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "category"},
                    {"name": "Publisher", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
        ]
    }
</schema>
</input>
<output>
<schema>
{
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.pqt",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "foreign_key", "link_to": "Keyword.KeywordID"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "HasAuthor",
                "source": "data/hasauthor.pqt",
                "columns": [
                    {"name": "AuthorID", "dtype": "foreign_key", "link_to": "Author.AuthorID"},
                    {"name": "PaperID", "dtype": "foreign_key", "link_to": "Paper.PaperID"},
                ]
            },
            {
                "name": "Journal",
                "source": "data/journal.pqt",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "foreign_key", "link_to": "Country.CountryID"}, 
                    {"name": "PublisherID", "dtype": "foreign_key", "link_to": "Publisher.PublisherID"}
                ]
            },
            {
                "name": "Publisher",
                "source": "data/publisher.pqt",
                "columns": [
                    {"name": "PublisherID", "dtype": "primary_key"},
                    {"name": "PublisherName", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
            
        ]
    }
</schema>
<algorithm>
    Row2Node/Edge
    </algorithm>
    <reason>
    To generate the graph schema for this example, I make the following changes:
    1. "Authors" in the original table is a multi_category column, which contains a list of authors, and it's 
    not easy to do representation learning for this format. To better capture the structural relationship, we 
    generate a new table "HasAuthor" to represent the relationship between authors and papers, which contains AuthorID
    and paperID as foreign keys. Please pay attention to the naming convention here, we use "Has" to indicate that 
    it's a relationship table between "Author" and "Paper". Moreover, you must remove the "Authors" column from the
    original table "Paper", otherwise it will break the primary key constraint of the original table.
    2. "Keywords" in the original table is a category column. "Keyword" is a different entity from "Paper", so we
    generate a new dummy table "Keyword" to represent the keywords. Since this is a dummy table with only one primary 
    key column, we don't explicitly add it to the schema, but only keeps it in the original table.
    3. "Country" in the original table is a category column. "Country" is a different entity from "Journal", so we
    generate a new dummy table "Country" to represent the countries. Since this is a dummy table with only one primary
    key column, we don't explicitly add it to the schema, but only keeps it in the original table.
    4. For attribute column with text type like "Name", "Title", "Abstract", we don't need to generate new tables for them. 
    Since they are already the attributes of the corresponding table.
    5. "Publisher" and "PublisherLocation" in the original table is a text and category column. "Publisher" is a different entity from "Journal", so we
    generate a new table with "Publisher" as the primary key. This is not a dummy table, so we need to explicitly add it to the schema.
    
    I choose to use the Row2Node/Edge algorithm because new "Writes" table represents a edge relationship.
    </reason>
</output>

"""

EXAMPLE_INPUT_2 = """
An example will be as follows:
<input>
<dataset_stats>
Table: Paper
{
  "Column": "PaperID",
  "data type": "primary_key"
}
{
    "Column": "Title",
    "data type": "text",
    "Number of unique values": 10000,
    "Number of nan values": 0,
    "Number of total values": 10000,
    "Mode values": "Transformers",
    "5 sampled values": [
        "Transformers",
        "Graph Neural Networks",
        "Reinforcement Learning",
        "Meta Learning",
        "Computer Vision"
    ]
}
{
    "Column": "Authors",
    "data type": "multi_category",
    "Number of unique values": 987,
    "Number of nan values": 0,
    "Number of total values": 74320,
    "Mode values": "Yann LeCun",
    "5 sampled values": [
        "Yann LeCun",
        "Geoffrey Hinton",
        "Yoshua Bengio",
        "Fei-Fei Li",
        "Jitendra Malik"
    ]
}
{
    "Column": "Journal",
    "data type": "foreign_key"
}
{
    "Column": "Year",
    "data type": "float",
}
{
    "Column": "Keywords",
    "data type": "category",
    "Number of unique values": 100,
    "Number of nan values": 0,
    "Number of total values": 10000,
    "Mode values": "Machine Learning",
    "5 sampled values": [
        "Machine Learning",
        "Deep Learning",
        "Graph Neural Networks",
        "Reinforcement Learning",
        "Meta Learning"
    ]
}
{
    "Column": "Abstract",
    "data type": "text",
    "Number of unique values": 10000,
    "Number of nan values": 0,
    "Number of total values": 10000,
    "Mode values": "This paper presents a new model for graph neural networks.",
    "5 sampled values": [
        "This paper presents a new model for graph neural networks.",
        "This paper introduces a new reinforcement learning algorithm.",
        "This paper presents a new model for transformers.",
        "This paper presents a new model for meta learning.",
        "This paper presents a new model for computer vision."
    ]
}
{
    "Column": "Category",
    "data type": "category",
    "Number of unique values": 10,
    "Number of nan values": 0,
    "Number of total values": 10000,
    "Mode values": 3,
    "5 sampled values": [
        3,
        4,
        1,
        6,
        9
    ]
}
{
  "Column": "ItemID",
  "data type": "foreign_key"
}
Table: Journal
{
  "Column": "JournalID",
  "data type": "primary_key"
}
{
  "Column": "Name",
  "data type": "text", 
    "Number of unique values": 100,
    "Number of nan values": 0,
    "Number of total values": 100,
    "Mode values": "Nature",
    "5 sampled values": [
        "Nature",
        "Science",
        "NeurIPS",
        "ICML",
        "CVPR"
    ]
}
{
    "Column": "ImpactFactor",
    "data type": "float"
}
{
    "Column": "Country",
    "data type": "category",
    "Number of unique values": 10,
    "Number of nan values": 0,
    "Number of total values": 100,
    "Mode values": "USA",
    "5 sampled values": [
        "USA",
        "USA",
        "Canada",
        "UK",
        "USA"
    ]
}
{
    "Column": "Publisher",
    "data type": "text",
    "Number of unique values": 9,
    "Number of nan values": 0,
    "Number of total values": 100,
    "Mode values": "Springer",
    "5 sampled values": [
        "Springer",
        "Elsevier",
        "ACM",
        "IEEE",
        "Nature"
    ]
}
{
    "Column": "PublisherLocation",
    "data type": "category",
    "Number of unique values": 5,
    "Number of nan values": 0,
    "Number of total values": 100,
    "Mode values": "USA",
    "5 sampled values": [
        "USA",
        "USA",
        "Canada",
        "UK",
        "USA"
    ]
}

</dataset_stats>
<tasks>
Now I want to train a model which can predict the category of a paper based on the information in the paper.
</tasks>
<schema>
{
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.npz",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Authors", "dtype": "multi_category"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "category"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "Journal",
                "source": "data/journal.npz",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "category"},
                    {"name": "Publisher", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
        ]
    }
</schema>
</input>
<output>
<schema>
{
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.pqt",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "foreign_key", "link_to": "Keyword.KeywordID"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "HasAuthor",
                "source": "data/hasauthor.pqt",
                "columns": [
                    {"name": "AuthorID", "dtype": "foreign_key", "link_to": "Author.AuthorID"},
                    {"name": "PaperID", "dtype": "foreign_key", "link_to": "Paper.PaperID"},
                ]
            },
            {
                "name": "Journal",
                "source": "data/journal.pqt",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "foreign_key", "link_to": "Country.CountryID"}, 
                    {"name": "PublisherID", "dtype": "foreign_key", "link_to": "Publisher.PublisherID"}
                ]
            },
            {
                "name": "Publisher",
                "source": "data/publisher.pqt",
                "columns": [
                    {"name": "PublisherID", "dtype": "primary_key"},
                    {"name": "PublisherName", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
            
        ]
    }
</schema>
<algorithm>
    Row2Node/Edge
    </algorithm>
    <reason>
    To generate the graph schema for this example, I make the following changes:
    1. "Authors" in the original table is a multi_category column, which contains a list of authors, and it's 
    not easy to do representation learning for this format. Moreover, from the statistics, we can induce that the
    ratio of unique values may result in a better structural relationship. To better capture the structural relationship, we 
    generate a new table "HasAuthor" to represent the relationship between authors and papers, which contains AuthorID
    and paperID as foreign keys. Please pay attention to the naming convention here, we use "Has" to indicate that 
    it's a relationship table between "Author" and "Paper". Moreover, you must remove the "Authors" column from the
    original table "Paper", otherwise it will break the primary key constraint of the original table.
    2. "Keywords" in the original table is a category column with proper unique value ratio. "Keyword" is a different entity from "Paper", so we
    generate a new dummy table "Keyword" to represent the keywords. Since this is a dummy table with only one primary 
    key column, we don't explicitly add it to the schema, but only keeps it in the original table.
    3. "Country" in the original table is a category column with proper unique value ratio. "Country" is a different entity from "Journal", so we
    generate a new dummy table "Country" to represent the countries. Since this is a dummy table with only one primary
    key column, we don't explicitly add it to the schema, but only keeps it in the original table.
    4. For attribute column with text type like "Name", "Title", "Abstract", we don't need to generate new tables for them. 
    Since they are already the attributes of the corresponding table.
    5. "Publisher" and "PublisherLocation" in the original table is a text and category column. "Publisher" is a different entity from "Journal", so we
    generate a new table with "Publisher" as the primary key. This is not a dummy table, so we need to explicitly add it to the schema.
    
    I choose to use the Row2Node/Edge algorithm because new "Writes" table represents a edge relationship.
    </reason>
</output>

"""


AUG_INPUT_1 = """
An example will be as follows:
    <input>
    <schema>
    {
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.npz",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Authors", "dtype": "multi_category"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "multi_category"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "category", "dtype": "category"}
                ]
            }, 
            {
                "name": "Journal",
                "source": "data/journal.npz",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "category"}
                ]
            }
        ]
    }
    </schema>
    <task>
        Now I want to train a model which can predict the category of a paper based on the information in the paper.
    </task>
    <input>
    <output>
    <decision> No augmentation </decision>
    <python_file></python_file>
    <python_schema></python_schema>
    <reason>
        There's no need to do augmentation to the tables. Since the category information is already in the Table "Paper".
    </reason>
    </output>

"""


AUG_INPUT_2 = """
An example will be as follows:
    <input>
    <schema>
    {
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.npz",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Authors", "dtype": "multi_category"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "multi_category"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"},
                    {"name": "Timestamp", "dtype": "datetime"}
                ], 
                "time_column": "Timestamp"
            }, 
            {
                "name": "Journal",
                "source": "data/journal.npz",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "category"}
                ]
            }
        ]
    }
    </schema>
    <task>
        Now I want to train a model which can predict the number of authors for a paper based on the information in the paper.
    </task>
    <input>
    <output>
    <decision> Augmentation </decision>
    <python>
    paper_value = np.load("data/paper.npz")
    paper_value['NumAuthors'] = [len(x) for x in paper_value['Authors']]
    np.savez("data/paper.npz", **paper_value)
    </python>
    <python_schema>
    ## identify the target table, which should be Paper table
    paper_table = [x for x in schema['tables'] if x['name'] == 'Paper'][0]
    ## augment the target table
    paper_table['columns'].append({"name": "NumAuthors", "dtype": "float"})
    </python_schema>
    <reason>
        The number of authors is not directly represented in the original table, so we need to generate a new column
        to represent the number of authors. We can do this by counting the number of authors in the "Authors" column.
    </reason>
    </output>

"""





def code_generation_prompt(dataset, meta_info, input_schema, output_schema):
    full_info = ""
    for tab in meta_info:
        full_info += f"Table: {tab['name']}, source: {tab['source']}, data type: {tab['format']}\n"
        for col in tab['columns']:
            full_info += f"\tColumn: {col['name']}, dtype: {col['dtype']}\n"
    EXAMPLE = """
    <example>
            <input schema>
                {
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.npz",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Authors", "dtype": "multi_category"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "category"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "Journal",
                "source": "data/journal.npz",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "category"},
                    {"name": "Publisher", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
        ]
    }
            </input schema>
            <output schema>
            {
        "dataset_name": "Papers",
        "tables": [
            {
                "name": "Paper",
                "source": "data/paper.pqt",
                "columns": [
                    {"name": "PaperID", "dtype": "primary_key"},
                    {"name": "Title", "dtype": "text"},
                    {"name": "Journal", "dtype": "foreign_key", "link_to": "Journal.JournalID"},
                    {"name": "Year", "dtype": "float"},
                    {"name": "Keywords", "dtype": "foreign_key", "link_to": "Keyword.KeywordID"},
                    {"name": "Abstract", "dtype": "text"},
                    {"name": "Category", "dtype": "category"}
                ]
            }, 
            {
                "name": "HasAuthor",
                "source": "data/hasauthor.pqt",
                "columns": [
                    {"name": "AuthorID", "dtype": "foreign_key", "link_to": "Author.AuthorID"},
                    {"name": "PaperID", "dtype": "foreign_key", "link_to": "Paper.PaperID"},
                ]
            },
            {
                "name": "Journal",
                "source": "data/journal.pqt",
                "columns": [
                    {"name": "JournalID", "dtype": "primary_key"},
                    {"name": "Name", "dtype": "text"},
                    {"name": "ImpactFactor", "dtype": "float"},
                    {"name": "Country", "dtype": "foreign_key", "link_to": "Country.CountryID"}, 
                    {"name": "PublisherID", "dtype": "foreign_key", "link_to": "Publisher.PublisherID"}
                ]
            },
            {
                "name": "Publisher",
                "source": "data/publisher.pqt",
                "columns": [
                    {"name": "PublisherID", "dtype": "primary_key"},
                    {"name": "PublisherName", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
            
        ]
    }
        </output schema>
        <thought>
            Comparing two schems, we can find that we do the following changes:
            1. "Author", a multi_category column in the original table, is converted to a new table "Writes" to represent the relationship between authors and papers. We need to 
            generate the code for augmenting this new table. For multi_category transformation, we can use the "explode" function from the pandas library.
            Here, we need to include the column "PaperID" in the new table since "Writes" is a relationship table between authors and papers.
            Here, we don't need to remove the columns "Authors" in the original table since it's already removed in the schema. 
            We don't need to generate code for removing columns and just removing them in schema is enough.
            2. "Keywords" in the original table is a category column. Since it's a dummy table with only one primary key column, we don't need to generate the code for it.
            3. "Country" in the original table is a category column. Since it's a dummy table with only one primary key column, we don't need to generate the code for it.
        </thought>
        <code>
            paper_df = pd.read_parquet("datasets/Papers/data/paper.pqt")
            ### augment the write relationship here
            ## no need to do any renaming here
            hasauthor_df = paper_df[["Authors", "PaperID"]].explode("Authors").reset_index(drop=True)
            hasauthor_df.to_parquet("datasets/Papers/data/hasauthor.pqt")
            ### no need to generate any code for "Keywords"
            
            ### no need to generate any code for "Country"
            
        </code>
    </example>
    """
    
    
    CODE_GENERATION_PROMPT = f"""
    Imagine you are an expert python programmer, please generate the corresponding python code
    to complete this augmentation by changing the value of the corresponding tables, you will encounter following data.

    <data format>
        You should change the files under the directory "datasets/{dataset}/data", where you should consider about the 
        following files:
        {full_info}
        
        * If you encounter a parquet file, you can just read it using pd.read_parquet and treat it like a pandas dataframe.
        * If you encounter a npz file, you can just read it using np.load and treat it like a dictionary. Each key 
        corresponds to a column in the table, and the value is the corresponding values.
        * You only need to consider the tables existing in the "tables" columns. For dummy tables with only one primary
        key column only shown in the "link_to" part, you don't need to generate them. 
        * If you generate a new table from the original table, you only need to generate the code to generate the 
        new table, but don't need to remove the data from the original table. 
        * If you find the newly added table is a dummy table with only one primary key column (declared with "link_to"), you don't need to generate
        the code to generate the table. 
    </data format>

    {EXAMPLE}
    
    Now you are given the following inputs:
    <input schema>
    {input_schema}
    </input schema>
    <output schema>
    {output_schema}
    </output schema>
    
    output the code in with the following format, directly generate runnable code with no bugs in the code block, 
    no need to wrap them with markdown.
    <thought>
    {{Your thought should be here}}
    </thought>
    <code>
    {{Your code should be here}}
    </code>
    """
    return CODE_GENERATION_PROMPT
    
def graph_check_prompt(schema):
    GRAPH_CHECK_PROMPT = f"""
        Imagine you are a graph data scientist, and you are given a dataset with the following schema:
        You will be given an original schema represented in the dictionary format:
        <data>
            You are given a series of tabular data, each of them containing some columns and represents
            one table. You will be given an original dataset schema with a dictionary format, which will contains the following information:
            1. dataset_name: name of the dataset 
            2. tables: meta data for list of tables, each one will present following attributes
                1. name: table name
                2. source: source of the data, can either be a numpy .npz file or a parquet file
                3. columns: list of columns, each column will have following attributes
                    1. name: column name
                    2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category.
                    primary_key and foreign_key are two special types of categorical columns, which presents a structural
                    relationship with other tables. Multi_category means this column is of list type, and each cell main contains
                    a list of categorical values. 
                    3. link_to (optional): if this column is a foreign key, point to which primary key from which table
        </data>
        
        Now I want you to give a score ranging from 0 to 100 to the following graph schema, which is represented in the dictionary format:
        <schema>
            {schema}
        </schema>
        
        You should output the score with the following format
        <score>
            {{Your score}}
        </score>
    """  
    
    return GRAPH_CHECK_PROMPT



