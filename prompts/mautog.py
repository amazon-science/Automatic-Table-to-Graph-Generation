"""
    Auto graph construction: multi-round conversation version. 
    Camera ready vs submission version. 
    We change the single-step-single-action version to single-step-multi-action version.
    We also add the similarity of joinable table discovery.
"""


def get_example():
    """
        old version
    """
    EXAMPLE1 = """
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
<selection>
generate_or_connect_dummy_table
</selection>

<parameters>
"Journal", "Country", "Country", "CountryID"
</parameters>

<explanation>
The "Country" column in the Journal table can be used to generate a new node type, there's no related columns as features, so we need to generate a dummy table.
From the column statistics, we can see that the value of "Country" is not an id starting from 0, so we need to turn it into an id and set the last parameter to "yes".
"r2n" should be adopted since apparently two tables correspond to two entites "Paper" and "Journal".
</explanation>

<construction>
r2n
</construction>
    """
    EXAMPLE2 = """
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
    "data type": "foreign_key"
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
                    {"name": "Country", "dtype": "foreign_key", "link_to": "Country.CountryID"},
                    {"name": "Publisher", "dtype": "text"},
                    {"name": "PublisherLocation", "dtype": "category"}
                ]
            }
        ]
    }
</schema>
</input>
<selection>
explode_multi_category_column
</selection>

<parameters>
"Paper", "Authors", "PaperID", "Writes", "Author", "foreign_key"
</parameters>

<explanation>
"Author" is a multi-category column which has a moderate number of unique values, which can potentially generate
good edge relationships. We can explode this column into multiple columns to represent the authors of the paper.
The generated relation is about a paper is written by an author. So we name it by "Writes". The dtype of the new column
should be "foreign_key" since "Writes" should be a relation. "r2n" should be adopted since apparently two tables correspond to two entites "Paper" and "Journal".
</explanation>

<construction>
r2n
</construction>
    """
    EXAMPLE3 = """
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
    <selection>
    connect_two_columns
    </selection>

    <parameters>
    "View", "UserID", "Purchase", "UserID", "User", "UserID"
    </parameters>

    <explanation>
    The "UserID" column in the View table and the Purchase table are related, but they are not yet in a primary-foreign key relationship.
    We can connect these two columns with a foreign key constraint, and generate a new table "User" with the "UserID" column.
    We select to generate a new table since "User" should be an independent node type. "r2ne" should be adopted since 
    after the transformation, "Purchase" is a table with two foreign keys and present the role of an edge.
    </explanation>

    <construction>
    r2ne
    </construction>
    """
    
    EXAMPLE4 = """
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
{
    "Column": "ManufacturerInfo",
    "data type": "text"
}
{
    "Column": "Manufacturer",
    "data type": "category",
    "Number of unique values": 312,
  "Number of nan values": 0,
  "Number of total values": 128564,
  "Mode values": "Apple",
  "5 sampled values": [
    "Apple",
    "Sony",
    "Samsung",
    "Apple",
    "Microsoft"
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
                    {"name": "Category", "dtype": "category"},
                    {"name": "ManufacturerInfo", "dtype": "text"}
                    {"name": "Manufacturer", "dtype": "category"}
                    
                ]
            }
        ]
    }
    </schema>
    <tasks>
    Now I want to train a model which can predict the category of a product based on the information in the product.
    </tasks>
    </input>
    <selection>
    generate_or_connect_dummy_table
    </selection>

    <parameters>
    "Product", ["ManufacturerInfo","Manufacturer"], "ManufacturerID", "Manufacturer" 
    </parameters>

    <explanation>
    The "Manufacturer" column in the Product table is an independent column which can be used to generate a new node type, and
    "ManufacturerInfo" is a text column which should be used as the feature of the new node type. We can generate a non-dummy table
    "Manufacturer" with columns "ManufacturerInfo" and "Manufacturer". Then in the original table, we can replace the "Manufacturer" column
    with a foreign key "ManufacturerID" to the new table. "r2n" should be adopted since there's no table with two foreign keys.
    </explanation>

    <construction>
    r2n
    </construction>
    """
    
    EXAMPLE5 = """
An example will be as follows:
    <input>
    <dataset_stats>
        Table: View
{
  "Column": "UserID",
  "data type": "foreign_key"
}
{
  "Column": "ItemID",
  "data type": "foreign_key"
}
Table: Purchase
{
  "Column": "UserID",
  "data type": "foreign_key"
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
Table: User
{
    "Column": "UserID",
    "data type": "primary_key"
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
    <tasks>
    Now I want to train a model which can predict the category of a product based on the information in the product.
    </tasks>
    </input>
    <selection>
    None
    </selection>

    <parameters>
    </parameters>

    <explanation>
    The schema is already well-constructed, and no further actions are needed. "r2ne" should be adopted since 
    after the transformation, "Purchase" is a table with two foreign keys and present the role of an edge.
    </explanation>

    <construction>
    r2ne
    </construction>
    
    """
    
    EXAMPLE6 = """
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
                    {"name": "UserInformation", "dtype": "foreign_key", "link_to": "UserTable.UserID"},
                    {"name": "ItemID", "dtype": "foreign_key", "link_to": "Product.ItemID"}
                ]
            }, 
            {
                "name": "Purchase",
                "source": "data/purchase.npz",
                "columns": [
                    {"name": "PurchaserID", "dtype": "category"},
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
    <selection>
    generate_or_connect_dummy_table
    </selection>

    <parameters>
    "Purchase", "PurchaserID", "UserTable", "UserID"
    </parameters>

    <explanation>
    Defined in the "View" table, "UserTable" is a dummy table defined with one primary key "UserID". "PurchaserID" in the "Purchase"
    represents the samething, as can be shown by the sampled data, but they are not yet in a primary-foreign key relationship.
    We can connect these two columns with a foreign key constraint. You should remember that you can only apply this action when the column is a categorical column,
    like "PurchaserID". "r2ne" should be adopted since after the transformation, "Purchase" and "View" are tables with two foreign keys and present the role of an edge.
    </explanation>
    
    <construction>
    r2ne
    </construction>
    """
    
    return [EXAMPLE1, EXAMPLE6, EXAMPLE3, EXAMPLE4, EXAMPLE5, EXAMPLE2]


def get_single_round_multi_step_prompt():
    """
        AutoG-S
    """
    prompt = """
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
    "data type": "category",
    "Number of unique values": 100,
    "Number of nan values": 0,
    "Number of total values": 10000,
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
                    {"name": "Journal", "dtype": "category"},
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
Here we gives the similarity score of each column pair, you can use this information to determine whether two columns may be joinable. The similarity score is scaled to [0, 1], the larger means the more similar.
<similarity>
The pair with the 1st highest similarity is column "Journal" from Table "Paper" and column "Name" from Table "Journal" with similarity 0.885
The pair with the 2nd highest similarity is column "Authors" from Table "Paper" and column "Name" from Table "Journal" with similarity 0.743
The pair with the 3rd highest similarity is column "Authors" from Table "Paper" and column "Country" from Table "Journal" with similarity 0.723
</similarity>
</input>



We need to think about whether we need to do one of the six actions:
1. First, for explode_multi_category_column, the Authors of the paper are in a multi-category column. Moreover, author is closely related to the category of the paper, so the relationship Paper-Author-Paper can be very useful. So, we need to explode this multi category column.
2. For connect_two_columns, the Journal column in the Paper table and the  column Name in the Journal table are highly similar, so we can connect these two columns with a foreign key constraint. Other pairs like Authors and Name, Authors and Country are not similar enough to be connected.
3. For generate_non_dummy_table, the Publisher and PublisherLocation columns are independent columns for the entity Publisher. We can generate a new table Publisher with these two columns.
4. For generate_or_connect_dummy_table, we need to find those categorical columns beneficial for downstream task. We have categorical columns (Journal has been deleted in step 2, Category is the final objective) Keyword, Country, this will result in relationship Paper-Keyword-Paper and Paper-Journal-Country-Journal-Paper respectively. Since the target is to predict the category of a paper, we can generate a dummy table for the column Keyword since paper sharing the same keyword are highly likely to share the same category. Country may be not beneficial since it doesn't present a strong semantic relationship with the category. 
5. For remove_primary_key and add_primary_key, there's no unreasonable primary key or missing primary key in the table, so we don't need to do this action. as a result, we have the following actions
<selection>
        [{{'explanation': "Author is multi-category and Paper-Author-Paper is probably useful. We set the dtype to foreign_key because we want to use the relation", 'action': 'explode_multi_category_column', 'parameters': {'original_table': 'Paper', 'multi_cat_col': 'Author', primary_key_column: 'PaperID', 'new_table_name': 'Author', 'new_col_name': 'AuthorName', 'dtype': 'foreign_key'}},
        {{'explanation': 'the Journal column in the Paper table and the  column Name in the Journal table are highly similar, both of them should refer to the name of the journal', 'action': 'connect_two_columns', 'parameters': {'table_1_name': 'Paper', 'table_1_col_name': 'Journal', 'table_2_name': 'Journal', 'table_2_col_name': 'Name', 'new_table_name': "", 'new_table_col_name': "" }}, 
        {{'explanation': 'Publisher and PublisherLocation are independent columns for the entity Publisher. We can generate a new table Publisher with these two columns', 'action': 'generate_non_dummy_table', 'parameters': {'base_table_name': 'Paper', 'cols': ['Publisher', 'PublisherLocation'],  'new_table_name': 'Publisher'}},
        {{'explanation': 'Keyword is a categorical column which can be used to generate a dummy table. Country is not beneficial for the downstream task', 'action': 'generate_or_connect_dummy_table', 'parameters': {'base_table_name': 'Paper', 'orig_col_name': 'Keyword', 'new_table_name': 'Keyword', 'new_col_name': 'Keyword'}},
        ]
        </selection>

    
""" 
    return [prompt]


def get_multi_round_single_step_prompt():
    """
        AutoG-A
    """
    prompt = """
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
    "data type": "category",
    "Number of unique values": 100,
    "Number of nan values": 0,
    "Number of total values": 10000,
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
                    {"name": "Journal", "dtype": "category"},
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
Here we gives the similarity score of each column pair, you can use this information to determine whether two columns may be joinable. The similarity score is scaled to [0, 1], the larger means the more similar.
<similarity>
The pair with the 1st highest similarity is column "Journal" from Table "Paper" and column "Name" from Table "Journal" with similarity 0.885
The pair with the 2nd highest similarity is column "Authors" from Table "Paper" and column "Name" from Table "Journal" with similarity 0.743
The pair with the 3rd highest similarity is column "Authors" from Table "Paper" and column "Country" from Table "Journal" with similarity 0.723
</similarity>
</input>



We need to think about whether we need to do one of the five actions:
1. First, for explode_multi_category_column, the Authors of the paper are in a multi-category column. Moreover, author is closely related to the category of the paper, so the relationship Paper-Author-Paper can be very useful. So, we need to explode this multi category column.
2. For connect_two_columns, the Journal column in the Paper table and the  column Name in the Journal table are highly similar, so we can connect these two columns with a foreign key constraint. Other pairs like Authors and Name, Authors and Country are not similar enough to be connected.
3. For generate_non_dummy_table, the Publisher and PublisherLocation columns are independent columns for the entity Publisher. We can generate a new table Publisher with these two columns.
4. For generate_or_connect_dummy_table, we need to find those categorical columns beneficial for downstream task. We have categorical columns (Journal has been deleted in step 2, Category is the final objective) Keyword, Country, this will result in relationship Paper-Keyword-Paper and Paper-Journal-Country-Journal-Paper respectively. Since the target is to predict the category of a paper, we can generate a dummy table for the column Keyword since paper sharing the same keyword are highly likely to share the same category. Country may be not beneficial since it doesn't present a strong semantic relationship with the category. 
5. For remove_primary_key and add_primary_key, there's no unreasonable primary key or missing primary key in the table, so we don't need to do this action. 

When you make the decision, you only need to output one action you think the most important. You don't need to output all actions you think of.

One candidate action can be:

<selection>
        [{{'explanation': "Author is multi-category and Paper-Author-Paper is probably useful. We set the dtype to foreign_key because we want to use the relation", 'action': 'explode_multi_category_column', 'parameters': {'original_table': 'Paper', 'multi_cat_col': 'Author', primary_key_column: 'PaperID', 'new_table_name': 'Author', 'new_col_name': 'AuthorName', 'dtype': 'foreign_key'}}
</selection>

One candidate action can be:
        <selection>
        [{{'explanation': 'the Journal column in the Paper table and the  column Name in the Journal table are highly similar, both of them should refer to the name of the journal', 'action': 'connect_two_columns', 'parameters': {'table_1_name': 'Paper', 'table_1_col_name': 'Journal', 'table_2_name': 'Journal', 'table_2_col_name': 'Name'}}] 
        </selection>

One candidate action can be:
        <selection>
        {{'explanation': 'Publisher and PublisherLocation are independent columns for the entity Publisher. We can generate a new table Publisher with these two columns', 'action': 'generate_non_dummy_table', 'parameters': {'base_table_name': 'Paper', 'cols': ['Publisher', 'PublisherLocation'],  'new_table_name': 'Publisher'}},
        </selection>

One candidate action can be:
        <selection>
        {{'explanation': 'Keyword is a categorical column which can be used to generate a dummy table. Country is not beneficial for the downstream task', 'action': 'generate_or_connect_dummy_table', 'parameters': {'base_table_name': 'Paper', 'orig_col_name': 'Keyword', 'new_table_name': 'Keyword', 'new_col_name': 'Keyword'}},
        ]
        </selection>

    
""" 
    return [prompt]


def get_multi_round_action_selection_prompt_epoch(actions, example, history_actions, input_schema, stats, task, jtd):
    MULTI_ROUND = """
        Imagine you are an expert graph data scientist, and now you are expected to construct graph schema based on the original
        inputs. You will be given an original schema represented in the dictionary format:
        <data>
            1. dataset_name: name of the dataset 
            2. tables: meta data for list of tables, each one will present following attributes
                1. name: table name
                2. source: source of the data, can either be a numpy .npz file or a parquet file
                3. columns: list of columns, each column will have following attributes
                    1. name: column name
                    2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category.
                    primary_key and foreign_key are two special types of categorical columns, which presents a structural
                    relationship with other tables. Multi_category means this column is of list type, and each cell main contains
                    a list of categorical values. After a column is set as primary_key or foreign_key, it should not be changed to other types.
                    3. link_to (optional): if this column is a foreign key, point to which primary key from which table
            3. statistics of the table: statistics of the column value of tables. These statistics can be used to help you
            determine the characteristics of the columns. For example, if one categorical column only contains one unique value,
            then creating a node type based on this column can result in a super node, which is not ideal for graph construction.
            You should also determine whether two columns represent the same thing based on these statistics. 
            4. Dummy table is a special type of table. It's not explicitly defined with a table slot. It's defined in other tables, such as
            {{"name": "nation", "dtype": "foreign_key", "link_to": "Country.CountryID"}}. In this case, "Country" is a dummy table, which is not 
            explicitly defined in the tables slot.
        </data>                
        Here are the documents of the actions:
        
        {actions}
        
        
        Now, you need to 
        1. Actively think about whether any one of the four actions should be conducted; If not, you can select "None" and then halt the program.
        2. Output the selected action
        
        <selection>
        [{{'explanation': <explanation for the selection>, 'action': <selected action>, 'parameters': <parameters for the action> }}]
        </selection>

        
        3. If you think there's no more action, you can output 
        <selection>
        None
        </selection>
        
        Example:
        {example}
        
        History Actions:
        {history_actions}
        
        <input>
        <dataset_stats>
        {stats}
        </dataset_stats>
        <task>
        {task}
        </task>
        <schema>
        {input_schema}
        </schema>
        Here we gives the similarity score of each column pair, you can use this information to determine whether two columns may be joinable. The similarity score is scaled to [0, 1], the larger means the more similar.
        <similarity>
        {jtd}
        </similarity>
        </input>
        Return your output in the json format.
    """
    return MULTI_ROUND.format(actions=actions, example=example, history_actions=history_actions, 
                              stats=stats, task=task, input_schema=input_schema, jtd=jtd)



def get_multi_round_action_selection_prompt(actions, example, history_actions, input_schema, stats, task, jtd):
    MULTI_ROUND = """
        Imagine you are an expert graph data scientist, and now you are expected to construct graph schema based on the original
        inputs. You will be given an original schema represented in the dictionary format:
        <data>
            1. dataset_name: name of the dataset 
            2. tables: meta data for list of tables, each one will present following attributes
                1. name: table name
                2. source: source of the data, can either be a numpy .npz file or a parquet file
                3. columns: list of columns, each column will have following attributes
                    1. name: column name
                    2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category.
                    primary_key and foreign_key are two special types of categorical columns, which presents a structural
                    relationship with other tables. Multi_category means this column is of list type, and each cell main contains
                    a list of categorical values. After a column is set as primary_key or foreign_key, it should not be changed to other types.
                    3. link_to (optional): if this column is a foreign key, point to which primary key from which table
            3. statistics of the table: statistics of the column value of tables. These statistics can be used to help you
            determine the characteristics of the columns. For example, if one categorical column only contains one unique value,
            then creating a node type based on this column can result in a super node, which is not ideal for graph construction.
            You should also determine whether two columns represent the same thing based on these statistics. 
            4. Dummy table is a special type of table. It's not explicitly defined with a table slot. It's defined in other tables, such as
            {{"name": "nation", "dtype": "foreign_key", "link_to": "Country.CountryID"}}. In this case, "Country" is a dummy table, which is not 
            explicitly defined in the tables slot.
        </data>                
        Here are the documents of the actions:
        
        {actions}

        
        Now, you need to 
        1. Actively think about whether any one of the four actions should be conducted; If not, you can select "None" and then halt the program.
        2. output all actions you can think of from the above list to perform, and output your selection in the following format. It should be noted that for those actions with sequential relation like one new categorical column generated after expanding a multi-category column, you don't need to generate in one round.
        
        <selection>
        [{{'explanation': <explanation for the selection>, 'action': <first action>, 'parameters': <parameters for the first action> }},
        {{'explanation': <explanation for the selection>, 'action': <second action>, 'parameters': <parameters for the second action> }}, ...
        ]
        </selection>

        
        3. If not more action, output <selection>None</selection>
        
        Example:
        {example}
        
        History Actions:
        {history_actions}
        
        <input>
        <dataset_stats>
        {stats}
        </dataset_stats>
        <task>
        {task}
        </task>
        <schema>
        {input_schema}
        </schema>
        Here we gives the similarity score of each column pair, you can use this information to determine whether two columns may be joinable. The similarity score is scaled to [0, 1], the larger means the more similar.
        <similarity>
        {jtd}
        </similarity>
        </input>
        Return your output in the json format inside <selection></selection>.
    """
    return MULTI_ROUND.format(actions=actions, example=example, history_actions=history_actions, 
                              stats=stats, task=task, input_schema=input_schema, jtd=jtd)