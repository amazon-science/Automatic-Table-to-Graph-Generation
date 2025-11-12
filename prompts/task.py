import json

task_description = {
    "mag": {
        "venue": "This task is to predict the venue of a paper given the paper's title, abstract, authors, and publication year. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
        "year": "This task is to predict the publication year of a paper given the paper's title, abstract, authors, and venue. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
        "cite": "This task is to predict whether two papers are cited by each other given the paper's title, abstract, authors, and venue. \
            You may use the meta relations between papers, authors, topics, and institutions to improve the performance",
    },
    "movielens": {
        "ratings": "This task is to predict user's ratings on movies given movie information and movie-user structural information"
    },
    "avs": {
        "relation": "This task is to find the primary keys and foreign keys among the given tables.",
        "kg": "This task is to detect and extract entities (e.g., Paper, Author, Product, Customer, Order, etc.) and relationships (e.g., written_by, belongs_to, purchased, cites, employed_by, etc.) from the given tables to construct a knowledge graph that represents the underlying data semantics.",
        "kg2": "This task is to detect and extract entities and relationships from the given tables to construct a knowledge graph that represents the underlying data semantics. Regarding entity Identification, to identify potential entities represented in the tables. Entities typically correspond to key columns (unique identifiers such as 'paper_id', 'author_id', 'product_id', 'customer_id', etc.). For each entity, determine its type (e.g., Paper, Author, Product, Customer, etc.). Regarding relationship extraction, to identify relationships within a single table (e.g., 'Paper_writer' --> 'Author', etc.), and to identify relationships across multiple tables via foreign keys, column references, or shared identifiers. Each relationship should connect two entities using a verb phrase or predicate that expresses their link (e.g., written_by, belongs_to, purchased, cites, employed_by, etc.)."
    },
    "custom_mag": {
        "venue": "This task is to predict the venue of a paper given the paper's title, abstract, authors, and publication year. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance"
    },
    "ieeecis": {
        "fraud": "This task is to predict whether a transaction is fraudulent given the transaction information and user-transaction structural information"
    },
    "diginetica": {
        "ctr": "This task is to predict the click-through rate of an ad given the ad information and user-ad structural information. In the task table, you are given itemId, queryId, timestamp, and clicked. The target is clicked. Moreover, itemId is a foreign key pointing to itemId of the Product table. queryId is a foreign key pointing to queryId of the Query table.", 
        "purchase": "This task is to predict whether a user will purchase an item given the item information and user-item structural information. In the task table, you are given itemId, queryId, timestamp, and clicked. The target is clicked. Moreover, itemId is a foreign key pointing to itemId of the Product table. purchase_session is a foreign key pointing to the Session table, which inspires that there should be one table Session"
    },
    "retailrocket": {
        "cvr": "The task is to classify whether an item will be added to the shopping cart by a visitor, i.e. predicting column View.added_to_cart"
    },
    "outbrain": {
        "ctr": "The task is to predict whether a promoted content will be clicked or not, i.e. predicting Click.clicked."
    },
    "stackexchange": {
        "upvote": "The task is to predict the Target column of table Posts, which means predicting whether the post will be upvoted or not.", 
        "churn": "The task is to predict the Target column of table Users, which means predicting whether the user will churn or not."
    },
    "ads": {
        "kg": "This task is to detect and extract entities and relationships from the given tables to construct a knowledge graph that represents the underlying data semantics. Regarding entity Identification, to identify potential entities represented in the tables. Entities typically correspond to key columns (unique identifiers such as 'node_id', 'customer_id', 'product_id', 'merchant_id', 'marketplace_id', etc.). For each entity, determine its type (e.g., Node, Customer, Product, Merchant, Marketplace, etc.). Regarding relationship extraction, to identify relationships within a single table (e.g., from Purchase table, 'order_id' contains 'asin', 'asin' purchased in 'order_id', etc.), and to identify relationships across multiple tables via foreign keys, column references, or shared identifiers. Each relationship should connect two entities using a verb phrase or predicate that expresses their link (e.g., sold_by, purchases, purchased_in, located_in, ordered_by, bought, etc.)."
    },
    "custom": {
        "relation": "This task is to find the primary keys and foreign keys among the given tables.",
        "kg": "This task is to detect and extract entities and relationships from the given tables to construct a knowledge graph that represents the underlying data semantics.",
        "kg2": "This task is to detect and extract entities and relationships from the given tables to construct a knowledge graph that represents the underlying data semantics. Regarding entity Identification, to identify potential entities represented in the tables and determine its type. Regarding relationship extraction, to identify relationships within a single table, and to identify relationships across multiple tables via foreign keys, column references, or shared identifiers."
    },
    "custom_mag": {
        "venue": "This task is to predict the venue of a paper given the paper's title, abstract, authors, and publication year. \
        You may use the meta relations between papers, authors, topics, and institutions to improve the performance"
    },
}


def get_task_description(dataset: str, task_name: str):
    try:
        return task_description[dataset][task_name]
    except KeyError:
        return ""


def get_task_meta_info(schema, selected_task):
    task_info = schema['tasks']
    meta_str = ""
    sel_task = None
    for i, info in enumerate(task_info):
        if info['name'] == selected_task:
            # import ipdb; ipdb.set_trace()
            meta_str += json.dumps(info)
            sel_task = info
            break
    meta_str += "\n"
    meta_str += f"Our target is to predict {sel_task['target_table']}.{sel_task['target_column']}. Don't change it."
    return meta_str


def get_task_meta(schema, selected_task):
    task_info = schema['tasks']
    for i, info in enumerate(task_info):
        if info['name'] == selected_task:
            return info 