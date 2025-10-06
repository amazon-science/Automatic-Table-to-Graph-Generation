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
        "repeater": "This task is to predict whether a user will repeat a purchase given the user's purchase history and user-item structural information"
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
    "custom": {
        "repeater": "This task is to predict whether a user will repeat a purchase given the user's purchase history and user-item structural information"
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