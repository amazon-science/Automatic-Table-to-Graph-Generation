from llama_index.llms.bedrock import Bedrock
import os
import hashlib
import sqlite3
import typer

class LLMCache:
    def __init__(self, db_name="llm_cache.db"):
        self.conn = sqlite3.connect(db_name)
        if not self.table_exists("cache"):
            self.create_table()

    def table_exists(self, table_name):
        cursor = self.conn.cursor()
        cursor.execute(f'''
            SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'
        ''')
        return cursor.fetchone() is not None
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                query_hash TEXT PRIMARY KEY,
                response TEXT
            )
        ''')
        self.conn.commit()

    def get(self, query):
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cursor = self.conn.cursor()
        cursor.execute("SELECT response FROM cache WHERE query_hash = ?", (query_hash,))
        row = cursor.fetchone()
        return row[0] if row else None

    def set(self, query, response):
        query_hash = hashlib.sha256(query.encode()).hexdigest()
        cursor = self.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (query_hash, response))
        self.conn.commit()

    def close(self):
        self.conn.close()



def get_bedrock_llm(model_name="anthropic.claude-3-sonnet-20240229-v1:0", context_size=4096):
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.environ.get("AWS_SESSION_TOKEN")
    aws_region_name = "us-west-2"
    llm = Bedrock(
    model=model_name,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=aws_region_name,
    context_size=context_size
    )
    return llm 

def bedrock_llm_query(llm, messages, max_tokens = 4096, cache=False, debug = True, debug_dataset = "mag", debug_task = "venue", debug_round = 0, **kwargs):
    """
        Disclaimer: the non-debug version of this function is not tested since we no longer have access to the bedrock API
    """
    if debug:
        ## this will load some pre-saved messages 
        ## we put those test messages in the test directory
        this_file_dir = os.path.dirname(os.path.realpath(__file__))
        test_dir = os.path.join(this_file_dir, '..', '..', "test")
        test_file = os.path.join(test_dir, f"{debug_dataset}_{debug_task}.txt")
        msgs = []
        with open(test_file, "r") as f:
            for line in f:
                msgs.append(line.strip())
        if debug_round >= len(msgs):
            return "<selection>None</selection>"
        else:
            return msgs[debug_round]
    if not cache:
        if max_tokens == -1:
            return llm.complete(messages, **kwargs).text
        return llm.complete(messages, max_tokens=max_tokens, **kwargs).text
    else:
        ocache = LLMCache()
        if ocache.get(messages) is not None:
            typer.echo("Cache hit, return response from the cache")
            return ocache.get(messages)
        else:
            if max_tokens == -1:
                response = llm.complete(messages, **kwargs).text
            else:
                response = llm.complete(messages, max_tokens=max_tokens, **kwargs).text
            ocache.set(messages, response)
            return response
        





