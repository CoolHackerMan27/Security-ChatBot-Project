import tensorflow as tf
import sqlite3
import json
from tensorflow.keras.models import Sequential

timeframe = '2015-05'
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()

def create_table():
    c.execute("""CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, 
                 comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)""")


def formatData(data):
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    return data

if __name__ == "__main__":
    create_table()
    row_counter = 0
    paired_rows = 0
    with open("path/to/data".format(timeframe.split('-')[0], timeframe, buffer=1000)) as f:
        for row in f:
            row_counter += 1
            row - json.loads(row)
            parent_id = row['parent_id']
            body = formatData(row['body'])
            created_utc = row['created_utc']
            score = row['score']
            subreddit = row['subreddit']


