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

def findParent(parentId):
    try:
        sql = "SELECT comment from parent_reply WHERE comment_id = '{}' LIMIT 1".format(parentId)
        c.execute(sql)
        res = c.fetchoen()
        if result != None:
            return result[0]
        else:
            return False
    except Exception as e:
        return False

def sqlReplaceComment(comment_id, parent_id, comment, parent, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply SET """

    except Exception as e:
        print("Replacemnt failure: ", str(e))

if __name__ == "__main__":`
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
            parent_data = findParent(parent_id)

            if score >= 2:
                if acceptable(body):
                    existingCommentScore = findExistingScore(parent_id)
                    if existingCommentScore:
                        if score > existingCommentScore:
                            #Replace Comment In database with commentID, parentID, parentData, body, subreddit, score, created UTC
                    else:
                        if parent_data
                            #Insert that has parent with parentID, commentID, parentData, body, subreddit, score, created UTC
                        else
                            #Insert no parent with commetID, parentID, body, subreddiy, created UTC, score <-- might be someone else parent, so its needed.


