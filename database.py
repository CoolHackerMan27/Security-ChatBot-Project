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

def transactionBuilder (sql):
    global sql_transaction
    sql_transaction.append(sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []
    
def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True

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
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transactionBuilder(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sqlInsertParent(comment_id, parent_id, comment, parent, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(parentid, commentid, parent, comment, subreddit, int(time), score)
        transactionBuilder(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def sqlInsertNoParent(comment_id, parent_id, comment, parent, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(parentid, commentid, comment, subreddit, int(time), score)
        transactionBuilder(sql)
    except Exception as e:
        print('s0 insertion',str(e))

def filterData():
    #TODO

    #Will filter all inapropiate language 


if __name__ == "__main__":`
    create_table()
    row_counter = 0
    paired_rows = 0
    with open("path/to/data".format(timeframe.split('-')[0], timeframe, buffer=1000)) as f:
        for row in f:
            row_counter += 1
            row - json.loads(row)
            parent_id = row['parent_id']
            comment_id = row['name']
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
                                sqlReplaceComment(comment_id, parent_id, body, parent_data, subreddit, created_utc, score)
                            #Replace Comment In database with commentID, parentID, parentData, body, subreddit, score, created UTC
                            
                    else:
                        if parent_data
                            #Insert that has parent with parentID, commentID, parentData, body, subreddit, score, created UTC
                            sqlInsertParent(comment_id, parent_id, body, parent_data, subreddit, created_utc, score)
                            paired_rows += 1
                        else
                            #Insert no parent with commetID, parentID, body, subreddiy, created UTC, score <-- might be someone else parent, so its needed.
                            sqlInsertNoParent(comment_id, parent_id, body, subreddit, created_utc, score)
           
            if row_counter % 100000 == 0:
                print('Total Rows Read: {}, Paired Rows: {}, Time: {}'.format(row_counter, paired_rows, str(datetime.now())))
                            


