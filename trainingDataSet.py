import sqlite3
import pandas as pd
#all months from 2007-07 to 2015-05
timeframes = ['2007-07', '2007-08', '2007-09', '2007-10', '2007-11', '2007-12',
    '2008-01', '2008-02', '2008-03', '2008-04', '2008-05', '2008-06',
    '2008-07', '2008-08', '2008-09', '2008-10', '2008-11', '2008-12',
    '2009-01', '2009-02', '2009-03', '2009-04', '2009-05', '2009-06',
    '2009-07', '2009-08', '2009-09', '2009-10', '2009-11', '2009-12',
    '2010-01', '2010-02', '2010-03', '2010-04', '2010-05', '2010-06',
    '2010-07', '2010-08', '2010-09', '2010-10', '2010-11', '2010-12',
    '2011-01', '2011-02', '2011-03', '2011-04', '2011-05', '2011-06',
    '2011-07', '2011-08', '2011-09', '2011-10', '2011-11', '2011-12',
    '2012-01', '2012-02', '2012-03', '2012-04', '2012-05', '2012-06',
    '2012-07', '2012-08', '2012-09', '2012-10', '2012-11', '2012-12',
    '2013-01', '2013-02', '2013-03', '2013-04', '2013-05', '2013-06',
    '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12',
    '2014-01', '2014-02', '2014-03', '2014-04', '2014-05', '2014-06',
    '2014-07', '2014-08', '2014-09', '2014-10', '2014-11', '2014-12',
    '2015-01', '2015-02', '2015-03', '2015-04', '2015-05']

for timeframe in timeframes:
    try:
            connection = sqlite3.connect('{}.db'.format(timeframe))
            c = connection.cursor()
            limit = 5000
            last_unix = 0
            cur_length = limit
            counter = 0
            test_done = False

            while cur_length == limit:

                df = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(last_unix,limit),connection)
                last_unix = df.tail(1)['unix'].values[0]
                cur_length = len(df)

                if not test_done:
                    with open('test.from','a', encoding='utf8') as f:
                        for content in df['parent'].values:
                            f.write(content+'\n')

                    with open('test.to','a', encoding='utf8') as f:
                        for content in df['comment'].values:
                            f.write(str(content)+'\n')

                    test_done = True

                else:
                    with open('train.from','a', encoding='utf8') as f:
                        for content in df['parent'].values:
                            f.write(content+'\n')

                    with open('train.to','a', encoding='utf8') as f:
                        for content in df['comment'].values:
                            f.write(str(content)+'\n')

                counter += 1
                if counter % 20 == 0:
                    print(counter*limit,'rows completed so far')
    except Exception as e:
        print(str(e))