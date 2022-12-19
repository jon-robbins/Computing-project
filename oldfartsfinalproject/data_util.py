import pymysql
import pandas as pd


class DataUtil:

    def __init__(self, host, user, password, port, dbname, charset):
        self.con = None
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.dbname = dbname
        self.charset = charset

    def connect(self):
        self.con = pymysql.connect(host=self.host, database=self.dbname, user=self.user, password=self.password,
                                   port=self.port, charset=self.charset)

    def close(self):
        self.con.close

    def datafromsql(self, sql):
        try:
            self.connect()
            df = pd.read_sql(sql, self.con)
            self.close()
            return df
        except:
            print("查询失败")
            return None

    def datafrompath(self, path):
        try:
            df = pd.read_csv(path, sep=',')
            return df
        except:
            print("查询失败")
            return None
