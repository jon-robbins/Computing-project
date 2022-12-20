import sqlalchemy
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
        connect_str = 'mysql+pymysql://'+self.user+':'+self.password+'@'+self.host+':'+str(self.port)+'/'+self.dbname
        self.con = sqlalchemy.create_engine(connect_str)

    def close(self):
        self.con.close

    def datafromsql(self, sql):
        try:
            self.connect()
            df = pd.read_sql(sql, self.con)
            return df
        except:
            print("failed")
            return None

    def datafrompath(self, path):
        try:
            df = pd.read_csv(path, sep=',')
            return df
        except:
            print("failed")
            return None
