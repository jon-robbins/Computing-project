from data_util import DataUtil

du = DataUtil(host='106.15.228.118', user='root', password='', port=3306, dbname='oldfarts', charset='utf8')

df = du.datafromsql('select * from sample_diabetes_mellitus_data')

df.info()