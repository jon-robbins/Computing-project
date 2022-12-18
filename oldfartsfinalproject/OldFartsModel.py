from DataUtil import DataUtil

du = DataUtil(host='106.15.228.118', user='root', password='', port=3306, dbname='oldfarts', charset='utf8')

df = du.query_data('select * from sample_diabetes_mellitus_data')

df.info()