from  Ashare import *

import datetime
today=datetime.date.today()
print(str(today))

df=get_price('sh512290',frequency='1d',count= 5000,end_date=str(today))  #512290    #默认获取今天往前5天的日线实时行情
df.to_csv("5112290.csv")
print('',df)