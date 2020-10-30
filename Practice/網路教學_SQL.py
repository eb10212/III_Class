import pymysql
import pandas as pd
# 定義連結到mysql的函式，返回連線物件
# db_name是當前資料庫的名字
def getcon(db_name):
    # host是選擇連線哪的資料庫localhost是本地資料庫，port是埠號預設3306
    #user是使用的人的身份，root是管理員身份，passwd是密碼。db是資料庫的名稱，charset是編碼格式
    conn=pymysql.connect(host="localhost",port=3306,user='root',passwd='u65p0123',db=db_name,charset='utf8')
    # 建立遊標物件
    cursor1=conn.cursor()
    return conn,cursor1
# 定義讀取檔案並且匯入資料庫資料sql語句
def insertData(db_name,table_name):
    # 呼叫連結到mysql的函式，返回我們的conn和cursor1
    conn,cursor1=getcon(db_name)
    # 使用pandas 讀取csv檔案
    df=pd.read_csv('m.csv')
    #使用for迴圈遍歷df，是利用df.values，但是每條資料都是一個列表
    # 使用counts計數一下，方便檢視一共添加了多少條資料
    counts = 0
    for each in df.values:
        # 每一條資料都應該單獨新增，所以每次新增的時候都要重置一遍sql語句
        sql = 'insert into '+table_name+' values('
        # 因為每條資料都是一個列表，所以使用for迴圈遍歷一下依次新增
        for i,n in enumerate(each):
            # 這個時候需要注意的是前面的資料可以直接前後加引號，最後加逗號，但是最後一條的時候不能新增逗號。
            # 所以使用if判斷一下
            if i < (len(each) - 1):
                sql = sql + '"' + str(n) + '"' + ','
                # #因為其中幾條資料為數值型，所以不用新增雙引號
                # if i<=4 or i==8 or i==9:
                #     sql = sql+ str(n) + ','
                # else:
                #     sql = sql + '"' + str(n) + '"' + ','
            else:
                sql = sql + '"' + str(n) + '"'
        sql = sql + ');'
        # print(sql)
        # 當添加當前一條資料sql語句完成以後，需要執行並且提交一次
        cursor1.execute(sql)
        # 提交sql語句執行操作
        conn.commit()
        # 沒提交一次就計數一次
        counts+=1
        #使用一個輸出來提示一下當前存到第幾條了
        print('成功添加了'+str(counts)+'條資料 ')
    return conn,cursor1
# 主函式
def main(db_name,table_name):
    conn, cursor1 =insertData(db_name,table_name)
    # 當新增完成之後需要關閉我們的遊標，以及與mysql的連線
    cursor1.close()
    conn.close()
# 判斷一下，防止再次在其他檔案呼叫當前函式的時候會使用錯誤，多次呼叫
if __name__=='__main__':
    main('movie_test','Basic_content')