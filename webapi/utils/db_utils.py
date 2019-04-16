import pymysql
from DBUtils.PooledDB import PooledDB
from webapi.app_configs import General_db_config,Tag_db_config,Log_config
from webapi.utils.log_utils import my_logger

class General_db_executor(object):

    def __init__(self):
        self.logger = my_logger(Log_config.log_dir + 'dbLog.txt')

    def getConnection(self,pool):
        return pool.getConnection()

    # 执行带参的sql
    def execute_uncommit_sql(self,pool,sql):
        self.logger.debug("executing uncommit sql:", sql)
        conn = pool.getConnection()
        cursor = conn.cursor()
        cursor.execute(sql)
        result = cursor.fetchall()
        cursor.close()
        # 非真正关闭
        conn.close()
        return result

    def execute_commit_sql(self,pool,sql):
        self.logger.debug("executing commit sql:", sql)
        conn = pool.getConnection()
        cursor = conn.cursor()
        rs = False
        try:
            rs = cursor.execute(sql)
            conn.commit()

        except:
            conn.rollback()

        cursor.close()
        conn.close()
        return rs

"""
************************************下面是连接池数据源*****************************************
"""

class Tag_db_pool(object):
    def __init__(self):
        self.pool = PooledDB(pymysql,mincached=General_db_config.mincached,maxcached=General_db_config.maxcached,
                            host=Tag_db_config.tag_host_url,
                            user=Tag_db_config.tag_username,
                            passwd=Tag_db_config.tag_password,
                            db=Tag_db_config.tag_db,
                            port=Tag_db_config.tag_port,
                            charset=Tag_db_config.tag_charset)

    def getConnection(self):
        return self.pool.connection()


# 测试一下
if __name__ == '__main__':
    pool = Tag_db_pool()
    executor = General_db_executor()
    result = executor.execute_uncommit_sql(pool=pool,sql="select * from robot_new_tags where tag_type_id='%d'" % (1))
    # rs2 = executor.execute_commit_sql(pool=pool,sql="insert into robot_new_tags (intype_id,tag_name,tag_type_id,cid,pid,tag_type_name,mission_id) values('%d','%s','%d','%d','%d','%s','%d')"
    #                                                 % (1,'测试',99,0,0,'神仙',100))
    print(type(result))
    print(result)
    # print(rs2)