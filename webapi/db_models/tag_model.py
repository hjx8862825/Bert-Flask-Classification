from webapi.utils.db_utils import Tag_db_pool,General_db_executor

def get_tag_by_mission_and_type(missionid,typeid):
    pool = Tag_db_pool()
    executor = General_db_executor()
    sql = "select * from robot_new_tags where mission_id = %d and tag_type_id = %d" % (missionid,typeid)
    res = executor.execute_uncommit_sql(pool=pool,sql=sql)
    return res

if __name__ == '__main__':
    result = get_tag_by_mission_and_type(1,1)
    print(result)