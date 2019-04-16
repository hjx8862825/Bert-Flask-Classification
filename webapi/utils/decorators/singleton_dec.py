"""
单例注解,和java用法一样@singleton
"""
def singleton(cls,*args,**kw):
    instance = {}
    def _singleton():
        if cls not in instance:
            instance[cls] = cls(*args,**kw)
        return instance[cls]
    return _singleton
