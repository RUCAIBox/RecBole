import datetime

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%F-%T')
    return cur
