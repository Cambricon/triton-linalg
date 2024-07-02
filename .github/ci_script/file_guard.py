import time
import sys
import os

guard_status_file = sys.argv[1]
guard_log_file = sys.argv[2]

if __name__ == '__main__':
    # where stores the last position that pointer pointed to.
    where = 0
    while True:
        file = open(guard_log_file, "r")
        file.seek(where)
        # if read any lines, call system echo to print each line.
        for line in file.readlines():
            new_line = line.strip().replace("\'", "_").replace("\"", "_")
            os.system('echo ' + "'%s'" % new_line)
        # update where
        where = file.tell()
        file.close()
        # check status, end process when read "success" or "fail"
        status_file = open(guard_status_file, "r")
        line = status_file.readline().strip()
        status_file.close()
        if "success" in line:
            break
        elif "fail" in line or "error" in line:
            exit(-1)
        # sleep for a while
        time.sleep(2)
