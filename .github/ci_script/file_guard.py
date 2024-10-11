import time
import sys
import os
import argparse


def file_guard(guard_status_file, guard_log_file):
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
        if "success" in line or "Success" in line:
            print("Task success.")
            break
        elif "fail" in line or "Fail" in line or "error" in line or "Error" in line:
            print("Task Fail.")
            exit(-1)
        # sleep for a while
        time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Monitor a log file and echo lines, check status to stop.")
    parser.add_argument('guard_status_file',
                        type=str,
                        help='The path to the status file.')
    parser.add_argument('guard_log_file',
                        type=str,
                        help='The path to the log file.')

    args = parser.parse_args()

    file_guard(args.guard_status_file, args.guard_log_file)
