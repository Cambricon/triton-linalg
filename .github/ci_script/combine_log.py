import time
import sys
import os
'''
  Get info.
    output_path: the target file that you want to combine sub log with.
    list_path: the list of sub log name. When it is updated, the correspondding file will be add to output tail.
    list_dir_path: the dir path where sub logs stored.
    status_path: the path of status file. When status file is written to "success" or "fail", exit script.
'''

output_path = sys.argv[1]
list_path = sys.argv[2]
list_dir_path = sys.argv[3]
status_path = sys.argv[4]

if __name__ == '__main__':
    # list_pos stores the last position that pointer of list file pointed to.
    list_pos = 0
    while True:
        list_file = open(list_path, 'r')
        list_file.seek(list_pos)
        # read all lines starting from list_pos.
        items = list_file.readlines()
        # update list_pos
        list_pos = list_file.tell()
        # if read any line
        if items is not None:
            items.sort()
            for item in items:
                sub_path = item.strip()
                if sub_path != "":
                    file_name = list_dir_path + '/' + sub_path
                    # while True:
                    if os.path.exists(file_name):
                        os.system('cat ' + file_name + ' >> ' + output_path)
                        # break
        # check status_file, when read "success" or "fail" exit cycle, or else, sleep some seconds and start from beginning.
        status_file = open(status_path)
        status = status_file.readline().strip()
        status_file.close()
        if "fail" in status or "success" in status:
            break
        else:
            time.sleep(2)
