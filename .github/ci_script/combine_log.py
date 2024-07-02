import time
import sys
import os
import argparse
'''
  Get the result information fed back from the job server. If it is 'success' or 'failed', exit the pipeline. Otherwise, continue to monitor job information every 2 seconds.
    output_path: the target file that you want to combine sub log with.
    list_path: the list of sub log name. When it is updated, the correspondding file will be add to output tail.
    list_dir_path: the dir path where sub logs stored.
    status_path: the path of status file. When status file is written to "success" or "fail", exit script.
'''

def combine_log(output_path, list_path, list_dir_path, status_path):
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
        if "fail" in status or "success" in status or "Success" in status or "Fail" in status or "error" in status or "Error" in status:
            break
        else:
            time.sleep(2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Monitor and concatenate files based on a list.")
    parser.add_argument('output_path', type=str, help='The path to the output file.')
    parser.add_argument('list_path', type=str, help='The path to the list file containing sub-paths.')
    parser.add_argument('list_dir_path', type=str, help='The base directory where sub-paths are located.')
    parser.add_argument('status_path', type=str, help='The path to the status file.')

    args = parser.parse_args()
    combine_log(args.output_path, args.list_path, args.list_dir_path, args.status_path)
    
