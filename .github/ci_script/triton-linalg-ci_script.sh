# /bin/bash
# Get PR id
PR_string=$(echo $GITHUB_REF | grep -Eo "/[0-9]*/")
pr_id=(${PR_string//// })

# Generate time stamp
current=`date "+%Y-%m-%d %H:%M:%S"`
timeStamp=`date -d "$current" +%s` 
currentTimeStamp=$((timeStamp*1000+10#`date "+%N"`/1000000))

# Temporally set to mlu370
card_type="MLU370-S4"

# Default repo name
repo_name="triton-linalg"
# Repo ci root path
repo_root="/home/user1/${repo_name}_ci/"
if [ ! -d $repo_root ];then
    mkdir $repo_root
fi
# Repo ci requests path
requests_path="$repo_root/requests"
if [ ! -d $requests_path ];then
    mkdir $requests_path
fi

# Gen name of this ci
request_name="${repo_name}_${pr_id}_${currentTimeStamp}_${card_type}.rqt"

# Gen file and dir for this request
request_root="$repo_root/$request_name/"
sub_logs_path="$request_root/sub_logs/"


if [ ! -d $request_root ];then
    mkdir $request_root
fi

if [ ! -d $sub_logs_path ];then
    mkdir $sub_logs_path
fi

echo "working" > "$request_root/status"
chmod o+w "$request_root/status"

if [ ! -f  "$request_root/log" ];then
    touch "$request_root/log"
fi

chmod o+w "$request_root/log"

if [ ! -f "$request_root/log_list" ];then
    touch "$request_root/log_list"
fi

chmod o+w "$request_root/log_list"

# Gen request file.

echo "repo:${repo_name}" > "$requests_path/${request_name}"
echo "pr_id:${pr_id}" >> "$requests_path/${request_name}"
echo "timestamp:${currentTimeStamp}" >> "$requests_path/${request_name}"

# change dir group for server and client, or when server/client try to delete request, ftp may raise error.
# start script
python3 .github/ci_script/file_guard.py "$request_root/status" "$request_root/log" &
python3 .github/ci_script/combine_log.py "$request_root/log" "$request_root/log_list" "$request_root/sub_logs" "$request_root/status" &

wait

status=$( head -n +1 ${request_root}/status )

if [ "$status" != "success" ];then
    echo "${status}"
    exit -1
else
    echo "${status}"
    exit 0
fi
