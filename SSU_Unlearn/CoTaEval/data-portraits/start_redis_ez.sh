##!/bin/bash
#
#port=${1:-8899}
#
#base_dir="./instances"
#datestamp=$(date +"%Y-%m-%d_%H-%M-%S")
#run_name="${datestamp}"_"$RANDOM"
#
#echo $run_name
#run_dir=$base_dir/$run_name
#echo $run_dir
#
## copy config and start redis
#mkdir -p $run_dir
## cp $conf_file $run_dir
#cd $run_dir
#
#echo "the port is $port"
#echo "Run name: $run_name"
#echo "Base directory: $base_dir"
#echo "Run directory: $run_dir"
#echo "RedisBloom module path: $re_bloom"
#echo "Redis configuration file: $conf_file"
#echo "Current working directory after cd: $(pwd)"
#
##redis-server $conf_file --port $port --daemonize yes --loadmodule $re_bloom
#redis-server --loadmodule $re_bloom --daemonize yes --port $port
#echo "test server has started"
#
##sleep 2 # give redis a chance to start
#
#if [ -e "redis.pid" ]; then
#    echo "pid file exists, redis was started!" 1>&2
#    echo $HOSTNAME:$port
#    cd ../../
#    exit 0
#fi
#
#echo "Failed to start, check log in $run_dir"
#cd ../../
#exit 1


port=${1:-8899}

base_dir="./instances"
datestamp=$(date +"%Y-%m-%d_%H-%M-%S")
run_name="${datestamp}"_"$RANDOM"
re_bloom="./SSU_Unlearn/CoTaEval/data-portraits/redis-stable/RedisBloom/bin/linux-x64-release/redisbloom.so"
conf_file="./SSU_Unlearn/CoTaEval/data-portraits/redis.conf"

#echo $run_name
run_dir=$base_dir/$run_name
#echo $run_dir

# copy config and start redis
mkdir -p $run_dir
# cp $conf_file $run_dir
cd $run_dir

redis-server $conf_file --port $port --daemonize yes --loadmodule $re_bloom
sleep 2 # give redis a chance to start

echo "Current working directory after cd: $(pwd)"
if [ -e "redis.pid" ]
then
    echo "pid file exists, redis was started!" 1>&2
    echo $HOSTNAME:$default_port
    exit 0
fi

echo "Failed to start, check log in $run_dir"
exit 1