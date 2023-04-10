#!/bin/bash
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.
CORES=`lscpu | grep 'Core(s)' | awk '{print $4}'`
SOCKETS=`cat /proc/cpuinfo | grep 'physical id' | sort | uniq | wc -l`
total_cores=`expr $CORES \* $SOCKETS \* 2`
start_core_idx=0
stop_core_idx=`expr $total_cores - 1`
cores_range=${start_core_idx}'-'${stop_core_idx}
head_address='0.0.0.0'
head_name='ray-head'
INDEXING_APPLICATION_DIR=/home/user/open-domain-question-and-answer/applications/indexing
WORKSPACE_DIR=/home/user/workspace
CUSTOMER_DIR=/home/user/data
DATASET_DIR=/home/user/dataset
PREPARE_ENV_FILE=prepare_env.sh
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'



usage() {
  echo "Usage: $0 -r [run_type] [optional parameters]"
  echo "  options:"
  echo "    -h Display usage"
  echo "    -r run_type"
  echo "         Run type = [startup_head, startup_worker, startup_database, exec_pipeline, stop_ray, clean_ray, stop_db, clean_db, clean_all]"
  echo "         The recommendation is a single instance using no more than a single socket."
  echo "    -a head_address"
  echo "         Ray head address (127.0.0.1)"
  echo "    -c cores_range"
  echo "         Cores range for ray containers"
  echo "    -m mkldnn_verbose"
  echo "         MKLDNN_VERBOSE value"
  echo "    -d database"
  echo "         startup the database=[postgresql, elasticsearch]"
  echo "    -f dataset"
  echo "         folder path of mounting to docker container"
  echo "    -w workspace"
  echo "         folder path of workspace which includes you source code"
  echo "    -b database_data"
  echo "         data folder path of mounting database container"
  echo "    -u user"
  echo "         user name for access worker server"
  echo "    -p password"
  echo "         password for access worker server"
  echo "    -i image"
  echo "         docker image for head, worker or database"
  echo "    -s worker_ip"
  echo "         worker ip or hostname for access remote worker"
  echo "    -l pipeline"
  echo "         pipeline for executing"
  echo "    -g custom_dir"
  echo "         customer's data folder for saving data or model"
  echo "    -t enable_sampling_limit"
  echo "         retrieve 500 samples for indexing"
  echo ""
  echo "  examples:"
  echo "    Startup the ray head"
  echo "      $0 -r startup_head -c 10"
  echo ""
}

while getopts "h?r:a:u:p:c:m:d:f:w:b:i:l:g:s:t:" opt; do
    case "$opt" in
    h|\?)
        usage
        exit 1
        ;;
    r)  run_type=$OPTARG
        ;;
    a)  head_address=$OPTARG
        ;;
    c)  cores_range=$OPTARG
        ;;
    m)  verbose=$OPTARG
        ;;
    d)  database=$OPTARG
        ;;
    f)  dataset=$OPTARG
        ;;
    w)  workspace=$OPTARG
        ;;
    b)  database_data=$OPTARG
        ;;
    i)  image=$OPTARG
        ;;
    u)  user=$OPTARG
        ;;
    p)  password=$OPTARG
        ;;
    s)  worker_ip=$OPTARG
        ;;
    l)  pipeline=$OPTARG
        ;;
    g)  custom_dir=$OPTARG
        ;;
    t)  enable_sampling_limit=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift


## Override default values for values specified by the user


if [ ! -z "$head_address" ]; then
  head_address=$head_address
fi

if [ ! -z "$cores_range" ]; then
  cores_range=$cores_range
fi

if [ ! -z "$verbose" ]; then
    export MKLDNN_VERBOSE=$verbose
fi

if [ ! -z $database ]; then
    database=$database
fi

if [ ! -z $dataset ]; then
    dataset=$dataset
fi

if [ ! -z $workspace ]; then
    workspace=$workspace
fi

if [ ! -z $database_data ]; then
    database_data=$database_data
fi

if [ ! -z $image ]; then
    image=$image
fi

if [ ! -z $user ]; then
    user=$user
fi

if [ ! -z $password ]; then
    password=$password
fi

if [ ! -z $worker_ip ]; then
    worker_ip=$worker_ip
fi

if [ ! -z $pipeline ]; then
    pipeline=$pipeline
fi

if [ ! -z $custom_dir ]; then
    custom_dir=$custom_dir
fi

if [ ! -z $enable_sampling_limit ]; then
    enable_sampling_limit=$enable_sampling_limit
fi
check_dirs() {
    if [ -z $dataset ]; then
        echo -e "${RED} Error: Didn't set the dataset directory!${NC}"
        exit 2
    else
        if [ ! -d $dataset ]; then
            echo -e "${RED} Error: Dataset directory is not found!${NC}"
            exit 2
        fi
    fi

    if [ -z $workspace ]; then
        echo -e "${RED} Error: Didn't set the workspace directory!${NC}"
        exit 2
    else
        if [ ! -d $workspace ]; then
            echo -e "${RED} Error: Workspace directory is not found!${NC}"
            exit 2
        fi
    fi

    if [ -z $custom_dir ]; then
        echo -e "${RED} Error: Didn't set the custom_dir directory!${NC}"
        exit 2
    else 
        if [ ! -d $custom_dir ]; then
            echo -e "${RED} Warning: Customer's directory is not found!${NC}"
            echo -e "${GREEN} Create the ${custom_dir} for customer's data folder with $user ownership!${NC}" 
            mkdir -p -m777 $custom_dir
        else
            chmod 777 $custom_dir
        fi
    fi
}

prepare_envs () {
    prepare_file=$workspace'/'$PREPARE_ENV_FILE
    prepare_file_on_container=$WORKSPACE_DIR'/'$PREPARE_ENV_FILE
    echo -e "${GREEN} Prepare the executing enviroment for $1 !${NC}"

    if [ ! -f $prepare_file ]; then
        echo -e "${RED} Warning: $prepare_file is not found ! ${NC}"
    else
        echo -e "${GREEN} chmod $prepare_file!${NC}"
        chmod +x $prepare_file
        if [[ $2 = "head" ]]; then
            echo -e "${GREEN} Execute the prepare_env.sh on $1 !${NC}"
            docker exec -i $1 /bin/bash -c "${prepare_file_on_container}"
        else
            echo -e "${GREEN} Execute the prepare_env.sh on $1 !${NC}"
            sshpass -p $password ssh -o StrictHostKeychecking=no $user@$worker_ip bash << EOF
            docker exec -i $1 /bin/bash -c "${prepare_file_on_container}" 
EOF
        fi
    fi

}



post_fix=`date +%Y%m%d`'-'`date +%s`

if [[ $run_type = "startup_head" ]]; then
    echo -e "${GREEN} Startup the Ray head with ${cores_range} cores on ${head_address} !${NC}" 
    check_dirs
    
    docker run -itd --network host \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e PYTHONWARNINGS=ignore \
        -v ${workspace}:${WORKSPACE_DIR} \
        -v ${dataset}:${DATASET_DIR} \
        -v ${custom_dir}:${CUSTOMER_DIR} \
        --cpuset-cpus=${cores_range} \
        --shm-size=64gb --name $head_name ${image} /bin/bash  &
    sleep 5
    prepare_envs $head_name "head"
    docker exec -i $head_name /bin/bash -c "ray start --node-ip-address=${head_address} --head --dashboard-host='0.0.0.0' --dashboard-port=8265"

elif [[ $run_type = "startup_database" ]]; then
    echo -e "${GREEN} Startup the ${database} database with ${cores_range} cores on ${head_address}!${NC}"


    if [[ $database = "elasticsearch" ]]; then
        es_name='ray-elasticsearch-'${post_fix}
        if [ ! -z $database_data ]; then
            if [ ! -d $database_data ]; then
                echo -e "${GREEN} Create the ${database_data} for ${database} data folder!${NC}" 
                mkdir -p -m777 $database_data
            fi
            docker run -d --name $es_name  --network host --cpuset-cpus=${cores_range} --shm-size=8gb -e "discovery.type=single-node" \
                -v ${database_data}:/usr/share/elasticsearch/data \
                -e ES_JAVA_OPTS="-Xmx8g -Xms8g" ${image}
        else
            docker run -d --name $es_name  --network host --cpuset-cpus=${cores_range} --shm-size=8gb -e "discovery.type=single-node" \
                -e ES_JAVA_OPTS="-Xmx8g -Xms8g" ${image}
        fi
    fi

    if [[ $database = "postgres" ]]; then
        postgres_name='ray-postgres-'${post_fix}
        if [ ! -z $database_data ]; then
            if [ ! -d $database_data ]; then
                echo -e "${GREEN} Create the ${database_data} for ${database} data folder!${NC}" 
                mkdir -p -m777 $database_data
            fi
            docker run --network host -d --name $postgres_name -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
                -v ${database_data}:/var/lib/postgresql/data --cpuset-cpus=${cores_range} ${image}
        else
            docker run --network host -d --name $postgres_name -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
                --cpuset-cpus=${cores_range} ${image}
        fi
        sleep 5
        docker exec -d $postgres_name psql -U postgres -c "CREATE DATABASE haystack;"
    fi



elif [[ $run_type = "startup_worker" ]]; then
    echo "cores_range = ${cores_range}"

    worker_name='ray-worker-'${post_fix}

    head_address=${head_address}':6379'
    echo -e "${GREEN} Access ${worker_ip} and startup the Ray on ${cores_range} cores!${NC}"
    sshpass -p $password ssh -o StrictHostKeychecking=no $user@$worker_ip bash << EOF
    docker run -itd \
        -v ${workspace}:${WORKSPACE_DIR} \
        -v ${dataset}:${DATASET_DIR} \
        -v ${custom_dir}:${CUSTOMER_DIR} \
        -e http_proxy=${http_proxy} \
        -e https_proxy=${https_proxy} \
        -e PYTHONWARNINGS=ignore \
        --cpuset-cpus=${cores_range} --network host \
        --shm-size=10.24gb \
        --name $worker_name  ${image}  /bin/bash &
EOF

    sleep 5
    prepare_envs $worker_name "worker"
    echo -e "${GREEN} The worker node(${worker_ip}) connects to ${head_address} ${NC}"
    sleep 5
    sshpass -p $password ssh -o StrictHostKeychecking=no $user@$worker_ip bash << EOF
    docker exec -i $worker_name /bin/bash -c "ray start --address=${head_address}"
EOF

elif [[ $run_type = "exec_pipeline" ]]; then
    echo -e "${GREEN} Execute pipeline of ${WORKSPACE_DIR}/${pipeline} with enable_sampling_limit=${enable_sampling_limit}${NC}"
    IFS='.'
    read -a strarr <<< "${pipeline}"
    docker exec -i $head_name /bin/bash -c "cd ${INDEXING_APPLICATION_DIR} && python indexing_pipeline.py -p ${WORKSPACE_DIR}/${pipeline} -s ${enable_sampling_limit} 2>&1|tee ${CUSTOMER_DIR}/${strarr[0]}-${post_fix}.log"

elif [[ $run_type = "stop_ray" ]]; then
    echo "Stop ray containers"
    docker stop $(docker ps -a |grep -E 'ray-head|ray-worker'|awk '{print $1 }')

elif [[ $run_type = "clean_ray" ]]; then
    echo "Clean ray containers"
    docker rm $(docker ps -a |grep -E 'ray-head|ray-worker'|awk '{print $1 }')

elif [[ $run_type = "stop_db" ]]; then
    echo "Stop elasticsearch container"
    docker stop $(docker ps -a |grep -E 'ray-elasticsearch|ray-postgres'|awk '{print $1 }')

elif [[ $run_type = "clean_db" ]]; then
    echo "Clean elasticsearch container"
    docker stop $(docker ps -a |grep -E 'ray-elasticsearch|ray-postgres'|awk '{print $1 }')
    docker rm $(docker ps -a |grep -E 'ray-elasticsearch|ray-postgres'|awk '{print $1 }')
    
elif [[ $run_type = "clean_all" ]]; then
    echo "Stop and clean ray and elasticsearch containers"
    docker stop $(docker ps -a |grep -E 'ray-head|ray-worker|ray-elasticsearch|ray-postgres'|awk '{print $1 }')
    docker rm $(docker ps -a |grep -E 'ray-head|ray-worker|ray-elasticsearch|ray-postgres'|awk '{print $1 }')
fi
