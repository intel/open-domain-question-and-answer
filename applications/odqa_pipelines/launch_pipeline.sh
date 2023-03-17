#!/bin/bash
# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.
rebuild='1'
nginx='0'
pipeline='emr_faq'
device='cpu'
database='marco'

usage() {
  echo "Usage: $0 -p [pipeline] [optional parameters]"
  echo "  options:"
  echo "    -h Display usage"
  echo "    -p pipeline"
  echo "         pipelines = [emr_faq, faiss_faq, colbert_faq, colbert_opt_faq]"
  echo "    -r rebuild"
  echo "         rebuild the images [1 : yes, 0 : no] "
  echo "    -d device"
  echo "         devices = [cpu, gpu]"
  echo "    -n nginx"
  echo "         Use nginx for load balance [1 : yes, 0 : no]"
  echo "    -e database"
  echo "         Use database [stackoverflow, marco] for searching"
  echo ""
  echo "  examples:"
  echo "    Run emr_faq pipeline on CPU "
  echo "      $0 -r 1 -d cpu -n 0 -p emr_faq"
  echo ""
}

while getopts "h?r:d:n:p:e:" opt; do
    case "$opt" in
    h|\?)
        usage
        exit 1
        ;;
    r)  rebuild=$OPTARG
        ;;
    n)  nginx=$OPTARG
        ;;
    p)  pipeline=$OPTARG
        ;;
    d)  device=$OPTARG
        ;;
    e)  database=$OPTARG
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift


## Override default values for values specified by the user
if [ ! -z "$rebuild" ]; then
    rebuild=$rebuild
fi


if [ ! -z "$device" ]; then
  device=$device
fi

if [ ! -z "$pipeline" ]; then
  pipeline=$pipeline
fi

if [ ! -z "$nginx" ]; then
  nginx=$nginx
fi

if [ ! -z "$database" ]; then
  database=$database
fi

config='config/env.stackoverflow.esds_emr_yml_faq'
build=''
yaml_file='docker-compose.yml'

if [[ $pipeline = "emr_faq" ]]; then
    config='config/env.'${database}'.esds_emr_faq'

elif [[ $pipeline = "faiss_faq" ]]; then
    config='config/env.'${database}'.faiss_dpr'
    yaml_file='docker-compose-dpr.yml'

elif [[ $pipeline = "colbert_faq" ]]; then
    config='config/env.'${database}'.esds_bm25r_colbert'
    if [[ $database = "stackoverflow" ]]; then
        echo "Cannot support ${pipeline} with ${database}, need the fine-tuned colbert model with ${database}"
        exit 0    
    fi 
fi

if [[ $rebuild = "1" ]]; then
    echo "rebuild docker images"
    build='--build'
fi
echo "device = ${device}"
if [[ $device = "gpu" ]]; then
    yaml_file='docker-compose-gpu.yml'
    if [[ $pipeline = "faiss_faq" ]]; then
        yaml_file='docker-compose-gpu-dpr.yml'
    fi
fi

if [[ $nginx = "1" ]]; then
    echo "use the nginx for load balance, only CPU mode supported!"
    yaml_file='docker-compose-nginx.yml'
fi

echo "run the ${pipeline} with ${database} on ${device}"
yaml_file='docker-compose/'$yaml_file

docker-compose --env-file $config -f $yaml_file up $build
