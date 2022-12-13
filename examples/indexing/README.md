### Introduction
The indexing example introduces how to generate the searching database with ElasticSearch and PostgreSQL.  

### Dependencies
#### 1. Build the haystack-ray docker image
```bash
# download the colbert model and build the haystack-ray image
cd ../../
mkdir data
wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz 
tar -xvzf colbertv2.0.tar.gz
cp -rf ./colbertv2.0/* ./data/
docker build -f Dockerfile-ray . -t haystack-ray
```
#### 2. download the stackoverflow and Microsoft Marco dataset
 **Note:** 
 Current directory(examples/indexing) will be mount to /home/user/indexing of haystack-ray container.
 
Stackoverflow:

    URL: https://www.kaggle.com/datasets/stackoverflow/stacksample
    download into ./stack-overflow/. 
Microsoft Marco:

    URL: https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
    download the train_v2.1.json.gz into ./marco/.
    gunzip train_v2.1.json.gz

#### 3. Startup the Elasticsearch container and haystack-ray containers
 **Note:** 
 For generating the Faiss database and indexing files, skip this section.

For example, if your machine has 10 cores, following command will startup elasticsearch container with 4 cores and master(head) haystack-ray container with 6 cores.
 ```bash
 ./run-ray-cluster.sh -r startup_all -e 4 -c 6 -u $host_ip
```
Use the '-h' option to check the command parameters.

If you can join other machines to the master node as workers, access these machines and run the following command to startup worker containers.
```bash
# For example, the machine has 10 cores, the command will startup one container. It is recommanded to only use one container with all cores. 
 ./run-ray-cluster.sh -r startup_workers -s 10 -u $master_ip
```
Or
```bash
# For example, the machine has 10 cores, the command will startup five containers.
 ./run-ray-cluster.sh -r startup_workers -s 2 -u $master_ip
```
### Generate the stackoverflow or marco database of Elasticsearch
Access the haystack-ray master container.
```bash
 docker exec -it $head_container /bin/bash
 cd /home/user/indexing
```
Generate elasticsearch database of marco with EMR embedding.
```bash
 python gen-marco-ray.py -d emr -p $host_ip
```
Generate elasticsearch database of marco with colbert embedding.
```bash
 python gen-marco-ray.py -d colbert -p $host_ip
```
Generate elasticsearch database of stackoverflow with EMR embedding.
```bash
 python gen-stack-ray.py -d emr -p $host_ip
```

Generate elasticsearch database of stackoverflow with colbert embedding.
```bash
 python gen-stack-ray.py -d colbert -p $host_ip
```

 **Note:**
 After generating the database, commit the Elasticsearch container immediately.

```bash
docker commit ${elasticsearch_container_name or hash code}  odqa:es-colbert-marco
```  
If you want to use the examples/odqa_pipelines to run the demos, name the docker images as following. They are default names in the demo config files, or you need to modify the config files.


    odqa:es-colbert-marco
    odqa:es-emr-marco
    odqa:es-emr-stackoverflow

### Generate the Faiss database and indexing files
Pull postgres image from the docker hub.
```bash
dockr pull postgres:14.1-alpine
```

Startup the postgres container and haystack-ray container.
```bash
docker run -d --name haystack-ray --network host --shm-size=8gb -e "discovery.type=single-node" haystack-ray:latest
docker run -d --name postsql-db --network host -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
-v ${host_path}/db:/var/lib/postgresql/data -p 5432:5432 postgres:14.1-alpine
```

Create database in postgres container.
```bash
docker exec -it postsql-db /bin/bash
bash-5.1# psql --username postgres
postgres=# \password postgres  #Change the database password to ensure that it is consistent with the settings.(POSTGRES_PASSWORD=postgres)
postgres=# CREATE DATABASE haystack;
```

Access the haystack-ray container.
```bash
docker exec -it haystack-ray /bin/bash
cd /home/user/indexing
```

Generate faiss database of stackoverflow.
```bash
python gen-sods.py -d faiss
```

Generate faiss database of marco
```bash
python gen-es_marco.py -d faiss
```

After generating the database, copy the indexing files to /var/lib/postgresql/data which is mount from a host directory(${host_path}). Modify the $HOST_SOURCE value of env files to the directory path when running the faiss pipeline demo. 
