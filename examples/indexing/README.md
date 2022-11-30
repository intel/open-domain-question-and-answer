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

    URL: https://microsoft.github.io/msmarco/
    download the Question Answering V2.1 into ./

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
 python gen-marco-ray.py -m train -d emr -p $host_ip
```
Generate elasticsearch database of marco with colbert embedding.
```bash
 python gen-marco-ray.py -m train -d colbert -p $host_ip
```
Generate elasticsearch database of stackoverflow with EMR embedding.
```bash
 python gen-stack-ray.py -d emr -p $host_ip
```

Generate elasticsearch database of stackoverflow with EMR embedding.(TODO)

 **Note:**
 After generating the database, commit the Elasticsearch container immediately.

```bash
docker commit ${elasticsearch_container_name or hash code}  odqa:es-colbert-macro
```  
If you want to use the examples/odqa_pipelines to run the demos, name the docker images as following. They are default names in the demo config files, or you need to modify the config files.


    odqa:es-colbert-marco
    odqa:es-emr-marco
    odqa:es-emr-stackoverflow

### Generate the Faiss database and indexing files(TODO)




