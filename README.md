### Introduction
This repository is an open domain question answering(ODQA) framework developed and maintained by Intel. It is based on the work of [haystack](https://haystack.deepset.ai/overview/intro), and you can find more detailed information on their [github](https://github.com/deepset-ai/haystack).
We provide three Intel Optimized pipelines, and setup a benchmark test tool to compare the performance and accuracy of these pipelines. With the help of this benchmark tool, we hope to get the best throughput and latency trade-off for each pipeline on different Intel CPU platforms.

### Dependencies
##### Install Docker and Docker Compose
**Note:** If you have docker and docker-compose installed on your machine, then skip this.

```bash
# change to sudo privileges
sudo su
# run shell script (support Red Hat Linux)
./prepare_env.sh
```

### Run Optimized Pipelines
#### Step-by-step Instructions
##### 1. Download the source code
```bash
git clone https://github.com/intel/open-domain-question-and-answer.git
cd open-domain-question-and-answer/
git checkout -b master origin/master
git submodule update --init --recursive
```

##### 2. Prepare work
##### Set proxy
docker container need to download model from [Huggingface](https://huggingface.co/) and install related dependencies from Internet, hence we may need to set environment param of proxy for it. Here we map HTTP_PROXY and HTTPS_PROXY from host to the docker container. So please set correct environment param for HTTP_PROXY and HTTPS_PROXY on the host machine. 
##### Prepare the searching database and faiss indexing files.
Please refer to [applications/Indexing](https://github.com/intel/open-domain-question-and-answer/tree/main/applications/indexing)

##### Modify the config
 After executing the previous step(examples/indexing), you should get the colbert model and faiss indexing files.
 - For pipeline 2, set the ENV params "HOST_SOURCE" of config/env.stackoverflow* or config/env.marco* files to the absolute model path you placed
 - For pipeline 3, set the ENV params "HOST_SOURCE" of config/env.stackoverflow* or config/env.marco* files to the absolute postsql path where you store indexing   
 
##### 3. Run Demo Commands

Go to applications/odqa-pipelines 
```bash
cd applications/odqa_pipelines
```

 **Note:** 
 Add the argument “-r 1” in the command, if it is first run.

- Pipeline1: ElasticsearchDocumentStore->EmbeddingRetriever(deepset/sentence_bert)->Docs2Answers

    **GPU:**
    ```bash
    #stackoverflow database
    ./launch_pipeline.sh -r 1 -p emr_faq -d gpu -n 0 -e stackoverflow  
    #marco database
    ./launch_pipeline.sh -r 1 -p emr_faq -d gpu -n 0 -e marco
   ``` 
    **CPU:**
    ```bash
    #stackoverflow database
    ./launch_pipeline.sh -r 1 -p emr_faq -d cpu -n 0 -e stackoverflow  
    #marco database
    ./launch_pipeline.sh -r 1 -p emr_faq -d cpu -n 0 -e marco
    ```
    
- Pipeline2: ElasticsearchDocumentStore->BM25Retriever->ColbertRanker-> Docs2Answers  
    prepare colBert model:
    ```bash
    # download the colbert model
    cd ../../
    mkdir data
    wget https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/colbertv2.0.tar.gz 
    tar -xvzf colbertv2.0.tar.gz
    mv colbertv2.0/* data/
    #or you can set HOST_SOURCE in config/env.marco.esds_bm25r_colbert file to where you place the model
    cd applications/odqa_pipelines
    ```

    **GPU:**
    ```bash
    #marco database
    ./launch_pipeline.sh -r 1 -p colbert_faq -d gpu -n 0 -e marco 
    ```
    **CPU:**
    ```bash
    ./launch_pipeline.sh -r 1 -p colbert_faq -d cpu -n 0 -e marco 
    ```
- Pipeline3：FAISSDocumentStore->DPR→Docs2Answers  
    prepare faiss indexed file for stackoverflow:
    ```bash
    cd faiss_data/stackoverflow/
    cat faiss-index-so.faiss.parta* > faiss-index-so.faiss
    cd ../../
    # change HOST_SOURCE in file config/env.stackoverflow.faiss_dpr to the path of faiss_data/stackoverflow/
    ```
    prepare faiss indexed file for marco:
    ```bash
    cd faiss_data/marco/
    cat faiss-index-so.faiss.parta* > faiss-index-so.faiss
    cd ../../
    # change HOST_SOURCE in file config/env.marco.faiss_dpr to the path of faiss_data/marco/
    ```
    
    **GPU:**
    ```bash
    #stackoverflow database
    ./launch_pipeline.sh -r 1 -p faiss_faq -d gpu -n 0 -e stackoverflow  
    #marco database
    ./launch_pipeline.sh -r 1 -p faiss_faq -d gpu -n 0 -e marco
    ```
    **CPU:**
    ```bash
    #stackoverflow database
    ./launch_pipeline.sh -r 1 -p faiss_faq -d cpu -n 0 -e stackoverflow  
    #marco database
    ./launch_pipeline.sh -r 1 -p faiss_faq -d cpu -n 0 -e marco
    ```

### Benchmark tool
We provide a script to caculate the accuracy or throughput of the pipelines we list in above section. After executing docker-compose up command line listed in above section, there should be backend worker wating for processing requests from other side.  And then we can run benchmark script.
#### 1.Preparation
start pipelines to serve as backend service. And then you need to stop the firewall by input
```bash
sudo service firewalld stop
``` 
or open related ports
```bash
sudo firewall-cmd --zone=public --permanent --add-port={8000,9200,5432}/tcp
```

#### 2.Usage

```bash
python3 benchmark.py --help
```
And it will return prompts as:
```bash
usage: benchmark.py [-h] [-p PROCESSES] [-n QUERY_NUMBER] [-m {0,1}] [-b BS] [-a {0,1}] [-c {0,1}][-t TOPK] [-ip IP_ADDRESS]

multi-process benchmark for haystack...

optional arguments:
  -h, --help       show this help message and exit
  -p PROCESSES     How many processes are used for the process pool
  -n QUERY_NUMBER  How many querys will be executed.
  -m {0,1}         Which pipeline will be tested. 0:colbert; 1:emr or faiss
  -b BS            batch size for DPR
  -a {0,1}         Is it an accuracy benchmark
  -c {0,1}         Use the real concurrent
  -t TOPK          Retriever and Ranker topk
  -ip IP_ADDRESS   Ip address of backend server
```
#### 3.Examples
- caculate accuracy:  
  You need to replace 'queries_file' and 'qrel_file' in benchmark.py, where 'queries_file' records lists of re-paraphrased query text in validation set, and 'qrel_file' stores relevant answer id in the validtion set. And then you can type in the terminal:
  ```bash
  python benchmark.py -a 1 -ip 127.0.0.1
  ```
  As we can see from "Usage" part, "-a 1" means we use accuracy benchmark, and "-ip 127.0.0.1" means the backend of this odqa pipeline is run on localhost. You can also specify the "-ip" param to another machine where runs the pipelines listed above

- caculate throughput and latency  
  ```bash
  python benchmark.py -m 0 -p 5 -n 100 -c 1 -ip $backend_ip
  ```
  where "-m 0" denotes the pipeline run in backend server is Pipeline 2, and "-p 5" denotes we will start 5 processes to do parallel requests, and "-n 100" means the requests number in total is 100

### 
