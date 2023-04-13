import argparse, time, os
import yaml


class Workflow :
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        workflow_config = self.read_from_yaml(cfg.workflow_yaml)
        self.cluster_config = workflow_config['nodes'] 
        self.pipelines_config = workflow_config['pipelines']


    def read_from_yaml(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Not found: {path}")
        with open(file, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
        

    def get_head_node(self) :
        head_node = None
        for node in self.cluster_config:
            if node["type"] == "head" :
                head_node = node
        return head_node
    

    def startup_ray_cluster(self) :
        for node in self.cluster_config:
            if node["type"] == "head" :
                ret = os.system(f'./run-ray-cluster.sh -r startup_head -c {node["cores"]} \
                        -f {node["dataset_dir"]} \
                        -w {node["workspace_dir"]} \
                        -g {node["customer_dir"]} \
                        -i {node["image"]} \
                        -a {node["node"]}')
                if ret == 0 :
                    print("Successfully startup the ray head!")
                else:
                    print("Startup the ray head failed!")
                    return False
                
            else:
                head_node = self.get_head_node()
                if head_node == None:
                    print("head_node cannot be None!!")
                    return False
                
                ret = os.system(f'./run-ray-cluster.sh -r startup_worker -c {node["cores"]} \
                        -f {node["dataset_dir"]} \
                        -w {node["workspace_dir"]} \
                        -g {node["customer_dir"]} \
                        -i {node["image"]} \
                        -u {node["user"]} \
                        -p {node["password"]} \
                        -s {node["node"]} \
                        -a {head_node["node"]}')
                if ret == 0 :
                    print("Successfully startup the workers!")
                else:
                    print("Startup the workers failed!")
                    return False
        return True
                
    def stop_database(self) :
        ret = os.system(f'./run-ray-cluster.sh -r stop_db')
        return ret 

    def startup_database(self, database) :
        head_node = self.get_head_node()
        if head_node == None:
            print("head_node cannot be None!!")
            return
        ret = os.system(f'./run-ray-cluster.sh -r startup_database -c {database["cores"]} \
            -b {database["data_dir"]} \
            -i {database["image"]} \
            -d {database["type"]}')
        return ret

    def exec_pipeline(self, pipeline: str) :
        head_node = self.get_head_node()
        if head_node == None:
            print("head_node cannot be None!!")
            return
        ret = os.system(f'./run-ray-cluster.sh -r exec_pipeline -l {pipeline} -t {self.cfg.enable_sample}')
        print(f'ret={ret}')       



    def run_pipelines(self) :
        
        for pipeline in self.pipelines_config :
            config_pipeline = pipeline["name"].split("/")[-1] 
            exec_pipeline = self.cfg.pipeline_yaml.split("/")[-1]
            if self.cfg.pipeline_yaml == "all" or config_pipeline == exec_pipeline :
                ret = self.stop_database()
                if ret == 0 :
                    print("Clean the database container successfully!")
                else:
                    print("There is no database to be stopped!")
                ret = self.startup_database(pipeline["database"])
                if ret == 0 :
                    print("Startup the database container successfully!")
                    self.exec_pipeline(config_pipeline)
                else:
                    print("Failed to startup the database containers")

    

def parse_cmd():
    desc = 'generate documentstore for marco dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-w', type=str, default='', dest='workflow_yaml', help='workflow config file')
    args.add_argument('-p', type=str, default='', dest='pipeline_yaml', help='pipeline name')
    args.add_argument('-s', type=str, default='0', dest='enable_sample', help='Only retrieve 500 samples for indexing')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    workflow= Workflow(config)
    if config.pipeline_yaml == '':
        workflow.startup_ray_cluster()
    else:
        workflow.run_pipelines()