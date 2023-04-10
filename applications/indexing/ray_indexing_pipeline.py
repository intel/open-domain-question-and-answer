from __future__ import annotations
import inspect
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import networkx as nx
from ray.data import ActorPoolStrategy
try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

import time
from haystack.errors import PipelineError
from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
    validate_config,
)
from haystack.nodes.base import BaseComponent, RootNode
from haystack.pipelines.base import Pipeline
from haystack.schema import Document, MultiLabel
import importlib.util as iu
import copy

logger = logging.getLogger(__name__)


class RayIndexingPipeline(Pipeline):
    
    def __init__(
        self,
        address: Optional[str] = None,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
        pipeline_config: Optional[Dict] = None
    ):
        """
        :param address: The IP address for the Ray cluster. If set to `None`, a local Ray instance is started.
        :param ray_args: Optional parameters for initializing Ray.
        :param serve_args: Optional parameters for initializing Ray Serve.
        """
        ray_args = ray_args or {}
        if not ray.is_initialized():
            ray.init(address=address, **ray_args)
        else:
            logger.warning("Ray was already initialized, so reusing that for this RayPipeline.")
        super().__init__()
        self.pipeline_config = pipeline_config

    @classmethod
    def load_from_config(
        cls,
        pipeline_config: Dict,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        strict_version_check: bool = False,
        address: Optional[str] = None,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
    ):
        #validate_config(pipeline_config, strict_version_check=strict_version_check, extras="ray")

        pipeline_definition = get_pipeline_definition(pipeline_config=pipeline_config, pipeline_name=pipeline_name)
        component_definitions = get_component_definitions(
            pipeline_config=pipeline_config, overwrite_with_env_variables=overwrite_with_env_variables
        )
        pipeline = cls(address=address, ray_args=ray_args or {}, serve_args=serve_args or {}, pipeline_config=pipeline_config)

        for node_config in pipeline_definition["nodes"]:
            if pipeline.root_node is None:
                root_node = node_config["inputs"][0]
                #For indexing pipeline, the root_node should be 'File'
                if root_node in ["File"]: 
                    handle = None
                    pipeline._add_node_in_graph(handle=handle, name=root_node, component_type='File', outgoing_edges=1, inputs=[])
                else:
                    raise KeyError(f"Root node '{root_node}' is invalid. Available options is 'File'.")

            name = node_config["name"]
            component_type = component_definitions[name]["type"]
            if 'path' in component_definitions[name].keys() :
                spec = iu.spec_from_file_location("module.name", component_definitions[name]["path"])
                loader = iu.module_from_spec(spec)
                spec.loader.exec_module(loader)


            component_class = BaseComponent.get_subclass(component_type)
            is_actor = bool(component_definitions[name]["actor"])

            handle = None
            serve_deployment_kwargs = next(node for node in pipeline_definition["nodes"] if node["name"] == name).get(
                "serve_deployment_kwargs", {}
            )
            
            if is_actor == False:
                component_params = component_definitions[name]["params"]
                handle = BaseComponent._create_instance(
                    component_type=component_type, component_params=component_params, name=name
                ) 
            
            pipeline._add_node_in_graph(
                handle=handle,
                name=name,
                component_type=component_type,
                outgoing_edges=component_class.outgoing_edges,
                inputs=node_config.get("inputs", []),
                serve_deployment_kwargs=serve_deployment_kwargs,
                is_actor=is_actor
            )

        return pipeline

    @classmethod
    def load_from_yaml(  # type: ignore
        cls,
        path: Path,
        pipeline_name: Optional[str] = None,
        overwrite_with_env_variables: bool = True,
        address: Optional[str] = None,
        strict_version_check: bool = False,
        ray_args: Optional[Dict[str, Any]] = None,
        serve_args: Optional[Dict[str, Any]] = None,
    ):

        pipeline_config = read_pipeline_config_from_yaml(path)
        return cls.load_from_config(
            pipeline_config=pipeline_config,
            pipeline_name=pipeline_name,
            overwrite_with_env_variables=overwrite_with_env_variables,
            strict_version_check=strict_version_check,
            address=address,
            ray_args=ray_args,
            serve_args=serve_args,
        )

    def add_node(self, component, name: str, inputs: List[str]):
        raise NotImplementedError(
            "The current implementation of RayPipeline only supports loading Pipelines from a YAML file."
        )

    def _add_node_in_graph(
        self,
        handle,
        name: str,
        component_type: str,
        outgoing_edges: int,
        inputs: List[str],
        serve_deployment_kwargs: Optional[Dict[str, Any]] = None,
        is_actor: Optional[bool] = True
    ):

        self.graph.add_node(name,
                            component=handle,
                            component_type=component_type,
                            inputs=inputs,
                            outgoing_edges=outgoing_edges,
                            actor_kwargs= serve_deployment_kwargs,
                            is_actor = is_actor)
        if len(self.graph.nodes) == 2:  # first node added; connect with Root
            self.graph.add_edge(self.root_node, name, label="output_1")
            return

        for i in inputs:
            if "." in i:
                [input_node_name, input_edge_name] = i.split(".")
                assert "output_" in input_edge_name, f"'{input_edge_name}' is not a valid edge name."
                outgoing_edges_input_node = self.graph.nodes[input_node_name]["outgoing_edges"]
                assert int(input_edge_name.split("_")[1]) <= outgoing_edges_input_node, (
                    f"Cannot connect '{input_edge_name}' from '{input_node_name}' as it only has "
                    f"{outgoing_edges_input_node} outgoing edge(s)."
                )
            else:
                outgoing_edges_input_node = self.graph.nodes[i]["outgoing_edges"]
                assert outgoing_edges_input_node == 1, (
                    f"Adding an edge from {i} to {name} is ambiguous as {i} has {outgoing_edges_input_node} edges. "
                    f"Please specify the output explicitly."
                )
                input_node_name = i
                input_edge_name = "output_1"
            self.graph.add_edge(input_node_name, name, label=input_edge_name)

    def _run_node(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:
        return ray.get(self.graph.nodes[node_id]["component"].remote(**node_input))

    def _run_node_actor(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:
        pass        

    def _get_run_node_signature(self, node_id: str):
        return inspect.signature(self.graph.nodes[node_id]["component"].remote).parameters.keys()

    def eval(self) :
        for component in self.pipeline_config["components"] :
            if component['type'] in ["ElasticsearchDocumentStore", "FAISSDocumentStore"] :
                component_params = copy.deepcopy(component["params"])
                if component['type'] == "FAISSDocumentStore" :
                    component_params = {}
                    component_params['faiss_index_path'] = component['faiss_index_path']

                print(f'componet_params= {component_params}')
                document_store = BaseComponent._create_instance(
                    component_type=component['type'], component_params=component_params, name=component['name']
                )

                documents_count = document_store.get_document_count()
                print(f'documents_count = {documents_count}')


    def run(  # type: ignore
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[Union[dict, List[dict]]] = None,
        params: Optional[dict] = None,
        debug: Optional[bool] = None,
    ):
        """
        Runs the Pipeline, one node at a time.

        :param query: The search query (for query pipelines only).
        :param file_paths: The files to index (for indexing pipelines only).
        :param labels: Ground-truth labels that you can use to perform an isolated evaluation of pipelines. These labels are input to nodes in the pipeline.
        :param documents: A list of Document objects to be processed by the Pipeline Nodes.
        :param meta: Files' metadata. Used in indexing pipelines in combination with `file_paths`.
        :param params: A dictionary of parameters that you want to pass to the nodes.
                       To pass a parameter to all Nodes, use: `{"top_k": 10}`.
                       To pass a parameter to targeted Nodes, run:
                        `{"Retriever": {"top_k": 10}, "Reader": {"top_k": 3, "debug": True}}`
        :param debug: Specifies whether the Pipeline should instruct Nodes to collect debug information
                      about their execution. By default, this information includes the input parameters
                      the Nodes received and the output they generated. You can then find all debug information in the dictionary returned by this method under the key `_debug`.
        """
        # validate the node names
        self._validate_node_names_in_params(params=params)
        root_node = self.root_node
        if not root_node:
            raise PipelineError("Cannot run a pipeline with no nodes.")
        node_id = root_node
        output = None
        next_nodes = self.get_next_nodes(node_id, stream_id=None)
        if len(next_nodes) > 0 :
            node_id = next_nodes[0]
            component_type = self.graph.nodes[node_id]["component_type"]
            component_class = BaseComponent.get_subclass(component_type)
            print(f"component_type = {component_type}, parent_name = {component_class.__base__.__name__}")
            if component_class.__base__.__name__ == "Dataset" :
                kwargs = {}
                self.graph.nodes[node_id]["component"]._dispatch_run(**kwargs)
                generator = self.graph.nodes[node_id]["component"].dataset_batched_generator()
                dataset_node = node_id
                for dataset in generator:
                    ancestor_type = self.graph.nodes[dataset_node]["component_type"]
                    next_nodes = self.get_next_nodes(dataset_node, stream_id=None)
                    output = ray.data.from_items(dataset)
                    print(output)
                    while len(next_nodes) == 1 :
                        node_id = next_nodes[0]
                        component_type = self.graph.nodes[node_id]["component_type"]
                        start = time.time()
                        if self.graph.nodes[node_id]["is_actor"]:
                            num_actors = 1
                            num_cpus = 1
                            batch_size = 10
                            resource = ray.cluster_resources()
                            all_cpus_num = int(resource['CPU'])
                    
                            if 'num_cpus' in self.graph.nodes[node_id]["actor_kwargs"] :
                                num_cpus = self.graph.nodes[node_id]["actor_kwargs"]["num_cpus"]
                            if 'batch_size' in self.graph.nodes[node_id]["actor_kwargs"] :
                                batch_size = self.graph.nodes[node_id]["actor_kwargs"]["batch_size"]
                            if 'num_replicas' in self.graph.nodes[node_id]["actor_kwargs"] :
                                num_actors = self.graph.nodes[node_id]["actor_kwargs"]["num_replicas"]
                            else:
                                num_actors = int((all_cpus_num/num_cpus)-(all_cpus_num/num_cpus/10)) #avoid the actor pending

                            input_kwargs = {'root_node':'File', 'ancestor_type': ancestor_type}
                            fn_constructor_kwargs = {'pipeline_config' : self.pipeline_config, 'component_name': node_id}
                            output = output.map_batches(_RayDeploymentWrapper, compute=ActorPoolStrategy(1, num_actors), num_cpus=num_cpus, batch_size=batch_size,
                                                fn_kwargs=input_kwargs, fn_constructor_kwargs=fn_constructor_kwargs)

                            logger.debug(output)
                        else:
                            for batch in output.iter_batches(batch_size=10000):
                                kwargs= {'documents': batch, 'root_node':'File'}
                                self.graph.nodes[node_id]["component"]._dispatch_run(**kwargs)
                            if component_type == 'FAISSDocumentStore':
                                component_definitions = get_component_definitions(pipeline_config=self.pipeline_config)
                                if 'faiss_index_path' in component_definitions[node_id].keys() :
                                    faiss_path = component_definitions[node_id]['faiss_index_path']
                                    self.graph.nodes[node_id]["component"].save(faiss_path)
                                else:
                                    logger.warning("Cannot save the faiss indexing files, the faiss file path is None!")
                        cost = time.time() - start
                        logger.info(f'node_id:{node_id}, spent time :{cost}')
                        ancestor_type = self.graph.nodes[node_id]["component_type"]
                        next_nodes = self.get_next_nodes(node_id, stream_id=None)

            else:
                raise PipelineError("Cannot run a pipeline with no Dataset node.")

        self.eval()
        return output



    def send_pipeline_event(self, is_indexing: bool = False):
        """To avoid the RayPipeline serialization bug described at
        https://github.com/deepset-ai/haystack/issues/3970"""
        pass


class _RayDeploymentWrapper:
    """
    Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

    In case of Haystack, some Components like Retrievers have complex init methods that needs objects
    like Document Stores.

    This wrapper class encapsulates the initialization of Components. Given a Component Class
    name, it creates an instance using the YAML Pipeline config.
    """

    node: BaseComponent

    def __init__(self, pipeline_config: dict, component_name: str):
        """
        Create an instance of Component.

        :param pipeline_config: Pipeline YAML parsed as a dict.
        :param component_name: Component Class name.
        """
        if component_name in ["Query", "File"]:
            self.node = RootNode()
        else:
            self.node = self.load_from_pipeline_config(pipeline_config, component_name)

    def __call__(self, *data, **kwargs):
        """
        Ray calls this method which is then re-directed to the corresponding component's run().
        """
        ancestor_type = kwargs.get("ancestor_type") or ""
        if ancestor_type == "Dataset":
            kwargs['file_paths'] = data[0]
        else:
            kwargs['documents'] = data[0]
        
        
        try:
            docs, _ = self.node._dispatch_run(**kwargs)
            if 'documents' in docs.keys():
                return docs['documents']
            else:
                return []
        except Exception as err:
            logger.error(f"exception: {err=}, {type(err)=}")
            return [] 

    @staticmethod
    def load_from_pipeline_config(pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        all_component_configs = pipeline_config["components"]
        all_component_names = [comp["name"] for comp in all_component_configs]
        component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
        component_params = component_config["params"]
        
        #It is customer node for yaml pipeline, need to import dynamically.
        if 'path' in component_config.keys() :
            spec = iu.spec_from_file_location("module.name", component_config["path"])
            loader = iu.module_from_spec(spec)
            spec.loader.exec_module(loader)

        for key, value in component_params.items():
            if value in all_component_names:  # check if the param value is a reference to another component
                component_params[key] = _RayDeploymentWrapper.load_from_pipeline_config(pipeline_config, value)
        component_instance = BaseComponent._create_instance(
            component_type=component_config["type"], component_params=component_params, name=component_name
        )
        logger.debug(f'component_instance:{component_instance.__dict__}')
        return component_instance
