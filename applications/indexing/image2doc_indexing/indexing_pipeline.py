from haystack.pipelines import RayIndexingPipeline

def run_indexing_pipeline():
    pipeline = RayIndexingPipeline.load_from_yaml(path="/home/user/indexing/image2doc_indexing/img2doc_dpr_indexing_pipeline.yml")
    pipeline.run()

run_indexing_pipeline()
