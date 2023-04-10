from ray_indexing_pipeline import RayIndexingPipeline
import argparse, time, os

def run_indexing_pipeline(cfg):
    if cfg.enable_sample == 1:
        os.environ["ENABLE_SAMPLING_LIMIT"] = "1"
    else:
        os.environ["ENABLE_SAMPLING_LIMIT"] = "0"

    start = time.time()
    pipeline = RayIndexingPipeline.load_from_yaml(path=cfg.pipeline_yaml)
    pipeline.run()
    cost = time.time() - start
    print(f'Spent {cost}s for pipeline: {cfg.pipeline_yaml}')


def parse_cmd():
    desc = 'generate documentstore for marco dataset...\n\n'
    args = argparse.ArgumentParser(description=desc, epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-p', type=str, default='faiss_indexing_pipeline.yml', dest='pipeline_yaml', help='pipeline config file')
    args.add_argument('-s', type=int, default=0, dest='enable_sample', help='Only retrieve 500 samples for indexing')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    run_indexing_pipeline(config)
