import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from dpnlp_lib.src.builder import run_builder
from dpnlp_lib.src.server import run_flower_server

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    run_builder(cfg)
    run_flower_server(cfg, strategy_obj=instantiate(cfg.algorithm))

if __name__ == "__main__":
    main()