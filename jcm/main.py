import hydra
from omegaconf import DictConfig
from jcm.model import Model
from hydra.core.hydra_config import HydraConfig
from pathlib import Path

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Run Speedy Model with adjustable parameters"""
    model = Model(
        time_step=cfg.model.time_step
    )
    
    predictions = model.run(
        save_interval=cfg.model.save_interval,
        total_time=cfg.model.total_time
    )
    
    ds = predictions.to_xarray()
    
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    
    filename = "model_state.nc"
    output_path = output_dir / filename
    
    ds.to_netcdf(str(output_path))

if __name__ == "__main__":
    main()
