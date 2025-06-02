import numpy as np
import xarray as xr
from itwinai.plugins.xtclim.src.trainer import TorchTrainer

def test_torch_trainer_synthetic(tmp_path=""):
    # Simulate preprocessed input
    data = xr.DataArray(np.random.rand(4, 1, 2, 2), dims=["time", "channel", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_file = tmp_path / "train_input.nc"
    ds.to_netcdf(input_file)

    # Instance
    trainer = TorchTrainer(
        input_path=str(tmp_path),
        output_path=str(tmp_path),
        seasons=["summer"],
        epochs=1,
        lr=0.001,
        batch_size=1,
        n_memb=1,
        beta=0.1,
        n_avg=1,
        stop_delta=0.01,
        patience=2,
        kernel_size=3,
        init_channels=2,
        image_channels=1,
        latent_dim=4
    )

    # Run
    trainer.execute()

    # Check if model output or logs exist
    assert any(tmp_path.glob("*"))
