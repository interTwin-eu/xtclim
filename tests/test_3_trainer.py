def test_torch_trainer_synthetic(tmp_path):
    import xarray as xr
    import numpy as np
    from itwinai.plugins.xtclim.model.trainer import TorchTrainer

    # Données prétraitées synthétiques
    data = xr.DataArray(np.random.rand(4, 1, 2, 2), dims=["time", "channel", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_file = tmp_path / "train_input.nc"
    ds.to_netcdf(input_file)

    step = TorchTrainer(
        input_path=str(input_file),
        output_path=str(tmp_path),
        input_vars=["tas"]
    )
    step.execute()
