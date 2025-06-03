def test_torch_inference_synthetic(tmp_path):
    import xarray as xr
    import numpy as np
    from itwinai.plugins.xtclim.src.trainer import TorchInference

    # Données pour l'inférence
    data = xr.DataArray(np.random.rand(4, 1, 2, 2), dims=["time", "channel", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_file = tmp_path / "train_input.nc"
    ds.to_netcdf(input_file)

    step = TorchInference(
        input_path=str(input_file),
        output_path=str(tmp_path),
        input_vars=["tas"]
    )
    step.execute()
