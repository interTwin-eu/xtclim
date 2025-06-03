def test_split_preprocessed_data_synthetic(tmp_path):
    import xarray as xr
    import numpy as np
    from itwinai.plugins.xtclim.preprocessing.preprocess_2d_seasons import SplitPreprocessedData

    # Créer un fichier NetCDF synthétique
    data = xr.DataArray(np.random.rand(10, 2, 2), dims=["time", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_file = tmp_path / "preprocessed.nc"
    ds.to_netcdf(input_file)

    # Lancer le split
    step = SplitPreprocessedData(
        input_file=str(input_file),
        output_folder=str(tmp_path),
        input_vars=["tas"]
    )
    step.execute()
