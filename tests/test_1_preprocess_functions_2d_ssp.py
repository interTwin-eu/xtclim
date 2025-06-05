def test_preprocess_data_synthetic(tmp_path):
    import xarray as xr
    import numpy as np
    from itwinai.plugins.xtclim.preprocessing.preprocess_functions_2d_ssp import PreprocessData

    # Créer les dossiers
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    dataset_root = tmp_path / "dataset_root"
    dataset_root.mkdir()

    # Fichier NetCDF d'entrée
    data = xr.DataArray(np.random.rand(3, 2, 2), dims=["time", "lat", "lon"],
                        coords={"time": [0, 1, 2], "lat": [0.25, 0.75], "lon": [0.25, 0.75]})
    ds = xr.Dataset({"tas": data})
    input_file = input_dir / "input.nc"
    ds.to_netcdf(input_file)

    # Fichiers requis
    mask = dataset_root / "landsea_mask.nc"
    mask_ds = ds["tas"].isel(time=0).to_dataset(name="mask")
    mask_ds.attrs["variable_id"] = "mask"
    mask_ds.to_netcdf(mask)

    extr = tmp_path / "scenario_extr.nc"
    ds.to_netcdf(extr)

    # Fichier SSP dans dataset_root
    ssp_file = dataset_root / "ssp1.nc"
    ds.to_netcdf(ssp_file)

    # Exécuter l'étape
    step = PreprocessData(
        dataset_root=str(dataset_root),
        input_path=str(input_dir),
        output_path=str(tmp_path),
        histo_extr=str(input_file),
        landsea_mask="landsea_mask.nc",
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=["ssp1"],
        scenario_extr=str(extr),
    )

    step.execute()
