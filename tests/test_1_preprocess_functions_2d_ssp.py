import numpy as np
import xarray as xr
import os
from itwinai.plugins.xtclim.preprocessing.preprocess_functions_2d_ssp import PreprocessData

def test_preprocess_data_synthetic(tmp_path):
    # Créer un dossier d'entrée
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    # Créer fichier NetCDF synthétique dans ce dossier
    data = xr.DataArray(np.random.rand(3, 2, 2), dims=["time", "lat", "lon"],
                        coords={"time": [0, 1, 2], "lat": [0.25, 0.75], "lon": [0.25, 0.75]})
    ds = xr.Dataset({"tas": data})
    input_file = input_dir / "input.nc"
    ds.to_netcdf(input_file)

    # Fichier fictif pour histo_extr, landsea_mask, scenario_extr (réutilise input)
    landsea_mask = tmp_path / "mask.nc"
    ds["tas"].isel(time=0).to_dataset(name="mask").to_netcdf(landsea_mask)
    scenario_extr = tmp_path / "scenario.nc"
    ds.to_netcdf(scenario_extr)

    # Créer instance
    step = PreprocessData(
        dataset_root="unused",
        input_path=str(input_dir),  # ✅ maintenant un dossier
        output_path=str(tmp_path),
        histo_extr=str(input_file),
        landsea_mask=str(landsea_mask),
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=["ssp1"],
        scenario_extr=str(scenario_extr),
    )

    # Exécuter
    step.execute()
    output_files = list(tmp_path.glob("*.nc"))
    assert output_files, "Aucun fichier de sortie n'a été généré."
