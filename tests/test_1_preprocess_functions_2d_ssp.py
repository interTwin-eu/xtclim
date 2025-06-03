import numpy as np
import xarray as xr
from itwinai.plugins.xtclim.preprocessing.preprocess_functions_2d_ssp import PreprocessData

def test_preprocess_data_synthetic(tmp_path):
    # Créer un sous-dossier d'entrée
    input_dir = tmp_path / "input_dir"
    input_dir.mkdir()

    # Écrire un fichier NetCDF dans ce dossier
    data = xr.DataArray(np.random.rand(3, 2, 2), dims=["time", "lat", "lon"],
                        coords={"time": [0, 1, 2], "lat": [0.25, 0.75], "lon": [0.25, 0.75]})
    ds = xr.Dataset({"tas": data})
    input_file = input_dir / "input.nc"
    ds.to_netcdf(input_file)

    # Créer des fichiers factices requis
    mask = tmp_path / "landsea_mask.nc"
    extr = tmp_path / "scenario_extr.nc"
    ds["tas"].isel(time=0).to_dataset(name="mask").to_netcdf(mask)
    ds.to_netcdf(extr)

    # Instancier
    step = PreprocessData(
        dataset_root="unused",
        input_path=str(input_dir),
        output_path=str(tmp_path),
        histo_extr=str(input_file),
        landsea_mask=str(mask),
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=["ssp1"],
        scenario_extr=str(extr),
    )

    # Exécuter
    step.execute()

    # Vérifier que des fichiers sont produits
    output_files = list(tmp_path.glob("*.nc"))
    assert output_files, "Aucun fichier NetCDF n'a été généré en sortie."
