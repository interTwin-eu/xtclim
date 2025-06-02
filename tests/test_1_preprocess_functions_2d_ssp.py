import numpy as np
import xarray as xr
from itwinai.plugins.xtclim.preprocessing.preprocess_functions_2d_ssp import PreprocessData

def test_preprocess_data_synthetic(tmp_path):
    # Create synthetic input NetCDF
    data = xr.DataArray(np.random.rand(3, 2, 2), dims=["time", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_path = tmp_path / "input.nc"
    ds.to_netcdf(input_path)

    # Create instance
    step = PreprocessData(
        dataset_root="unused",
        input_path=str(input_path),
        output_path=str(tmp_path),
        histo_extr="dummy",
        landsea_mask="dummy",
        min_lon=0,
        max_lon=1,
        min_lat=0,
        max_lat=1,
        scenarios=["ssp1"],
        scenario_extr="dummy",
    )

    # Run and check output
    step.execute()
    assert any(tmp_path.glob("*.nc"))
