import xarray as xr
import numpy as np
from itwinai.plugins.xtclim.preprocessing.preprocess_2d_seasons import SplitPreprocessedData

def test_split_preprocessed_data_synthetic(tmp_path):
    # Create synthetic input NetCDF
    data = xr.DataArray(np.random.rand(10, 2, 2), dims=["time", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    input_file = tmp_path / "preprocessed.nc"
    ds.to_netcdf(input_file)

    # Instance
    step = SplitPreprocessedData(
        input_path=str(tmp_path),
        n_memb=2,
        scenarios=["ssp1"]
    )

    # Run
    step.execute()  

    # Check if files were created
    files = list(tmp_path.glob("*.nc"))
    assert len(files) > 0
