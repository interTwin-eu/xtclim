from itwinai.plugins.xtclim.src.trainer import TorchInference

def test_torch_inference_synthetic(tmp_path):
    # Simulated inference input
    data = xr.DataArray(np.random.rand(4, 1, 2, 2), dims=["time", "channel", "lat", "lon"])
    ds = xr.Dataset({"tas": data})
    ds.to_netcdf(tmp_path / "train_input.nc")

    # Instance
    infer = TorchInference(
        input_path=str(tmp_path),
        output_path=str(tmp_path),
        scenarios=["ssp1"],
        seasons=["summer"],
        n_memb=1,
        on_train_test=True,
        kernel_size=3,
        init_channels=2,
        image_channels=1,
        latent_dim=4
    )

    # Run
    infer()

    # Check output
    assert any(tmp_path.glob("*.nc"))
