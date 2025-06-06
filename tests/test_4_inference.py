import os
import numpy as np
import pandas as pd
import torch

from itwinai.plugins.xtclim.src.trainer import TorchTrainer

def generate_mock_data(path: str, seasons, modes=["train", "test"], n_memb=1, num_samples=20):
    os.makedirs(path, exist_ok=True)
    for season in seasons:
        for mode in modes:
            # Données aléatoires (2, 32, 32) -> aplaties
            data = np.random.rand(num_samples, 2 * 32 * 32).astype(np.float32)
            np.save(f"{path}/preprocessed_1d_{mode}_{season}_data_{n_memb}memb.npy", data)

            # Dates fictives
            dates = pd.date_range("2000-01-01", periods=num_samples, freq="D")
            pd.DataFrame({"date": dates.strftime("%Y-%m-%d")}).to_csv(
                f"{path}/dates_{mode}_{season}_data_{n_memb}memb.csv", index=False
            )


def test_trainer():
    input_path = "mock_inputs"
    output_path = "mock_outputs"
    seasons = ["winter", "spring"]

    # 1. Génère des données factices
    generate_mock_data(input_path, seasons)

    # ✅ Crée le dossier de sortie s'il n'existe pas
    os.makedirs(output_path, exist_ok=True)

    # 2. Lance le trainer
    trainer = TorchTrainer(
        input_path=input_path,
        output_path=output_path,
        seasons=seasons,
        epochs=5,
        lr=1e-3,
        batch_size=4,
        n_memb=1,
        beta=0.1,
        n_avg=2,
        stop_delta=1e-2,
        patience=3,
        kernel_size=4,
        init_channels=8,
        image_channels=2,
        latent_dim=16
    )

    trainer.execute()


from itwinai.plugins.xtclim.src.trainer import TorchInference
from itwinai.plugins.xtclim.src.model import ConvVAE

def test_inference():
    input_path = "mock_inputs"
    output_path = "mock_outputs"
    seasons = ["winter", "spring"]
    scenarios = ["ssp245"]
    n_memb = 1
    latent_dim = 16
    init_channels = 8
    kernel_size = 4
    image_channels = 2

    # 1. Génère des données factices pour les projections
    generate_mock_data(input_path, seasons, modes=["proj"], n_memb=n_memb)

    # 2. Crée un modèle simulé et sauvegarde un état entraîné factice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for season in seasons:
        dummy_model = ConvVAE(
            kernel_size=kernel_size,
            init_channels=init_channels,
            image_channels=image_channels,
            latent_dim=latent_dim
        ).to(device)
        torch.save(dummy_model.state_dict(), f"{output_path}/cvae_model_{season}_1d_{n_memb}memb.pth")

    # 3. Exécute l'inférence
    inference = TorchInference(
        input_path=input_path,
        output_path=output_path,
        scenarios=scenarios,
        seasons=seasons,
        on_train_test=False,  # ou True pour tester aussi sur train/test
        n_memb=n_memb,
        kernel_size=kernel_size,
        init_channels=init_channels,
        image_channels=image_channels,
        latent_dim=latent_dim
    )
    inference.execute()

    # 4. Vérifie qu’un fichier de sortie a été créé
    for season in seasons:
        for scenario in scenarios:
            expected_path = f"{output_path}/proj{scenario}_loss_indiv_{season}_1d_{n_memb}memb.csv"
            assert os.path.exists(expected_path), f"{expected_path} not found!"

    print("✅ Inference test passed.")
