import numpy as np
import pandas as pd
from pathlib import Path

from itwinai.plugins.xtclim.preprocessing.preprocess_2d_seasons import SplitPreprocessedData


def test_split_preprocessed_data(tmp_path):
    # === Créer le dossier d'entrée
    input_dir = tmp_path
    input_path = str(input_dir)
    n_memb = 1
    scenarios = ["126"]

    # === Exécuter l'étape
    step = SplitPreprocessedData(input_path=input_path, scenarios=scenarios, n_memb=n_memb)
    step.execute()

    # === Vérifier que les fichiers ont été créés
    for dataset_type in ["train", "test", "proj126"]:
        for season in ["winter", "spring", "summer", "autumn"]:
            npy_file = input_dir / f"preprocessed_1d_{dataset_type}_data_{n_memb}memb.npy"
            csv_file = input_dir / f"dates_{dataset_type}_{season}_data_{n_memb}memb.csv"
            assert npy_file.exists(), f"Manquant : {npy_file}"
            assert csv_file.exists(), f"Manquant : {csv_file}"
