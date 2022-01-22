import os
import pathlib

ROOT_PATH = pathlib.Path(__file__).resolve().parents[1].absolute()

DATASET_PATH = os.path.join(ROOT_PATH, 'dataset')  # dataset.csv files path
if not os.path.exists(DATASET_PATH):
    raise RuntimeError('No dataset folder found, please create it')

RESULTS_PATH = os.path.join(ROOT_PATH, 'trained_models')  # Checkpoints and graphs
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
