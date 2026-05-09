import logging
import sys

from dotenv import find_dotenv, load_dotenv
from PySide6 import QtWidgets

from .gui import MainWindow
from .core.utils.config import parse_configs

CONFIG_PATH = "./configs/app.yml"
MODEL_CONFIG_PATH = "./configs/models.yml"


def main():
    logging.basicConfig(filename="logs/app_error.log", level=logging.INFO)
    # Load .env from project root (LIVEAVATAR_API_KEY, OPENAI_API_KEY, etc.)
    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))
    app = QtWidgets.QApplication([])

    configs = parse_configs(CONFIG_PATH)
    model_configs = parse_configs(MODEL_CONFIG_PATH)
    classifier_configs = model_configs["classifiers"]

    configs["gui_window"]["classifier_list"] = classifier_configs
    classifier_name = configs["model"]["classifier_name"]
    if classifier_name in classifier_configs.keys():
        configs["model"]["classifier"] = classifier_configs[classifier_name]
    else:
        error_msg = (
            f"Classifier name defined in {CONFIG_PATH} does not exist in {MODEL_CONFIG_PATH}."
        )
        logging.error(error_msg)
        QtWidgets.QMessageBox.information(
            None, "Error", error_msg, QtWidgets.QMessageBox.StandardButton.Ok
        )
        sys.exit()

    configs["bytetrack"] = model_configs.get("bytetrack", {})

    mw = MainWindow(configs=configs)
    mw.show()

    sys.exit(app.exec_())
