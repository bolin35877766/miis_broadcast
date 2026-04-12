import logging
import sys
from PySide6 import QtWidgets

from .gui import MainWindow
from .core.utils.config import parse_configs

CONFIG_PATH = './configs/app.yml'
MODEL_CONFIG_PATH = './configs/models.yml'

def main():
    logging.basicConfig(filename='logs/app_error.log', level=logging.INFO)
    app = QtWidgets.QApplication([])

    # Config parsing
    configs = parse_configs(CONFIG_PATH)
    model_configs = parse_configs(MODEL_CONFIG_PATH)
    classifier_configs = model_configs['classifiers']

    # Merge configs
    configs['gui_window']['classifier_list'] = classifier_configs
    classifier_name = configs['model']['classifier_name']
    if classifier_name in classifier_configs.keys():
        configs['model']['classifier'] = classifier_configs[classifier_name]
    else:
        error_msg = f"Classifier name defined in {CONFIG_PATH} does not exist in {MODEL_CONFIG_PATH}."
        logging.error(error_msg)
        QtWidgets.QMessageBox.information(None, 'Error', error_msg, QtWidgets.QMessageBox.StandardButton.Ok)
        sys.exit()

    # Pass bytetrack section to MainWindow
    configs['bytetrack'] = model_configs.get('bytetrack', {})

    mw = MainWindow(configs=configs)
    mw.show()

    sys.exit(app.exec_())
