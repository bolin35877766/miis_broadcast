import logging
import os
import sys

# decord (used by livecc_utils) bundles its own libxcb which conflicts with
# Qt/PySide6's system libxcb, causing SIGSEGV in X11 operations at runtime.
# Re-exec with LD_PRELOAD to force the system libxcb before decord loads it.
_SYSTEM_LIBXCB = "/usr/lib/x86_64-linux-gnu/libxcb.so.1"
if os.path.exists(_SYSTEM_LIBXCB) and _SYSTEM_LIBXCB not in os.environ.get("LD_PRELOAD", ""):
    env = os.environ.copy()
    preload = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = (_SYSTEM_LIBXCB + ":" + preload).strip(":")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)

from dotenv import find_dotenv, load_dotenv
from PySide6 import QtWidgets

from .core.utils.config import parse_configs

CONFIG_PATH = "./configs/app.yml"
MODEL_CONFIG_PATH = "./configs/models.yml"


def main():
    logging.basicConfig(filename="logs/app_error.log", level=logging.INFO)
    # Load .env from project root (OPENAI_API_KEY, GEMINI_API_KEY, etc.)
    load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

    # QApplication must be created BEFORE importing gui/torch to avoid segfault
    # (CUDA libraries conflict with Qt display init if loaded first)
    app = QtWidgets.QApplication([])

    # Pre-init Gemini clients here — before torch/CUDA loads — so httpx's
    # SSL context and thread pool are established in a clean single-threaded
    # environment. Doing this after model load causes heap corruption because
    # CUDA runtime threads and httpx threads race on the system allocator.
    try:
        from .core.models.gemini_broadcaster import _get_client as _gcb
        _gcb()
    except Exception:
        pass  # Missing API key or network error — workers will retry

    from .gui import MainWindow
    from .core.models.openai_tts import enable_dry_run

    # 設 TTS_DRY_RUN=1 啟用 log 模擬模式（不播音，驗證 interrupt 機制）
    if os.environ.get("TTS_DRY_RUN", "0") == "1":
        enable_dry_run(True)

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
