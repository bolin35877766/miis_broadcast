import os
import datetime
import logging
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

class VRAMMonitor:
    def __init__(self, log_dir="logs"):
        """
        初始化 VRAM 監控器。
        
        Args:
            log_dir (str): log 資料夾路徑，預設存在專案根目錄的 logs 下
        """
        # 以目前工作目錄為基準往上找專案根目錄，預設將 logs 建在 miis_broadcast 的外層或內層
        self.log_dir = Path(os.getcwd()) / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "vram_tracker.log"
        
        self.logger = logging.getLogger("VRAMMonitor")
        self.logger.setLevel(logging.INFO)
        
        # 避免重複綁定 handler
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file, encoding='utf-8')
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
            
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(ch)

    def get_allocated_gb(self, device="cuda:0"):
        if torch is None or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(device) / (1024**3)

    def log_diff(self, module_name: str, before_gb: float, after_gb: float):
        """記錄模組單獨佔用的 VRAM 資源"""
        diff = after_gb - before_gb
        self.logger.info(f"[{module_name}] 初始化佔用淨 VRAM: {diff:.2f} GB (載入前 {before_gb:.2f}GB -> 載入後 {after_gb:.2f}GB)")

    def measure_and_evaluate(self, module_name: str, device: str = "cuda:0"):
        """測量目前的 VRAM，並記錄至 log"""
        if torch is None or not torch.cuda.is_available():
            self.logger.warning("CUDA 不可用，無法測量 VRAM。")
            return 0.0

        try:
            allocated = self.get_allocated_gb(device)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            
            msg = f"[{module_name}] 運行中佔用 VRAM: {allocated:.2f} GB | 保留(Reserved): {reserved:.2f} GB"
            self.logger.info(msg)
            
            return allocated
            
        except Exception as e:
            self.logger.error(f"測量 {module_name} VRAM 時發生錯誤: {e}")
            return 0.0

# 提供一個全域單例，方便各個檔案引入並共用
vram_monitor = VRAMMonitor(log_dir="miis_broadcast/logs")
