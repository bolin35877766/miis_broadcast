"""
延迟监控工具 - 用于计算LiveCC和TTS的延迟
"""

import time
import threading
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TextSegment:
    """文本段落信息"""
    text: str
    start_time: float  # 视频时间戳
    stop_time: float   # 视频时间戳
    char_count: int    # 字符数
    generation_start_time: float  # 生成开始的系统时间
    generation_end_time: float    # 生成结束的系统时间
    tts_enqueue_time: float       # 送入TTS队列的时间
    tts_first_audio_time: float   # 首个音频块到达的时间
    
    @property
    def generation_latency(self) -> float:
        """计算生成延迟(秒)"""
        return self.generation_end_time - self.generation_start_time
    
    @property
    def tts_latency(self) -> float:
        """计算TTS延迟(秒)"""
        if self.tts_first_audio_time <= 0:
            return 0
        return self.tts_first_audio_time - self.tts_enqueue_time
    
    @property
    def total_latency(self) -> float:
        """计算总延迟(秒)"""
        if self.tts_first_audio_time <= 0:
            return self.generation_latency
        return self.tts_first_audio_time - self.generation_start_time
    
    @property
    def chars_per_second(self) -> float:
        """计算生成速度(字符/秒)"""
        if self.generation_latency <= 0:
            return 0
        return self.char_count / self.generation_latency


class LatencyMonitor:
    """延迟监控器"""
    
    def __init__(self):
        self._segments: List[TextSegment] = []
        self._lock = threading.Lock()
        self._current_segment: Optional[TextSegment] = None
        
    def start_generation(self, start_time: float, stop_time: float) -> None:
        """开始生成文本"""
        with self._lock:
            self._current_segment = TextSegment(
                text="",
                start_time=start_time,
                stop_time=stop_time,
                char_count=0,
                generation_start_time=time.time(),
                generation_end_time=0,
                tts_enqueue_time=0,
                tts_first_audio_time=0
            )
    
    def end_generation(self, text: str) -> None:
        """结束生成文本"""
        now = time.time()
        with self._lock:
            if self._current_segment:
                self._current_segment.text = text
                self._current_segment.char_count = len(text)
                self._current_segment.generation_end_time = now
                self._segments.append(self._current_segment)
                self._current_segment = None
    
    def record_tts_enqueue(self, text: str, timestamp: float) -> None:
        """记录TTS入队时间"""
        now = time.time()
        with self._lock:
            # 找到最近的匹配文本
            for segment in reversed(self._segments):
                if segment.text == text and segment.tts_enqueue_time == 0:
                    segment.tts_enqueue_time = now
                    break
    
    def record_tts_first_audio(self, timestamp: float) -> None:
        """记录TTS首个音频块时间"""
        now = time.time()
        with self._lock:
            # 找到最近的未记录TTS音频时间的段落
            for segment in reversed(self._segments):
                if segment.tts_enqueue_time > 0 and segment.tts_first_audio_time == 0:
                    segment.tts_first_audio_time = now
                    break
    
    def get_statistics(self) -> Dict[str, float]:
        """获取统计数据"""
        with self._lock:
            if not self._segments:
                return {
                    "avg_generation_latency": 0,
                    "avg_tts_latency": 0,
                    "avg_total_latency": 0,
                    "avg_chars_per_second": 0,
                    "segment_count": 0
                }
            
            gen_latencies = [s.generation_latency for s in self._segments]
            tts_latencies = [s.tts_latency for s in self._segments if s.tts_latency > 0]
            total_latencies = [s.total_latency for s in self._segments]
            chars_per_second = [s.chars_per_second for s in self._segments if s.chars_per_second > 0]
            
            return {
                "avg_generation_latency": statistics.mean(gen_latencies) if gen_latencies else 0,
                "avg_tts_latency": statistics.mean(tts_latencies) if tts_latencies else 0,
                "avg_total_latency": statistics.mean(total_latencies) if total_latencies else 0,
                "avg_chars_per_second": statistics.mean(chars_per_second) if chars_per_second else 0,
                "segment_count": len(self._segments)
            }
    
    def print_statistics(self) -> None:
        """打印统计数据"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 50)
        print("LiveCC 与 TTS 延迟统计")
        print("=" * 50)
        print(f"处理段落数: {stats['segment_count']}")
        print(f"平均文本生成延迟: {stats['avg_generation_latency']:.3f} 秒")
        print(f"平均生成速度: {stats['avg_chars_per_second']:.2f} 字/秒")
        print(f"平均TTS延迟: {stats['avg_tts_latency']:.3f} 秒")
        print(f"平均总延迟(生成+TTS): {stats['avg_total_latency']:.3f} 秒")
        print("=" * 50 + "\n")
    
    def clear(self) -> None:
        """清空统计数据"""
        with self._lock:
            self._segments.clear()
            self._current_segment = None


# 全局延迟监控器实例
latency_monitor = LatencyMonitor()

def get_latency_monitor() -> LatencyMonitor:
    """获取全局延迟监控器实例"""
    return latency_monitor