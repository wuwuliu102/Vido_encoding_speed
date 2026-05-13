# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
dcvc_test_split.py

基于你附件 test.txt（moban.py）脚本的工程化改造版本：把"端到端(encode+decode)"拆成
可独立执行的两个阶段：
  1) encode: 读源视频 → DCVC-RT compress → 写 .bin
  2) decode: 读 .bin   → DCVC-RT decompress → (可选)算 PSNR/SSIM/写重建帧

关键目标：
- 编码与解码可以单独运行（例如只做编码性能、或只做解码性能/只解码不同机器上的同一 bitstream）
- 保持原工程 JSON 输出结构（generate_log_json），方便延续你 Word 表格的统计口径
- decode-only 模式下也能独立计算 bits（通过解析 .bin 的字节消耗），不依赖 encode log
- 完善时间统计：同时记录GPU核心时间和端到端时间


注意：
- 如你要把它合并回原 test_video.py，只需要把"新增参数 + 拆分函数"合并即可

"""

import argparse
import io
import json
import os
import sys
import time
import threading
import struct
import traceback
from dataclasses import dataclass
from collections import deque
from queue import Queue
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    import cv2
except Exception:
    cv2 = None


# ============================================================
#  工程内部导入
# ============================================================
from src.layers.cuda_inference import replicate_pad, round_and_to_int8, CUSTOMIZED_CUDA_INFERENCE
from src.models.video_model import DMC
from src.models.image_model import DMCI
from src.utils.common import (
    str2bool,
    create_folder,
    generate_log_json,
    get_state_dict,
    dump_json,
    set_torch_env,
)
from src.utils.stream_helper import (
    SPSHelper,
    NalType,
    write_sps,
    read_header,
    read_sps_remaining,
    read_ip_remaining,
    write_ip,
)
from src.utils.video_reader import PNGReader, YUV420Reader
from src.utils.video_writer import PNGWriter, YUV420Writer
from src.utils.metrics import calc_psnr, calc_msssim, calc_msssim_rgb
from src.utils.transforms import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_444_to_420,
    ycbcr420_to_444_np,
)


# ============================================================
# 参数解析
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DCVC-RT split-stage test script")
    parser.add_argument("--force_zero_thres", type=float, default=0.22, required=False)
    parser.add_argument("--model_path_i", type=str, default="checkpoints/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", type=str, default="checkpoints/cvpr2025_video.pth.tar")
    parser.add_argument("--rate_num", type=int, default=2)
    parser.add_argument("--qp_i", type=int, nargs="+")
    parser.add_argument("--qp_p", type=int, nargs="+")
    parser.add_argument("--force_intra", type=str2bool, default=False)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--force_intra_period", type=int, default=32)
    parser.add_argument("--reset_interval", type=int, default=128, required=False)
    parser.add_argument("--test_config", type=str, default="mytest_720.json")
    parser.add_argument("--force_root_path", type=str, default=None, required=False)

    parser.add_argument("--worker", "-w", type=int, default=1, help="worker number ")
    parser.add_argument("--cuda", type=str2bool, default=True)
    parser.add_argument("--cuda_idx", type=int, nargs="+", default=[0], help="GPU indexes to use")

    parser.add_argument("--calc_ssim", type=str2bool, default=False, required=False)
    parser.add_argument("--write_stream", type=str2bool, default=True)
    parser.add_argument("--check_existing", type=str2bool, default=False)
    parser.add_argument("--stream_path", type=str, default="out_bin")
    parser.add_argument("--save_decoded_frame", type=str2bool, default=False)
    parser.add_argument("--output_path", type=str, default="outde5.json")
    parser.add_argument("--verbose_json", type=str2bool, default=False)
    parser.add_argument("--verbose", type=int, default=1)

    # rt_mode: 0=research mode (with metrics/output), 1=real-time mode (encode/decode only)
    parser.add_argument("--rt_mode", type=int, default=0,
                        help="0=research mode (with metrics), 1=real-time mode (no metrics)")

    # -------- Precision / Warmup --------
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="inference precision: fp16 or fp32")
    parser.add_argument("--min_warmup", type=int, default=5,
                        help="warmup frames not counted in average timing")

    # -------- Stage Control --------
    parser.add_argument("--stage", type=str, default="encode",
                        choices=["encode", "decode", "both"],
                        help="encode=encode only, decode=decode only, both=end-to-end")
    parser.add_argument("--decode_bin_override", type=str, default=None,
                        help="decode-only 指定输入bin文件 路径可选")
    parser.add_argument("--enc_log_suffix", type=str, default="_enc.json")
    parser.add_argument("--dec_log_suffix", type=str, default="_dec.json")

    # -------- Time Metrics --------
    parser.add_argument("--profile_i_path", type=str2bool, default=False,
                        help="enable detailed profiling for I-frame path")
    parser.add_argument("--profile_p_path", type=str2bool, default=False,
                        help="enable detailed profiling for P-frame path")
    parser.add_argument("--optimize_p_prior2x_workspace", type=str2bool, default=False,
                        help="Enable workspace/cat reuse in P prior2x path")
    parser.add_argument("--optimize_p_resprior_workspace", type=str2bool, default=False,
                        help="Enable workspace/cat reuse in P res-prior path")
    parser.add_argument("--optimize_p_encoder_cat_workspace", type=str2bool, default=False,
                        help="Enable workspace/cat reuse for Encoder feature+ctx concat")
    parser.add_argument("--strict_frame_sync", type=str2bool, default=True,
                        help="strictly align decoded frames and reference frames in decode/metrics mode")
    parser.add_argument("--max_nonframe_nals", type=int, default=1000,
                        help="max continuous non-frame NAL count to avoid abnormal bitstream dead-loop")

    # -------- 输入模式--------
    parser.add_argument("--input_mode", type=str, default="file", choices=["file", "camera"],
                        help="file=绂荤嚎鏂囦欢锛沜amera=Jetson 澶栭儴宸ヤ笟鐩告満")

    # -------- Jetson camera 参数 --------
    parser.add_argument("--camera_backend", type=str, default="jetson_gst", choices=["jetson_gst"])
    parser.add_argument("--camera_device", type=str, default="/dev/video0")
    parser.add_argument("--camera_format", type=str, default="YUY2", choices=["YUY2", "MJPG", "MJPEG"])
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--camera_fps", type=int, default=60, help="")
    parser.add_argument("--camera_frames", type=int, default=120, help="number of frames in fixed_frames mode")
    parser.add_argument("--camera_control_mode", type=str, default="fixed_frames",
                        choices=["fixed_frames", "manual_session"],
                        help="fixed_frames=按编码帧自动结束 manual_session=预览+按键")
    parser.add_argument("--camera_preview", type=str2bool, default=True,
                        help="enable preview window in manual_session mode (requires X11 + OpenCV)")
    parser.add_argument("--camera_timeout_sec", type=float, default=2.0)
    parser.add_argument("--camera_v4l2_io_mode", type=int, default=2,
                        help="2=mmap驱动支持时可以测试")
    parser.add_argument("--camera_session_prefix", type=str, default="camera720p60")
    parser.add_argument("--camera_ds_name", type=str, default="camera_rt")
    parser.add_argument("--camera_dump_reference", type=str2bool, default=True,
                        help="实际送入编码器的 I420 保存为*_src.yuv供 decode-only 严格对齐PSNR")
    parser.add_argument("--camera_reference_yuv", type=str, default=None,
                        help="decode-only 不填则为  {bin}_src.yuv")
    parser.add_argument("--camera_start_key", type=str, default="s")
    parser.add_argument("--camera_stop_key", type=str, default="e")
    parser.add_argument("--camera_quit_key", type=str, default="q")
    parser.add_argument("--camera_buffer_mode", type=str, default="fifo", choices=["fifo", "latest"])
    parser.add_argument("--camera_buffer_max_frames", type=int, default=0)
    parser.add_argument("--camera_drain_print_interval", type=float, default=1.0)
    parser.add_argument("--save_decoded_yuv", type=str2bool, default=False)
    parser.add_argument("--save_decoded_video", type=str2bool, default=False)
    parser.add_argument("--decoded_video_ext", type=str, default="avi", choices=["avi", "mp4"])
    parser.add_argument("--decoded_video_codec", type=str, default="MJPG")
    parser.add_argument("--decoded_video_fps", type=float, default=0.0)

    parser.add_argument("--optimize_prior4x_workspace", type=str2bool, default=True,
                        help="是否启用 prior_4x workspace/cache/in-place 优化")
    parser.add_argument("--optimize_yuv420_tensorize", type=str2bool, default=False,
                        help="是否启用 YUV420->Tensor workspace 优化（默认关闭）")
    parser.add_argument("--yuv420_use_pinned_memory", type=str2bool, default=True,
                        help="YUV420 tensorize 优化是否启用pinned memory")
    parser.add_argument("--yuv420_async_h2d", type=str2bool, default=True,
                        help="YUV420 tensorize 优化是否启用async H2D")
    return parser.parse_args()


# ============================================================
# 环境设置
# ============================================================

def setup_environment() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ CUDA environment optimization enabled")


# ============================================================
# 全局模型缓存（用于多任务优化）
# ============================================================

i_frame_net_global: DMCI = None
p_frame_net_global: DMC = None


def init_models(args: argparse.Namespace):
    global i_frame_net_global, p_frame_net_global
    clear_prior4x_caches()

    # Torch & CUDA 环境
    set_torch_env()
    setup_environment()

    # 设备设置
    device = "cuda:0" if args.cuda else "cpu"
    if args.cuda and args.cuda_idx is not None and len(args.cuda_idx) > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_idx[0])

    print(f"设备: {device}, 精度: {args.precision}")

    # ===== 初始化 I-frame 模型 =====
    i_frame_net_global = DMCI()
    i_state_dict = get_state_dict(args.model_path_i)
    i_frame_net_global.load_state_dict(i_state_dict)
    i_frame_net_global = i_frame_net_global.to(device)
    i_frame_net_global.eval()
    if args.force_zero_thres is not None:
        i_frame_net_global.update(args.force_zero_thres)

    # ===== 初始化 P-frame 模型 =====
    p_frame_net_global = DMC()
    if not args.force_intra:
        p_state_dict = get_state_dict(args.model_path_p)
        p_frame_net_global.load_state_dict(p_state_dict)
        p_frame_net_global = p_frame_net_global.to(device)
        p_frame_net_global.eval()
        if args.force_zero_thres is not None:
            p_frame_net_global.update(args.force_zero_thres)

    p_frame_net_global.fast_decode = True

    # ===== 应用精度优化 =====
    if args.precision == "fp16":
        i_frame_net_global = i_frame_net_global.half()
        if not args.force_intra:
            p_frame_net_global = p_frame_net_global.half()
        print("模型使用 FP16 推理")
    else:
        i_frame_net_global = i_frame_net_global.float()
        if not args.force_intra:
            p_frame_net_global = p_frame_net_global.float()
        print("模型使用 FP32 推理")

    if not args.force_intra:
        p_frame_net_global._enable_prior2x_workspace = bool(getattr(args, "optimize_p_prior2x_workspace", False)) and bool(args.cuda)
        p_frame_net_global._prior2x_cat_workspace_cache = {}
        p_frame_net_global._enable_resprior_workspace = bool(getattr(args, "optimize_p_resprior_workspace", False)) and bool(args.cuda)
        p_frame_net_global._resprior_cat_workspace_cache = {}
        p_frame_net_global.encoder._enable_cat_workspace = bool(getattr(args, "optimize_p_encoder_cat_workspace", False)) and bool(args.cuda)
        p_frame_net_global.encoder._cat_workspace_cache = {}
        print(f"✓ P prior2x workspace/cat reuse: {'ON' if p_frame_net_global._enable_prior2x_workspace else 'OFF'}")
        print(f"✓ P res-prior workspace/cat reuse: {'ON' if p_frame_net_global._enable_resprior_workspace else 'OFF'}")
        print(f"✓ P encoder-cat workspace/cat reuse: {'ON' if p_frame_net_global.encoder._enable_cat_workspace else 'OFF'}")

    print("Model initialization complete")
    return i_frame_net_global, p_frame_net_global


# ============================================================
# 数据源 Reader 构造（PNG / YUV420 / Jetson Camera）
# ============================================================


def normalize_camera_format(fmt: str) -> str:
    fmt = str(fmt).upper()
    if fmt == 'MJPEG':
        fmt = 'MJPG'
    return fmt


class JetsonCameraReader:
    """
    Jetson 摄像头读取器：
    - 采集链路固定输出 I420
    - 支持同步 read_one_frame()
    - 支持后台线程持续抓帧 + latest-frame 消费（编码跟不上时自动丢旧帧）
    """

    def __init__(self, args_or_dict: Dict):
        self.args = dict(args_or_dict)
        self.width = int(args_or_dict['src_width'])
        self.height = int(args_or_dict['src_height'])
        self.fps = int(args_or_dict.get('camera_fps', 60))
        self.device = args_or_dict.get('camera_device', '/dev/video0')
        self.camera_format = normalize_camera_format(args_or_dict.get('camera_format', 'YUY2'))
        self.timeout_sec = float(args_or_dict.get('camera_timeout_sec', 2.0))
        self.io_mode = int(args_or_dict.get('camera_v4l2_io_mode', 2))

        self.Gst = None
        self.pipeline = None
        self.appsink = None
        self.bus = None

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._capture_thread = None
        self._latest_seq = 0
        self._latest_ts = 0.0
        self._latest_y = None
        self._latest_uv = None
        self._captured_total = 0
        self._buffer_mode = str(args_or_dict.get('camera_buffer_mode', 'fifo')).lower()
        self._buffer_max_frames = int(args_or_dict.get('camera_buffer_max_frames', 0))
        self._fifo_queue = deque()
        self._fifo_accepting = self._buffer_mode == 'fifo'
        self._fifo_enqueued_total = 0
        self._fifo_overflow_drop_total = 0

        self._open()

    def _lazy_import_gst(self):
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(
                "camera mode requires python3-gi and GStreamer runtime on Jetson"
            ) from exc
        if not Gst.is_initialized():
            Gst.init(None)
        self.Gst = Gst

    def _build_pipeline(self) -> str:
        if self.camera_format == 'YUY2':
            return (
                f"v4l2src device={self.device} io-mode={self.io_mode} do-timestamp=true ! "
                f"video/x-raw,width=(int){self.width},height=(int){self.height},framerate=(fraction){self.fps}/1,format=(string)YUY2 ! "
                f"nvvidconv ! video/x-raw,format=(string)I420 ! "
                f"appsink name=dcvcsink sync=false max-buffers=1 drop=true wait-on-eos=false"
            )
        elif self.camera_format == 'MJPG':
            return (
                f"v4l2src device={self.device} io-mode={self.io_mode} do-timestamp=true ! "
                f"image/jpeg,width=(int){self.width},height=(int){self.height},framerate=(fraction){self.fps}/1 ! "
                f"jpegparse ! nvv4l2decoder mjpeg=1 ! "
                f"nvvidconv ! video/x-raw,format=(string)I420 ! "
                f"appsink name=dcvcsink sync=false max-buffers=1 drop=true wait-on-eos=false"
            )
        raise ValueError(f'Unsupported camera_format: {self.camera_format}')

    def _open(self):
        self._lazy_import_gst()
        Gst = self.Gst
        pipe = self._build_pipeline()
        self.pipeline = Gst.parse_launch(pipe)
        self.appsink = self.pipeline.get_by_name('dcvcsink')
        if self.appsink is None:
            raise RuntimeError(f'appsink not found in pipeline: {pipe}')
        self.appsink.set_property('emit-signals', False)
        self.appsink.set_property('sync', False)
        self.appsink.set_property('max-buffers', 1)
        self.appsink.set_property('drop', True)
        self.bus = self.pipeline.get_bus()
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f'Failed to start camera pipeline: {pipe}')
        state_ret, _, _ = self.pipeline.get_state(int(self.timeout_sec * Gst.SECOND))
        if state_ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f'Camera pipeline state transition failed: {pipe}')

    def _poll_bus(self):
        if self.bus is None:
            return
        Gst = self.Gst
        while True:
            msg = self.bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.WARNING)
            if msg is None:
                break
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                raise RuntimeError(f'GStreamer camera error: {err}; debug={dbg}')
            if msg.type == Gst.MessageType.EOS:
                raise EOFError('Camera pipeline EOS')
            if msg.type == Gst.MessageType.WARNING:
                warn, dbg = msg.parse_warning()
                print(f'[WARN][GST] {warn}; debug={dbg}')

    def read_one_frame(self):
        Gst = self.Gst
        self._poll_bus()
        sample = self.appsink.emit('try-pull-sample', int(self.timeout_sec * Gst.SECOND))
        if sample is None:
            self._poll_bus()
            return None, None
        buffer = sample.get_buffer()
        if buffer is None:
            return None, None
        ok, map_info = buffer.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError('Failed to map camera appsink buffer')
        try:
            expected = self.width * self.height * 3 // 2
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            if frame.size < expected:
                raise RuntimeError(f'Camera I420 frame too small: got={frame.size}, expected>={expected}')
            frame = frame[:expected]
            y_size = self.width * self.height
            uv_size = y_size // 4
            y = frame[:y_size].reshape(1, self.height, self.width).copy()
            u = frame[y_size:y_size + uv_size].reshape(self.height // 2, self.width // 2)
            v = frame[y_size + uv_size:y_size + 2 * uv_size].reshape(self.height // 2, self.width // 2)
            uv = np.stack([u, v], axis=0).copy()
            return y, uv
        finally:
            buffer.unmap(map_info)

    def start_background_capture(self):
        if self._capture_thread is not None:
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def _capture_loop(self):
        while not self._stop_event.is_set():
            y, uv = self.read_one_frame()
            if y is None:
                continue
            ts = time.time()
            with self._cond:
                self._latest_seq += 1
                self._latest_ts = ts
                self._latest_y = y
                self._latest_uv = uv
                self._captured_total += 1
                if self._buffer_mode == 'fifo' and self._fifo_accepting:
                    self._fifo_enqueued_total += 1
                    item = {
                        'seq': self._latest_seq,
                        'ts': ts,
                        'y': y,
                        'uv': uv,
                        'skipped': 0,
                        'captured_total': self._captured_total,
                        'fifo_enqueued_total': self._fifo_enqueued_total,
                    }
                    if self._buffer_max_frames > 0 and len(self._fifo_queue) >= self._buffer_max_frames:
                        self._fifo_queue.popleft()
                        self._fifo_overflow_drop_total += 1
                    self._fifo_queue.append(item)
                self._cond.notify_all()

    def wait_latest_after(self, last_seq: int, timeout_sec: Optional[float] = None):
        timeout_sec = self.timeout_sec if timeout_sec is None else timeout_sec
        deadline = time.time() + timeout_sec
        with self._cond:
            while self._latest_seq <= last_seq and not self._stop_event.is_set():
                remain = deadline - time.time()
                if remain <= 0:
                    return None
                self._cond.wait(timeout=remain)
            if self._latest_seq <= last_seq:
                return None
            skipped = max(0, self._latest_seq - last_seq - 1)
            return {
                'seq': self._latest_seq,
                'ts': self._latest_ts,
                'y': self._latest_y,
                'uv': self._latest_uv,
                'skipped': skipped,
                'captured_total': self._captured_total,
            }

    def wait_next_frame(self, timeout_sec: Optional[float] = None):
        timeout_sec = self.timeout_sec if timeout_sec is None else timeout_sec
        deadline = time.time() + timeout_sec
        with self._cond:
            while len(self._fifo_queue) == 0 and not self._stop_event.is_set():
                remain = deadline - time.time()
                if remain <= 0:
                    return None
                self._cond.wait(timeout=remain)
            if len(self._fifo_queue) == 0:
                return None
            return self._fifo_queue.popleft()

    def get_queue_size(self) -> int:
        with self._lock:
            return len(self._fifo_queue)

    def clear_fifo_queue(self) -> int:
        with self._lock:
            n = len(self._fifo_queue)
            self._fifo_queue.clear()
            return n

    def set_fifo_accepting(self, accept: bool):
        with self._lock:
            self._fifo_accepting = bool(accept)

    def is_fifo_accepting(self) -> bool:
        with self._lock:
            return bool(self._fifo_accepting)

    def get_fifo_enqueued_total(self) -> int:
        with self._lock:
            return int(self._fifo_enqueued_total)

    def get_fifo_overflow_drop_total(self) -> int:
        with self._lock:
            return int(self._fifo_overflow_drop_total)

    def peek_latest(self):
        with self._lock:
            if self._latest_seq <= 0:
                return None
            return {
                'seq': self._latest_seq,
                'ts': self._latest_ts,
                'y': self._latest_y,
                'uv': self._latest_uv,
                'captured_total': self._captured_total,
            }

    def get_captured_total(self) -> int:
        with self._lock:
            return self._captured_total

    def close(self):
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self.pipeline is not None:
            try:
                self.pipeline.set_state(self.Gst.State.NULL)
            except Exception:
                pass
            self.pipeline = None
            self.appsink = None
            self.bus = None


def get_src_reader(args_or_dict: Dict):
    if args_or_dict['src_type'] == 'png':
        return PNGReader(args_or_dict['src_path'], args_or_dict['src_width'], args_or_dict['src_height'])
    elif args_or_dict['src_type'] == 'yuv420':
        return YUV420Reader(args_or_dict['src_path'], args_or_dict['src_width'], args_or_dict['src_height'])
    elif args_or_dict['src_type'] == 'camera':
        return JetsonCameraReader(args_or_dict)
    else:
        raise ValueError(f"Unknown src_type: {args_or_dict['src_type']}")


# ============================================================

# ============================================================

class TimeMetrics:
    """Fixed standard timing: per-frame synchronized GPU+CPU timing."""

    def __init__(self, args: Dict):
        self.min_warmup = int(args.get('min_warmup', 5))
        self.device = None
        self.gpu_timing = False

    def set_device(self, device):
        self.device = device
        self.gpu_timing = device.type.startswith('cuda')

    def create_events(self):
        if self.gpu_timing:
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        return None, None

    def record_start(self):
        cpu_start = time.time()
        gpu_start, gpu_end = self.create_events()
        if gpu_start is not None:
            gpu_start.record()
        return cpu_start, gpu_start, gpu_end

    def record_end(self, cpu_start, gpu_start, gpu_end):
        cpu_time = None
        gpu_time = None
        if gpu_end is not None and self.gpu_timing:
            gpu_end.record()
            torch.cuda.synchronize(self.device)
            if gpu_start is not None:
                gpu_time = gpu_start.elapsed_time(gpu_end) / 1000.0
        cpu_time = time.time() - cpu_start
        return cpu_time, gpu_time

    def finalize_gpu_times(self, gpu_times: List[float]) -> List[float]:
        return gpu_times

    def filter_warmup(self, data_list):
        if len(data_list) > self.min_warmup:
            return data_list[self.min_warmup:]
        return data_list

    def compute_averages(self, cpu_times, gpu_times):
        gpu_times = self.finalize_gpu_times(gpu_times)
        cpu_times_filtered = self.filter_warmup(cpu_times) if cpu_times else []
        gpu_times_filtered = self.filter_warmup(gpu_times) if gpu_times else []
        avg_cpu = sum(cpu_times_filtered) / len(cpu_times_filtered) if cpu_times_filtered else 0.0
        avg_gpu = sum(gpu_times_filtered) / len(gpu_times_filtered) if gpu_times_filtered else 0.0
        return avg_cpu, avg_gpu

    def print_time_summary(self, stage, frame_count, avg_cpu, avg_gpu):
        stage_name = '编码' if stage == 'ENC' else '解码'
        if self.gpu_timing:
            extra = avg_cpu - avg_gpu if avg_cpu > avg_gpu else 0.0
            print(f'[{stage}] 有效帧 {frame_count} | GPU核心: {avg_gpu * 1000:.2f} ms/帧 | '
                  f'{stage_name}墙钟: {avg_cpu * 1000:.2f} ms/帧 | 额外开销: {extra * 1000:.2f} ms/帧')
        else:
            print(f'[{stage}] 有效帧 {frame_count} | {stage_name}墙钟: {avg_cpu * 1000:.2f} ms/帧')


def _to_float(x) -> float:
    return float(x) if x is not None else 0.0


def _safe_percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def _build_scalar_summary(values: List[float]) -> Dict[str, float]:
    values = [float(v) for v in values]
    if len(values) == 0:
        return {'mean': 0.0, 'p50': 0.0, 'p95': 0.0, 'p99': 0.0, 'max': 0.0}
    return {
        'mean': float(sum(values) / len(values)),
        'p50': _safe_percentile(values, 50),
        'p95': _safe_percentile(values, 95),
        'p99': _safe_percentile(values, 99),
        'max': float(max(values)),
    }


def classify_frame_type(frame_idx: int, intra_period: int, reset_interval: int) -> Tuple[bool, str, int]:
    is_i_frame = (frame_idx == 0) or (intra_period > 0 and frame_idx % intra_period == 0)
    if is_i_frame:
        return True, 'I', 0
    if reset_interval > 0 and frame_idx % reset_interval == 1:
        return False, 'P_reset', 1
    if intra_period > 0 and frame_idx % intra_period == 1:
        return False, 'P_bootstrap', 0
    return False, 'P_steady', 0


PRIOR4X_MASK_CACHE: Dict[Tuple[int, int, int, int, str, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
PRIOR4X_WS_CACHE: Dict[Tuple[int, int, int, int, int, int, str, str], Dict[str, torch.Tensor]] = {}
ENABLE_PRIOR4X_WORKSPACE_OPT = True


def _prior4x_cache_key(B: int, C: int, H: int, W: int, dtype, device) -> Tuple[int, int, int, int, str, str]:
    return (B, C, H, W, str(dtype), str(device))


def get_cached_mask_4x(i_frame_net, B: int, C: int, H: int, W: int, dtype, device):
    key = _prior4x_cache_key(B, C, H, W, dtype, device)
    cached = PRIOR4X_MASK_CACHE.get(key)
    if cached is None:
        cached = i_frame_net.get_mask_4x(B, C, H, W, dtype, device)
        PRIOR4X_MASK_CACHE[key] = cached
    return cached


def get_prior4x_workspace(y: torch.Tensor, common_params: torch.Tensor):
    B, C, H, W = y.shape
    common_c = common_params.shape[1]
    key = (B, C, H, W, common_c, C + common_c, str(y.dtype), str(y.device))
    ws = PRIOR4X_WS_CACHE.get(key)
    if ws is None:
        ws = {
            'y_q': torch.empty_like(y),
            'y_hat_so_far': torch.empty_like(y),
            'params_cat': torch.empty((B, C + common_c, H, W), dtype=y.dtype, device=y.device),
            'y_hat_final': torch.empty_like(y),
        }
        PRIOR4X_WS_CACHE[key] = ws
    return ws


def clear_prior4x_caches():
    PRIOR4X_MASK_CACHE.clear()
    PRIOR4X_WS_CACHE.clear()


def _timed_cuda_segment(device, enabled: bool, fn):
    if not enabled:
        result = fn()
        return result, 0.0, 0.0
    torch.cuda.synchronize(device)
    cpu_t0 = time.time()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    result = fn()
    ev1.record()
    torch.cuda.synchronize(device)
    cpu_dt = time.time() - cpu_t0
    gpu_dt = ev0.elapsed_time(ev1) / 1000.0
    return result, cpu_dt, gpu_dt


def profile_intra_decoder_breakdown(decoder, y_hat, curr_q_dec, device, enable_timing: bool = True):
    def _run_torch_path(inp):
        out = inp

        def seg_up():
            return decoder.dec_1[0](out)

        out1, t1_cpu, t1_gpu = _timed_cuda_segment(device, enable_timing, seg_up)

        def seg_stack():
            tmp = out1
            if len(decoder.dec_1) > 2:
                for layer in decoder.dec_1[1:-1]:
                    tmp = layer(tmp)
            return tmp

        out2, t2_cpu, t2_gpu = _timed_cuda_segment(device, enable_timing, seg_stack)

        def seg_tail():
            tmp = decoder.dec_1[-1](out2)
            tmp = tmp * curr_q_dec
            tmp = decoder.dec_2(tmp)
            return tmp

        out3, t3_cpu, t3_gpu = _timed_cuda_segment(device, enable_timing, seg_tail)

        def seg_shuffle():
            return F.pixel_shuffle(out3, 8).clamp_(0, 1)

        x_hat, t4_cpu, t4_gpu = _timed_cuda_segment(device, enable_timing, seg_shuffle)
        return x_hat, (t1_cpu, t1_gpu, t2_cpu, t2_gpu, t3_cpu, t3_gpu, t4_cpu, t4_gpu)

    if (not y_hat.is_cuda) or (not CUSTOMIZED_CUDA_INFERENCE):
        return _run_torch_path(y_hat)

    out = y_hat

    def seg_up():
        return decoder.dec_1[0](out)

    out1, t1_cpu, t1_gpu = _timed_cuda_segment(device, enable_timing, seg_up)

    def seg_stack():
        tmp = out1
        for idx in range(1, len(decoder.dec_1) - 1):
            tmp = decoder.dec_1[idx](tmp)
        return tmp

    out2, t2_cpu, t2_gpu = _timed_cuda_segment(device, enable_timing, seg_stack)

    def seg_tail():
        tmp = decoder.dec_1[len(decoder.dec_1) - 1](out2, quant_step=curr_q_dec)
        tmp = decoder.dec_2(tmp)
        return tmp

    out3, t3_cpu, t3_gpu = _timed_cuda_segment(device, enable_timing, seg_tail)

    def seg_shuffle():
        return F.pixel_shuffle(out3, 8).clamp_(0, 1)

    x_hat, t4_cpu, t4_gpu = _timed_cuda_segment(device, enable_timing, seg_shuffle)
    return x_hat, (t1_cpu, t1_gpu, t2_cpu, t2_gpu, t3_cpu, t3_gpu, t4_cpu, t4_gpu)


def profile_prior4x_breakdown(i_frame_net, y, z_hat, device, enable_timing: bool = True):
    def seg_params_gen():
        params = i_frame_net.hyper_dec(z_hat)
        params = i_frame_net.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        return params

    params, p0_cpu, p0_gpu = _timed_cuda_segment(device, enable_timing, seg_params_gen)

    def seg_stage0_opt():
        q_enc, q_dec, scales, means = i_frame_net.separate_prior(params, False)
        common_params = i_frame_net.y_spatial_prior_reduction(params)
        dtype = y.dtype
        device0 = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = get_cached_mask_4x(i_frame_net, B, C, H, W, dtype, device0)
        ws = get_prior4x_workspace(y, common_params)
        torch.mul(y, q_enc, out=ws['y_q'])
        _, y_q_0, y_hat_0, s_hat_0 = i_frame_net.process_with_mask(ws['y_q'], scales, means, mask_0)
        ws['y_hat_so_far'].copy_(y_hat_0)
        return {
            'q_dec': q_dec,
            'common_params': common_params,
            'mask_1': mask_1,
            'mask_2': mask_2,
            'mask_3': mask_3,
            'ws': ws,
            'y_q_0': y_q_0,
            's_hat_0': s_hat_0,
            's_w_0': i_frame_net.single_part_for_writing_4x(y_q_0),
            'scale_w_0': i_frame_net.single_part_for_writing_4x(s_hat_0),
        }

    def seg_stage0_baseline():
        q_enc, q_dec, scales, means = i_frame_net.separate_prior(params, False)
        common_params = i_frame_net.y_spatial_prior_reduction(params)
        dtype = y.dtype
        device0 = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = i_frame_net.get_mask_4x(B, C, H, W, dtype, device0)
        y_q = y * q_enc
        _, y_q_0, y_hat_0, s_hat_0 = i_frame_net.process_with_mask(y_q, scales, means, mask_0)
        return {
            'q_dec': q_dec,
            'common_params': common_params,
            'mask_1': mask_1,
            'mask_2': mask_2,
            'mask_3': mask_3,
            'y_q': y_q,
            'y_hat_so_far': y_hat_0,
            's_hat_0': s_hat_0,
            'y_q_0': y_q_0,
            's_w_0': i_frame_net.single_part_for_writing_4x(y_q_0),
            'scale_w_0': i_frame_net.single_part_for_writing_4x(s_hat_0),
        }

    seg_stage0 = seg_stage0_opt if ENABLE_PRIOR4X_WORKSPACE_OPT else seg_stage0_baseline
    info, p1_cpu, p1_gpu = _timed_cuda_segment(device, enable_timing, seg_stage0)

    def seg_stage1_opt():
        ws = info['ws']
        C = ws['y_hat_so_far'].shape[1]
        ws['params_cat'][:, :C].copy_(ws['y_hat_so_far'])
        ws['params_cat'][:, C:].copy_(info['common_params'])
        scales1, means1 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_1(ws['params_cat'])).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = i_frame_net.process_with_mask(ws['y_q'], scales1, means1, info['mask_1'])
        ws['y_hat_so_far'].add_(y_hat_1)
        return {
            'y_q_1': y_q_1,
            's_hat_1': s_hat_1,
        }

    def seg_stage1_baseline():
        params1 = torch.cat((info['y_hat_so_far'], info['common_params']), dim=1)
        scales1, means1 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_1(params1)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = i_frame_net.process_with_mask(info['y_q'], scales1, means1, info['mask_1'])
        y_hat_so_far = info['y_hat_so_far'] + y_hat_1
        return {
            'y_hat_so_far': y_hat_so_far,
            'y_q_1': y_q_1,
            's_hat_1': s_hat_1,
        }

    seg_stage1 = seg_stage1_opt if ENABLE_PRIOR4X_WORKSPACE_OPT else seg_stage1_baseline
    s1_out, p2_cpu, p2_gpu = _timed_cuda_segment(device, enable_timing, seg_stage1)

    def seg_stage2_opt():
        ws = info['ws']
        C = ws['y_hat_so_far'].shape[1]
        ws['params_cat'][:, :C].copy_(ws['y_hat_so_far'])
        ws['params_cat'][:, C:].copy_(info['common_params'])
        scales2, means2 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_2(ws['params_cat'])).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = i_frame_net.process_with_mask(ws['y_q'], scales2, means2, info['mask_2'])
        ws['y_hat_so_far'].add_(y_hat_2)
        return {
            'y_q_2': y_q_2,
            's_hat_2': s_hat_2,
        }

    def seg_stage2_baseline():
        params2 = torch.cat((s1_out['y_hat_so_far'], info['common_params']), dim=1)
        scales2, means2 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_2(params2)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = i_frame_net.process_with_mask(info['y_q'], scales2, means2, info['mask_2'])
        y_hat_so_far = s1_out['y_hat_so_far'] + y_hat_2
        return {
            'y_hat_so_far': y_hat_so_far,
            'y_q_2': y_q_2,
            's_hat_2': s_hat_2,
        }

    seg_stage2 = seg_stage2_opt if ENABLE_PRIOR4X_WORKSPACE_OPT else seg_stage2_baseline
    s2_out, p3_cpu, p3_gpu = _timed_cuda_segment(device, enable_timing, seg_stage2)

    def seg_stage3_opt():
        ws = info['ws']
        C = ws['y_hat_so_far'].shape[1]
        ws['params_cat'][:, :C].copy_(ws['y_hat_so_far'])
        ws['params_cat'][:, C:].copy_(info['common_params'])
        scales3, means3 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_3(ws['params_cat'])).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = i_frame_net.process_with_mask(ws['y_q'], scales3, means3, info['mask_3'])
        ws['y_hat_so_far'].add_(y_hat_3)
        torch.mul(ws['y_hat_so_far'], info['q_dec'], out=ws['y_hat_final'])
        return {
            'y_hat': ws['y_hat_final'],
            'y_q_w_0': info['s_w_0'],
            'y_q_w_1': i_frame_net.single_part_for_writing_4x(s1_out['y_q_1']),
            'y_q_w_2': i_frame_net.single_part_for_writing_4x(s2_out['y_q_2']),
            'y_q_w_3': i_frame_net.single_part_for_writing_4x(y_q_3),
            's_w_0': info['scale_w_0'],
            's_w_1': i_frame_net.single_part_for_writing_4x(s1_out['s_hat_1']),
            's_w_2': i_frame_net.single_part_for_writing_4x(s2_out['s_hat_2']),
            's_w_3': i_frame_net.single_part_for_writing_4x(s_hat_3),
        }

    def seg_stage3_baseline():
        params3 = torch.cat((s2_out['y_hat_so_far'], info['common_params']), dim=1)
        scales3, means3 = i_frame_net.y_spatial_prior(i_frame_net.y_spatial_prior_adaptor_3(params3)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = i_frame_net.process_with_mask(info['y_q'], scales3, means3, info['mask_3'])
        y_hat = (s2_out['y_hat_so_far'] + y_hat_3) * info['q_dec']
        return {
            'y_hat': y_hat,
            'y_q_w_0': info['s_w_0'],
            'y_q_w_1': i_frame_net.single_part_for_writing_4x(s1_out['y_q_1']),
            'y_q_w_2': i_frame_net.single_part_for_writing_4x(s2_out['y_q_2']),
            'y_q_w_3': i_frame_net.single_part_for_writing_4x(y_q_3),
            's_w_0': info['scale_w_0'],
            's_w_1': i_frame_net.single_part_for_writing_4x(s1_out['s_hat_1']),
            's_w_2': i_frame_net.single_part_for_writing_4x(s2_out['s_hat_2']),
            's_w_3': i_frame_net.single_part_for_writing_4x(s_hat_3),
        }

    seg_stage3 = seg_stage3_opt if ENABLE_PRIOR4X_WORKSPACE_OPT else seg_stage3_baseline
    s3_out, p4_cpu, p4_gpu = _timed_cuda_segment(device, enable_timing, seg_stage3)
    prior_profile = {
        'i_prior_paramgen_cpu': float(p0_cpu),
        'i_prior_paramgen_gpu': float(p0_gpu),
        'i_prior_stage0_cpu': float(p1_cpu),
        'i_prior_stage0_gpu': float(p1_gpu),
        'i_prior_stage1_cpu': float(p2_cpu),
        'i_prior_stage1_gpu': float(p2_gpu),
        'i_prior_stage2_cpu': float(p3_cpu),
        'i_prior_stage2_gpu': float(p3_gpu),
        'i_prior_stage3_cpu': float(p4_cpu),
        'i_prior_stage3_gpu': float(p4_gpu),
    }
    return (
        s3_out['y_q_w_0'], s3_out['y_q_w_1'], s3_out['y_q_w_2'], s3_out['y_q_w_3'],
        s3_out['s_w_0'], s3_out['s_w_1'], s3_out['s_w_2'], s3_out['s_w_3'], s3_out['y_hat']
    ), prior_profile


def compress_i_frame_profiled(i_frame_net, x, qp: int, enable_timing: bool = True):
    device = x.device
    curr_q_enc = i_frame_net.q_scale_enc[qp:qp+1, :, :, :]
    curr_q_dec = i_frame_net.q_scale_dec[qp:qp+1, :, :, :]

    def seg1():
        y = i_frame_net.enc(x, curr_q_enc)
        y_pad = i_frame_net.pad_for_y(y)
        z = i_frame_net.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)
        return y, z_hat, z_hat_write

    (y, z_hat, z_hat_write), s1_cpu, s1_gpu = _timed_cuda_segment(device, enable_timing, seg1)

    prior_out, prior_parts = profile_prior4x_breakdown(i_frame_net, y, z_hat, device, enable_timing)
    y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = prior_out
    s2_cpu = sum(prior_parts[k] for k in ['i_prior_paramgen_cpu','i_prior_stage0_cpu','i_prior_stage1_cpu','i_prior_stage2_cpu','i_prior_stage3_cpu'])
    s2_gpu = sum(prior_parts[k] for k in ['i_prior_paramgen_gpu','i_prior_stage0_gpu','i_prior_stage1_gpu','i_prior_stage2_gpu','i_prior_stage3_gpu'])

    x_hat, dec_parts = profile_intra_decoder_breakdown(i_frame_net.dec, y_hat, curr_q_dec, device, enable_timing)
    (d1_cpu, d1_gpu, d2_cpu, d2_gpu, d3_cpu, d3_gpu, d4_cpu, d4_gpu) = dec_parts
    s3_cpu = d1_cpu + d2_cpu + d3_cpu + d4_cpu
    s3_gpu = d1_gpu + d2_gpu + d3_gpu + d4_gpu

    entropy_cpu_t0 = time.time()
    cuda_stream = i_frame_net.get_cuda_stream(device=device, priority=-1)
    cuda_event = torch.cuda.Event()
    cuda_event.record()
    with torch.cuda.stream(cuda_stream):
        cuda_event.wait()
        i_frame_net.entropy_coder.reset()
        i_frame_net.bit_estimator_z.encode_z(z_hat_write, qp)
        i_frame_net.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
        i_frame_net.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
        i_frame_net.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
        i_frame_net.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
        i_frame_net.entropy_coder.flush()
    bit_stream = i_frame_net.entropy_coder.get_encoded_stream()
    cuda_stream.synchronize()
    s4_cpu = time.time() - entropy_cpu_t0
    s4_gpu = 0.0

    result = {
        'bit_stream': bit_stream,
        'x_hat': x_hat,
    }
    i_path_profile = {
        'i_enc_hyper_cpu': float(s1_cpu),
        'i_enc_hyper_gpu': float(s1_gpu),
        'i_prior4x_cpu': float(s2_cpu),
        'i_prior4x_gpu': float(s2_gpu),
        'i_dec_recon_cpu': float(s3_cpu),
        'i_dec_recon_gpu': float(s3_gpu),
        'i_dec_up_cpu': float(d1_cpu),
        'i_dec_up_gpu': float(d1_gpu),
        'i_dec_stack_cpu': float(d2_cpu),
        'i_dec_stack_gpu': float(d2_gpu),
        'i_dec_tail_cpu': float(d3_cpu),
        'i_dec_tail_gpu': float(d3_gpu),
        'i_dec_shuffle_cpu': float(d4_cpu),
        'i_dec_shuffle_gpu': float(d4_gpu),
        'i_entropy_cpu': float(s4_cpu),
        'i_entropy_gpu': float(s4_gpu),
        **prior_parts,
    }
    return result, i_path_profile


def compress_p_frame_profiled(p_frame_net, x, qp: int, enable_timing: bool = True):
    device = x.device
    q_encoder = p_frame_net.q_encoder[qp:qp+1, :, :, :]
    q_decoder = p_frame_net.q_decoder[qp:qp+1, :, :, :]
    q_feature = p_frame_net.q_feature[qp:qp+1, :, :, :]

    feature = None
    x1 = None
    ctx = None
    ctx_t = None
    y = None
    z_hat_write = None
    y_q_w_0 = y_q_w_1 = s_w_0 = s_w_1 = y_hat = None

    def seg_ref():
        return p_frame_net.apply_feature_adaptor()

    feature, s0_cpu, s0_gpu = _timed_cuda_segment(device, enable_timing, seg_ref)

    def seg_feature_part1():
        return p_frame_net.feature_extractor.forward_part1(feature, q_feature)

    (x1, ctx_t), f1_cpu, f1_gpu = _timed_cuda_segment(device, enable_timing, seg_feature_part1)

    def seg_feature_part2():
        return p_frame_net.feature_extractor.forward_part2(x1)

    ctx, f2_cpu, f2_gpu = _timed_cuda_segment(device, enable_timing, seg_feature_part2)
    s1_cpu = f1_cpu + f2_cpu
    s1_gpu = f1_gpu + f2_gpu

    def seg_enc_hyper():
        y = p_frame_net.encoder(x, ctx, q_encoder)
        hyper_inp = p_frame_net.pad_for_y(y)
        z = p_frame_net.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)
        return y, z_hat, z_hat_write

    (y, z_hat, z_hat_write), s2_cpu, s2_gpu = _timed_cuda_segment(device, enable_timing, seg_enc_hyper)

    def seg_prior2x():
        params = p_frame_net.res_prior_param_decoder(z_hat, ctx_t)
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = p_frame_net.compress_prior_2x(y, params, p_frame_net.y_spatial_prior)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

    (y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat), s3_cpu, s3_gpu = _timed_cuda_segment(device, enable_timing, seg_prior2x)

    def seg_decoder():
        return p_frame_net.decoder(y_hat, ctx, q_decoder)

    feature_out, s4_cpu, s4_gpu = _timed_cuda_segment(device, enable_timing, seg_decoder)

    entropy_cpu_t0 = time.time()
    cuda_stream = p_frame_net.get_cuda_stream(device=device, priority=-1)
    cuda_event_z_ready = torch.cuda.Event()
    cuda_event_y_ready = torch.cuda.Event()
    cuda_event_z_ready.record()
    cuda_event_y_ready.record()
    with torch.cuda.stream(cuda_stream):
        p_frame_net.entropy_coder.reset()
        cuda_event_z_ready.wait()
        p_frame_net.bit_estimator_z.encode_z(z_hat_write, qp)
        cuda_event_y_ready.wait()
        p_frame_net.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
        p_frame_net.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
        p_frame_net.entropy_coder.flush()
    bit_stream = p_frame_net.entropy_coder.get_encoded_stream()
    cuda_stream.synchronize()
    s5_cpu = time.time() - entropy_cpu_t0
    s5_gpu = 0.0

    p_frame_net.add_ref_frame(feature_out, None)
    result = {'bit_stream': bit_stream}
    p_path_profile = {
        'p_ref_adaptor_cpu': float(s0_cpu),
        'p_ref_adaptor_gpu': float(s0_gpu),
        'p_feature_ctx_cpu': float(s1_cpu),
        'p_feature_ctx_gpu': float(s1_gpu),
        'p_feat_part1_cpu': float(f1_cpu),
        'p_feat_part1_gpu': float(f1_gpu),
        'p_feat_part2_cpu': float(f2_cpu),
        'p_feat_part2_gpu': float(f2_gpu),
        'p_enc_hyper_cpu': float(s2_cpu),
        'p_enc_hyper_gpu': float(s2_gpu),
        'p_prior2x_cpu': float(s3_cpu),
        'p_prior2x_gpu': float(s3_gpu),
        'p_decoder_cpu': float(s4_cpu),
        'p_decoder_gpu': float(s4_gpu),
        'p_entropy_cpu': float(s5_cpu),
        'p_entropy_gpu': float(s5_gpu),
    }
    return result, p_path_profile


def init_encode_profile() -> Dict:
    return {
        'per_frame': [],
        'by_class': {'I': [], 'P_bootstrap': [], 'P_steady': [], 'P_reset': []},
    }


def append_encode_profile(profile: Dict, frame_profile: Dict):
    profile['per_frame'].append(frame_profile)
    profile['by_class'].setdefault(frame_profile['frame_class'], []).append(frame_profile)


def build_encode_profile_summary(profile: Dict) -> Dict:
    metric_keys = [
        'input_cpu',
        'tensorize_cpu',
        'colorspace_cpu',
        'pad_cpu',
        'prepare_ref_cpu',
        'model_cpu',
        'model_gpu',
        'write_stream_cpu',
        'total_cpu',
        'bit_count',
        'i_enc_hyper_cpu',
        'i_enc_hyper_gpu',
        'i_prior4x_cpu',
        'i_prior4x_gpu',
        'i_dec_recon_cpu',
        'i_dec_recon_gpu',
        'i_dec_up_cpu',
        'i_dec_up_gpu',
        'i_dec_stack_cpu',
        'i_dec_stack_gpu',
        'i_dec_tail_cpu',
        'i_dec_tail_gpu',
        'i_dec_shuffle_cpu',
        'i_dec_shuffle_gpu',
        'i_entropy_cpu',
        'i_entropy_gpu',
        'i_prior_paramgen_cpu',
        'i_prior_paramgen_gpu',
        'i_prior_stage0_cpu',
        'i_prior_stage0_gpu',
        'i_prior_stage1_cpu',
        'i_prior_stage1_gpu',
        'i_prior_stage2_cpu',
        'i_prior_stage2_gpu',
        'i_prior_stage3_cpu',
        'i_prior_stage3_gpu',
        'p_ref_adaptor_cpu',
        'p_ref_adaptor_gpu',
        'p_feature_ctx_cpu',
        'p_feature_ctx_gpu',
        'p_feat_part1_cpu',
        'p_feat_part1_gpu',
        'p_feat_part2_cpu',
        'p_feat_part2_gpu',
        'p_enc_hyper_cpu',
        'p_enc_hyper_gpu',
        'p_prior2x_cpu',
        'p_prior2x_gpu',
        'p_decoder_cpu',
        'p_decoder_gpu',
        'p_entropy_cpu',
        'p_entropy_gpu',
    ]
    summary = {
        'frame_count': len(profile.get('per_frame', [])),
        'frame_classes': {},
        'overall': {},
    }
    for name, records in [('overall', profile.get('per_frame', []))]:
        summary[name] = {
            'count': len(records),
        }
        for key in metric_keys:
            summary[name][key] = _build_scalar_summary([float(r.get(key, 0.0)) for r in records])
    i_records = profile.get('by_class', {}).get('I', [])
    summary['i_path_breakdown'] = {
        'count': len(i_records),
        'i_enc_hyper_cpu': _build_scalar_summary([float(r.get('i_enc_hyper_cpu', 0.0)) for r in i_records]),
        'i_enc_hyper_gpu': _build_scalar_summary([float(r.get('i_enc_hyper_gpu', 0.0)) for r in i_records]),
        'i_prior4x_cpu': _build_scalar_summary([float(r.get('i_prior4x_cpu', 0.0)) for r in i_records]),
        'i_prior4x_gpu': _build_scalar_summary([float(r.get('i_prior4x_gpu', 0.0)) for r in i_records]),
        'i_prior_paramgen_cpu': _build_scalar_summary([float(r.get('i_prior_paramgen_cpu', 0.0)) for r in i_records]),
        'i_prior_paramgen_gpu': _build_scalar_summary([float(r.get('i_prior_paramgen_gpu', 0.0)) for r in i_records]),
        'i_prior_stage0_cpu': _build_scalar_summary([float(r.get('i_prior_stage0_cpu', 0.0)) for r in i_records]),
        'i_prior_stage0_gpu': _build_scalar_summary([float(r.get('i_prior_stage0_gpu', 0.0)) for r in i_records]),
        'i_prior_stage1_cpu': _build_scalar_summary([float(r.get('i_prior_stage1_cpu', 0.0)) for r in i_records]),
        'i_prior_stage1_gpu': _build_scalar_summary([float(r.get('i_prior_stage1_gpu', 0.0)) for r in i_records]),
        'i_prior_stage2_cpu': _build_scalar_summary([float(r.get('i_prior_stage2_cpu', 0.0)) for r in i_records]),
        'i_prior_stage2_gpu': _build_scalar_summary([float(r.get('i_prior_stage2_gpu', 0.0)) for r in i_records]),
        'i_prior_stage3_cpu': _build_scalar_summary([float(r.get('i_prior_stage3_cpu', 0.0)) for r in i_records]),
        'i_prior_stage3_gpu': _build_scalar_summary([float(r.get('i_prior_stage3_gpu', 0.0)) for r in i_records]),
        'i_dec_recon_cpu': _build_scalar_summary([float(r.get('i_dec_recon_cpu', 0.0)) for r in i_records]),
        'i_dec_recon_gpu': _build_scalar_summary([float(r.get('i_dec_recon_gpu', 0.0)) for r in i_records]),
        'i_dec_up_cpu': _build_scalar_summary([float(r.get('i_dec_up_cpu', 0.0)) for r in i_records]),
        'i_dec_up_gpu': _build_scalar_summary([float(r.get('i_dec_up_gpu', 0.0)) for r in i_records]),
        'i_dec_stack_cpu': _build_scalar_summary([float(r.get('i_dec_stack_cpu', 0.0)) for r in i_records]),
        'i_dec_stack_gpu': _build_scalar_summary([float(r.get('i_dec_stack_gpu', 0.0)) for r in i_records]),
        'i_dec_tail_cpu': _build_scalar_summary([float(r.get('i_dec_tail_cpu', 0.0)) for r in i_records]),
        'i_dec_tail_gpu': _build_scalar_summary([float(r.get('i_dec_tail_gpu', 0.0)) for r in i_records]),
        'i_dec_shuffle_cpu': _build_scalar_summary([float(r.get('i_dec_shuffle_cpu', 0.0)) for r in i_records]),
        'i_dec_shuffle_gpu': _build_scalar_summary([float(r.get('i_dec_shuffle_gpu', 0.0)) for r in i_records]),
        'i_entropy_cpu': _build_scalar_summary([float(r.get('i_entropy_cpu', 0.0)) for r in i_records]),
        'i_entropy_gpu': _build_scalar_summary([float(r.get('i_entropy_gpu', 0.0)) for r in i_records]),
    }
    p_records_all = []
    for _fc in ['P_bootstrap', 'P_steady', 'P_reset']:
        p_records_all.extend(profile.get('by_class', {}).get(_fc, []))
    summary['p_path_breakdown'] = {
        'overall': {
            'count': len(p_records_all),
            'p_ref_adaptor_cpu': _build_scalar_summary([float(r.get('p_ref_adaptor_cpu', 0.0)) for r in p_records_all]),
            'p_ref_adaptor_gpu': _build_scalar_summary([float(r.get('p_ref_adaptor_gpu', 0.0)) for r in p_records_all]),
            'p_feature_ctx_cpu': _build_scalar_summary([float(r.get('p_feature_ctx_cpu', 0.0)) for r in p_records_all]),
            'p_feature_ctx_gpu': _build_scalar_summary([float(r.get('p_feature_ctx_gpu', 0.0)) for r in p_records_all]),
            'p_feat_part1_cpu': _build_scalar_summary([float(r.get('p_feat_part1_cpu', 0.0)) for r in p_records_all]),
            'p_feat_part1_gpu': _build_scalar_summary([float(r.get('p_feat_part1_gpu', 0.0)) for r in p_records_all]),
            'p_feat_part2_cpu': _build_scalar_summary([float(r.get('p_feat_part2_cpu', 0.0)) for r in p_records_all]),
            'p_feat_part2_gpu': _build_scalar_summary([float(r.get('p_feat_part2_gpu', 0.0)) for r in p_records_all]),
            'p_enc_hyper_cpu': _build_scalar_summary([float(r.get('p_enc_hyper_cpu', 0.0)) for r in p_records_all]),
            'p_enc_hyper_gpu': _build_scalar_summary([float(r.get('p_enc_hyper_gpu', 0.0)) for r in p_records_all]),
            'p_prior2x_cpu': _build_scalar_summary([float(r.get('p_prior2x_cpu', 0.0)) for r in p_records_all]),
            'p_prior2x_gpu': _build_scalar_summary([float(r.get('p_prior2x_gpu', 0.0)) for r in p_records_all]),
            'p_decoder_cpu': _build_scalar_summary([float(r.get('p_decoder_cpu', 0.0)) for r in p_records_all]),
            'p_decoder_gpu': _build_scalar_summary([float(r.get('p_decoder_gpu', 0.0)) for r in p_records_all]),
            'p_entropy_cpu': _build_scalar_summary([float(r.get('p_entropy_cpu', 0.0)) for r in p_records_all]),
            'p_entropy_gpu': _build_scalar_summary([float(r.get('p_entropy_gpu', 0.0)) for r in p_records_all]),
        },
        'by_class': {}
    }
    for _fc in ['P_bootstrap', 'P_steady', 'P_reset']:
        _records = profile.get('by_class', {}).get(_fc, [])
        summary['p_path_breakdown']['by_class'][_fc] = {
            'count': len(_records),
            'p_ref_adaptor_cpu': _build_scalar_summary([float(r.get('p_ref_adaptor_cpu', 0.0)) for r in _records]),
            'p_ref_adaptor_gpu': _build_scalar_summary([float(r.get('p_ref_adaptor_gpu', 0.0)) for r in _records]),
            'p_feature_ctx_cpu': _build_scalar_summary([float(r.get('p_feature_ctx_cpu', 0.0)) for r in _records]),
            'p_feature_ctx_gpu': _build_scalar_summary([float(r.get('p_feature_ctx_gpu', 0.0)) for r in _records]),
            'p_feat_part1_cpu': _build_scalar_summary([float(r.get('p_feat_part1_cpu', 0.0)) for r in _records]),
            'p_feat_part1_gpu': _build_scalar_summary([float(r.get('p_feat_part1_gpu', 0.0)) for r in _records]),
            'p_feat_part2_cpu': _build_scalar_summary([float(r.get('p_feat_part2_cpu', 0.0)) for r in _records]),
            'p_feat_part2_gpu': _build_scalar_summary([float(r.get('p_feat_part2_gpu', 0.0)) for r in _records]),
            'p_enc_hyper_cpu': _build_scalar_summary([float(r.get('p_enc_hyper_cpu', 0.0)) for r in _records]),
            'p_enc_hyper_gpu': _build_scalar_summary([float(r.get('p_enc_hyper_gpu', 0.0)) for r in _records]),
            'p_prior2x_cpu': _build_scalar_summary([float(r.get('p_prior2x_cpu', 0.0)) for r in _records]),
            'p_prior2x_gpu': _build_scalar_summary([float(r.get('p_prior2x_gpu', 0.0)) for r in _records]),
            'p_decoder_cpu': _build_scalar_summary([float(r.get('p_decoder_cpu', 0.0)) for r in _records]),
            'p_decoder_gpu': _build_scalar_summary([float(r.get('p_decoder_gpu', 0.0)) for r in _records]),
            'p_entropy_cpu': _build_scalar_summary([float(r.get('p_entropy_cpu', 0.0)) for r in _records]),
            'p_entropy_gpu': _build_scalar_summary([float(r.get('p_entropy_gpu', 0.0)) for r in _records]),
        }
    for frame_class in ['I', 'P_bootstrap', 'P_steady', 'P_reset']:
        records = profile.get('by_class', {}).get(frame_class, [])
        if not records:
            continue
        class_summary = {
            'count': len(records),
        }
        for key in metric_keys:
            class_summary[key] = _build_scalar_summary([float(r.get(key, 0.0)) for r in records])
        summary['frame_classes'][frame_class] = class_summary
    return summary


def print_encode_profile_summary(profile_summary: Dict):
    if not profile_summary or profile_summary.get('frame_count', 0) == 0:
        return
        print('[ENC][PROFILE] 分类型统计（mean / p95, 单位 ms）:')
    for frame_class in ['I', 'P_bootstrap', 'P_reset', 'P_steady']:
        s = profile_summary.get('frame_classes', {}).get(frame_class)
        if not s:
            continue
        print(
            f"    {frame_class:<7} n={s['count']:<4d} | "
            f"total={s['total_cpu']['mean'] * 1000:.2f}/{s['total_cpu']['p95'] * 1000:.2f} | "
            f"model_gpu={s['model_gpu']['mean'] * 1000:.2f}/{s['model_gpu']['p95'] * 1000:.2f} | "
            f"model_cpu={s['model_cpu']['mean'] * 1000:.2f}/{s['model_cpu']['p95'] * 1000:.2f} | "
            f"input={s['input_cpu']['mean'] * 1000:.2f} | "
            f"tensor={s['tensorize_cpu']['mean'] * 1000:.2f} | "
            f"pad={s['pad_cpu']['mean'] * 1000:.2f} | "
            f"prep_ref={s['prepare_ref_cpu']['mean'] * 1000:.2f} | "
            f"write={s['write_stream_cpu']['mean'] * 1000:.2f}"
        )
    i_path = profile_summary.get('i_path_breakdown', {})
    if i_path.get('count', 0) > 0:
        print('[ENC][I-PATH] I帧内部四段统计（mean / p95, 单位 ms）:')
        print(
            f"    enc/hyper = {i_path['i_enc_hyper_cpu']['mean'] * 1000:.2f}/{i_path['i_enc_hyper_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_enc_hyper_gpu']['mean'] * 1000:.2f}/{i_path['i_enc_hyper_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"    prior_4x  = {i_path['i_prior4x_cpu']['mean'] * 1000:.2f}/{i_path['i_prior4x_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior4x_gpu']['mean'] * 1000:.2f}/{i_path['i_prior4x_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      param_gen = {i_path['i_prior_paramgen_cpu']['mean'] * 1000:.2f}/{i_path['i_prior_paramgen_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior_paramgen_gpu']['mean'] * 1000:.2f}/{i_path['i_prior_paramgen_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      stage0    = {i_path['i_prior_stage0_cpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage0_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior_stage0_gpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage0_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      stage1    = {i_path['i_prior_stage1_cpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage1_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior_stage1_gpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage1_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      stage2    = {i_path['i_prior_stage2_cpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage2_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior_stage2_gpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage2_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      stage3    = {i_path['i_prior_stage3_cpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage3_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_prior_stage3_gpu']['mean'] * 1000:.2f}/{i_path['i_prior_stage3_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"    dec/recon = {i_path['i_dec_recon_cpu']['mean'] * 1000:.2f}/{i_path['i_dec_recon_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_dec_recon_gpu']['mean'] * 1000:.2f}/{i_path['i_dec_recon_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      upsample  = {i_path['i_dec_up_cpu']['mean'] * 1000:.2f}/{i_path['i_dec_up_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_dec_up_gpu']['mean'] * 1000:.2f}/{i_path['i_dec_up_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      convstack = {i_path['i_dec_stack_cpu']['mean'] * 1000:.2f}/{i_path['i_dec_stack_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_dec_stack_gpu']['mean'] * 1000:.2f}/{i_path['i_dec_stack_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      tail      = {i_path['i_dec_tail_cpu']['mean'] * 1000:.2f}/{i_path['i_dec_tail_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_dec_tail_gpu']['mean'] * 1000:.2f}/{i_path['i_dec_tail_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"      shuffle   = {i_path['i_dec_shuffle_cpu']['mean'] * 1000:.2f}/{i_path['i_dec_shuffle_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_dec_shuffle_gpu']['mean'] * 1000:.2f}/{i_path['i_dec_shuffle_gpu']['p95'] * 1000:.2f} gpu"
        )
        print(
            f"    entropy   = {i_path['i_entropy_cpu']['mean'] * 1000:.2f}/{i_path['i_entropy_cpu']['p95'] * 1000:.2f} cpu | "
            f"{i_path['i_entropy_gpu']['mean'] * 1000:.2f}/{i_path['i_entropy_gpu']['p95'] * 1000:.2f} gpu"
        )
    p_path = profile_summary.get('p_path_breakdown', {})
    if p_path.get('overall', {}).get('count', 0) > 0:
        print('[ENC][P-PATH] P帧内部六段统计（mean / p95, 单位 ms）:')
        for frame_class in ['P_bootstrap', 'P_reset', 'P_steady']:
            s = p_path.get('by_class', {}).get(frame_class, {})
            if s.get('count', 0) <= 0:
                continue
            print(
                f"    {frame_class:<11} n={s['count']:<4d} | "
                f"ref={s['p_ref_adaptor_cpu']['mean'] * 1000:.2f}/{s['p_ref_adaptor_cpu']['p95'] * 1000:.2f} cpu | "
                f"feat={s['p_feature_ctx_cpu']['mean'] * 1000:.2f}/{s['p_feature_ctx_cpu']['p95'] * 1000:.2f} cpu | "
                f"f1={s['p_feat_part1_cpu']['mean'] * 1000:.2f}/{s['p_feat_part1_cpu']['p95'] * 1000:.2f} cpu | "
                f"f2={s['p_feat_part2_cpu']['mean'] * 1000:.2f}/{s['p_feat_part2_cpu']['p95'] * 1000:.2f} cpu | "
                f"enc={s['p_enc_hyper_cpu']['mean'] * 1000:.2f}/{s['p_enc_hyper_cpu']['p95'] * 1000:.2f} cpu | "
                f"prior={s['p_prior2x_cpu']['mean'] * 1000:.2f}/{s['p_prior2x_cpu']['p95'] * 1000:.2f} cpu | "
                f"dec={s['p_decoder_cpu']['mean'] * 1000:.2f}/{s['p_decoder_cpu']['p95'] * 1000:.2f} cpu | "
                f"ent={s['p_entropy_cpu']['mean'] * 1000:.2f}/{s['p_entropy_cpu']['p95'] * 1000:.2f} cpu"
            )


# ============================================================
# YUV420 → GPU Tensor（纯PyTorch实现，替代不稳定的CUDA扩展）
# ============================================================

class YUV420Tensorizer:
    def __init__(self, device, use_fp16: bool, use_pinned_memory: bool = True, async_h2d: bool = True):
        self.device = torch.device(device)
        self.use_fp16 = bool(use_fp16)
        self.float_dtype = torch.float16 if self.use_fp16 else torch.float32
        self.use_pinned_memory = bool(use_pinned_memory) and self.device.type == 'cuda'
        self.async_h2d = bool(async_h2d) and self.use_pinned_memory
        self._workspace_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        self._copy_stream = torch.cuda.Stream(device=self.device) if self.async_h2d else None

    def _get_workspace(self, h: int, w: int) -> Dict[str, torch.Tensor]:
        key = (h, w)
        ws = self._workspace_cache.get(key)
        if ws is not None:
            return ws
        uv_h, uv_w = h // 2, w // 2
        ws = {
            'host_y': torch.empty((h, w), dtype=torch.uint8, pin_memory=self.use_pinned_memory),
            'host_uv': torch.empty((2, uv_h, uv_w), dtype=torch.uint8, pin_memory=self.use_pinned_memory),
            'dev_y': torch.empty((h, w), dtype=torch.uint8, device=self.device),
            'dev_uv': torch.empty((2, uv_h, uv_w), dtype=torch.uint8, device=self.device),
        }
        self._workspace_cache[key] = ws
        return ws

    def convert(self, y, uv) -> torch.Tensor:
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if isinstance(uv, torch.Tensor):
            uv = uv.detach().cpu().numpy()

        if y.ndim == 3:
            y = y[0]
        if uv.ndim == 4:
            uv = uv[0]
        if uv.ndim != 3 or uv.shape[0] != 2:
            raise ValueError(f"Unexpected uv shape: {uv.shape}")

        y = np.ascontiguousarray(y)
        uv = np.ascontiguousarray(uv)
        h, w = int(y.shape[0]), int(y.shape[1])
        ws = self._get_workspace(h, w)

        ws['host_y'].copy_(torch.from_numpy(y), non_blocking=False)
        ws['host_uv'].copy_(torch.from_numpy(uv), non_blocking=False)

        if self._copy_stream is not None:
            with torch.cuda.stream(self._copy_stream):
                ws['dev_y'].copy_(ws['host_y'], non_blocking=True)
                ws['dev_uv'].copy_(ws['host_uv'], non_blocking=True)
                y_f = ws['dev_y'].to(dtype=self.float_dtype).unsqueeze(0).unsqueeze(0)
                uv_f = ws['dev_uv'].to(dtype=self.float_dtype).unsqueeze(0)
                uv_up = F.interpolate(uv_f, size=(h, w), mode='nearest')
                x = torch.cat((y_f, uv_up), dim=1)
                x.mul_(1.0 / 255.0)
            torch.cuda.current_stream(self.device).wait_stream(self._copy_stream)
            return x

        ws['dev_y'].copy_(ws['host_y'], non_blocking=False)
        ws['dev_uv'].copy_(ws['host_uv'], non_blocking=False)
        y_f = ws['dev_y'].to(dtype=self.float_dtype).unsqueeze(0).unsqueeze(0)
        uv_f = ws['dev_uv'].to(dtype=self.float_dtype).unsqueeze(0)
        uv_up = F.interpolate(uv_f, size=(h, w), mode='nearest')
        x = torch.cat((y_f, uv_up), dim=1)
        x.mul_(1.0 / 255.0)
        return x


_YUV420_TENSORIZER_CACHE: Dict[Tuple[str, bool, bool, bool], YUV420Tensorizer] = {}


def _get_yuv420_tensorizer(args_or_dict: Dict, device, use_fp16: bool) -> Optional[YUV420Tensorizer]:
    if not bool(args_or_dict.get('optimize_yuv420_tensorize', False)):
        return None
    dev = torch.device(device)
    use_pinned = bool(args_or_dict.get('yuv420_use_pinned_memory', True))
    async_h2d = bool(args_or_dict.get('yuv420_async_h2d', True))
    key = (str(dev), bool(use_fp16), use_pinned, async_h2d)
    tensorizer = _YUV420_TENSORIZER_CACHE.get(key)
    if tensorizer is None:
        tensorizer = YUV420Tensorizer(dev, use_fp16=use_fp16, use_pinned_memory=use_pinned, async_h2d=async_h2d)
        _YUV420_TENSORIZER_CACHE[key] = tensorizer
    return tensorizer


def yuv420_frame_to_gpu_tensor(y, uv, device: str, use_fp16: bool = True,
                               tensorizer: Optional[YUV420Tensorizer] = None) -> torch.Tensor:
    """
    y:  (1, H, W) or (H, W)   uint8
    uv: (2, H/2, W/2)          uint8
    return: (1, 3, H, W)       float16/float32, [0,1]
    """
    
    if tensorizer is not None:
        return tensorizer.convert(y, uv)
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if isinstance(uv, torch.Tensor):
        uv = uv.detach().cpu().numpy()
    if y.ndim == 3:
        y = y[0]
    H, W = y.shape
    if uv.ndim == 3:
        u = uv[0]
        v = uv[1]
    elif uv.ndim == 4:
        u = uv[0, 0]
        v = uv[0, 1]
    else:
        raise ValueError(f"Unexpected uv shape: {uv.shape}")
    # 杞负torch tensor骞剁Щ鑷矴PU
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.uint8)
    u_t = torch.from_numpy(u).to(device=device, dtype=torch.uint8)
    v_t = torch.from_numpy(v).to(device=device, dtype=torch.uint8)
    # 涓婇噰鏍稶V (鏈€杩戦偦)
    u_up = torch.repeat_interleave(torch.repeat_interleave(u_t, 2, dim=0), 2, dim=1)
    v_up = torch.repeat_interleave(torch.repeat_interleave(v_t, 2, dim=0), 2, dim=1)
    # 鍚堝苟涓篩UV444
    yuv444 = torch.stack([y_t, u_up, v_up], dim=0)  # (3, H, W)
    # 褰掍竴鍖?
    x = yuv444.float() / 255.0
    if use_fp16:
        x = x.half()
    return x.unsqueeze(0)  # (1,3,H,W)


# ============================================================
# 鍗曞抚璇诲彇锛堜紶缁熻矾寰?/ camera 鍙傝€冭矾寰勶級
# ============================================================


def get_src_frame_fast(args_or_dict, src_reader, device, use_fp16=True):
    if args_or_dict['src_type'] == 'yuv420':
        y, uv = src_reader.read_one_frame()
        if y is None:
            return None, None, None, None, None
        yuv444 = ycbcr420_to_444_np(y, uv)
        x = torch.from_numpy(yuv444).to(device=device,
                                        dtype=torch.float16 if use_fp16 else torch.float32) / 255.0
        x = x.unsqueeze(0)
        y = y[0, :, :]
        u = uv[0, :, :]
        v = uv[1, :, :]
        rgb = None
    else:
        rgb = src_reader.read_one_frame()
        if rgb is None:
            return None, None, None, None, None
        x = torch.from_numpy(rgb).to(device=device,
                                     dtype=torch.float16 if use_fp16 else torch.float32) / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = rgb2ycbcr(x)
        y = u = v = None
    return x, y, u, v, rgb


def get_distortion(args_or_dict, x_hat, y, u, v, rgb):
    if args_or_dict['src_type'] == 'yuv420':
        y_rec, uv_rec = yuv_444_to_420(x_hat)
        y_rec = torch.clamp(y_rec * 255, 0, 255).round().squeeze(0).cpu().numpy()
        uv_rec = torch.clamp(uv_rec * 255, 0, 255).round().squeeze(0).cpu().numpy()
        y_rec = y_rec[0, :, :]
        u_rec = uv_rec[0, :, :]
        v_rec = uv_rec[1, :, :]

        psnr_y = calc_psnr(y, y_rec)
        psnr_u = calc_psnr(u, u_rec)
        psnr_v = calc_psnr(v, v_rec)
        psnr = (6 * psnr_y + psnr_u + psnr_v) / 8

        if args_or_dict['calc_ssim']:
            ssim_y = calc_msssim(y, y_rec)
            ssim_u = calc_msssim(u, u_rec)
            ssim_v = calc_msssim(v, v_rec)
        else:
            ssim_y = ssim_u = ssim_v = 0.0
        ssim = (6 * ssim_y + ssim_u + ssim_v) / 8
        curr_psnr = [psnr, psnr_y, psnr_u, psnr_v]
        curr_ssim = [ssim, ssim_y, ssim_u, ssim_v]
    else:
        rgb_rec = ycbcr2rgb(x_hat)
        rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().squeeze(0).cpu().numpy()
        psnr = calc_psnr(rgb, rgb_rec)
        if args_or_dict['calc_ssim']:
            msssim = calc_msssim_rgb(rgb, rgb_rec)
        else:
            msssim = 0.0
        curr_psnr = [psnr]
        curr_ssim = [msssim]
    return curr_psnr, curr_ssim


def make_zero_metrics(src_type: str, frame_count: int):
    if src_type == 'yuv420':
        psnrs = [[0.0, 0.0, 0.0, 0.0] for _ in range(frame_count)]
        msssims = [[0.0, 0.0, 0.0, 0.0] for _ in range(frame_count)]
    else:
        psnrs = [[0.0] for _ in range(frame_count)]
        msssims = [[0.0] for _ in range(frame_count)]
    return psnrs, msssims


def build_metric_args(task_args: Dict) -> Dict:
    if task_args.get('input_mode', 'file') != 'camera':
        return task_args
    ref_path = task_args.get('camera_reference_yuv') or task_args.get('curr_src_yuv_path')
    if ref_path is None:
        ref_path = task_args['curr_bin_path'].replace('.bin', '_src.yuv')
    if not os.path.exists(ref_path):
        raise FileNotFoundError(
            f'camera decode/metrics mode requires a reference I420 file, not found: {ref_path}. '
            'Enable --camera_dump_reference=1 in encode stage, or pass --camera_reference_yuv explicitly.'
        )
    metric_args = dict(task_args)
    metric_args['src_type'] = 'yuv420'
    metric_args['src_path'] = ref_path
    return metric_args


# ============================================================
# 缂栫爜闃舵锛坋ncode-only锛?
# ============================================================

def warmup_models_for_camera(camera_reader, p_frame_net, i_frame_net, task_args: Dict, device, use_fp16: bool):
    min_warmup = int(task_args.get('min_warmup', 5))
    if min_warmup <= 0:
        return
    pic_h = task_args['src_height']
    pic_w = task_args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_h, pic_w, 16)
    intra_period = task_args['intra_period']
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    tensorizer = _get_yuv420_tensorizer(task_args, device, use_fp16)
    last_seq = 0
    with torch.inference_mode():
        for wi in range(min_warmup):
            item = camera_reader.wait_latest_after(last_seq, timeout_sec=3.0)
            if item is None:
                break
            last_seq = item['seq']
            x = yuv420_frame_to_gpu_tensor(item['y'], item['uv'], str(device), use_fp16, tensorizer=tensorizer)
            x_padded = replicate_pad(x, padding_b, padding_r)
            if wi == 0 or (intra_period > 0 and wi % intra_period == 0):
                encoded = i_frame_net.compress(x_padded, task_args['qp_i'])
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
            else:
                if len(p_frame_net.dpb) == 0:
                    encoded = i_frame_net.compress(x_padded, task_args['qp_i'])
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded['x_hat'])
                else:
                    fa_idx = index_map[wi % len(index_map)]
                    curr_qp = p_frame_net.shift_qp(task_args['qp_p'], fa_idx)
                    _ = p_frame_net.compress(x_padded, curr_qp)
    p_frame_net.clear_dpb()
    p_frame_net.set_curr_poc(0)


def encode_camera_segment(camera_reader, p_frame_net, i_frame_net, task_args: Dict,
                          segment_name: str, bin_path: str, json_path: str, src_yuv_path: Optional[str],
                          encoded_frame_target: Optional[int], preview_hook=None) -> Dict:
    device = next(i_frame_net.parameters()).device
    use_fp16 = task_args.get('precision', 'fp16') == 'fp16'
    verbose_json = task_args['verbose_json']
    verbose = task_args['verbose']
    pic_h = task_args['src_height']
    pic_w = task_args['src_width']
    intra_period = task_args['intra_period']
    reset_interval = task_args['reset_interval']
    padding_r, padding_b = DMCI.get_padding_size(pic_h, pic_w, 16)
    use_two_entropy_coders = pic_h * pic_w > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    time_metrics = TimeMetrics(task_args)
    time_metrics.set_device(device)
    tensorizer = _get_yuv420_tensorizer(task_args, device, use_fp16)
    time_metrics.min_warmup = 0

    create_folder(os.path.dirname(bin_path), True)
    src_writer = None
    if src_yuv_path is not None and task_args.get('camera_dump_reference', True):
        create_folder(os.path.dirname(src_yuv_path), True)
        src_writer = YUV420Writer(src_yuv_path, pic_w, pic_h)

    p_frame_net.clear_dpb()
    p_frame_net.set_curr_poc(0)

    output_stream = io.BytesIO()
    sps_helper = SPSHelper()
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    last_qp = 0
    last_seq = camera_reader.get_captured_total()
    session_capture_start = camera_reader.get_captured_total()
    session_start = time.time()
    frame_types = []
    bits = []
    enc_cpu_times = []
    enc_gpu_times = []
    encode_profile = init_encode_profile()
    encoded_frames = 0
    dropped_frames = 0
    draining = False
    stop_after_seq = None
    stop_after_enqueued = None
    status_last = 0.0
    status_interval = float(task_args.get('camera_drain_print_interval', 1.0))
    buffer_mode = str(task_args.get('camera_buffer_mode', 'fifo')).lower()
    session_enqueued_start = camera_reader.get_fifo_enqueued_total() if buffer_mode == 'fifo' else 0
    session_fifo_drop_start = camera_reader.get_fifo_overflow_drop_total() if buffer_mode == 'fifo' else 0

    try:
        with torch.inference_mode():
            while True:
                if encoded_frame_target is not None and encoded_frames >= encoded_frame_target and not draining:
                    break

                if preview_hook is not None:
                    signal = preview_hook('idle' if not draining else 'drain')
                    if isinstance(signal, dict) and signal.get('action') == 'stop' and not draining:
                        draining = True
                        if buffer_mode == 'fifo':
                            camera_reader.set_fifo_accepting(False)
                            stop_after_enqueued = int(camera_reader.get_fifo_enqueued_total())
                            stop_after_seq = int(signal.get('stop_after_seq', camera_reader.get_captured_total()))
                            session_target_enqueued = max(0, stop_after_enqueued - session_enqueued_start)
                            _clear_inline_status()
                            print(f">>> 收到结束命令，停止当前采集并开始drain，当前阶段目标入队={session_target_enqueued}累计入队 {stop_after_enqueued}目标序号={stop_after_seq}")
                        else:
                            stop_after_seq = int(signal.get('stop_after_seq', camera_reader.get_captured_total()))
                            _clear_inline_status()
                            print(f">>> 收到结束命令，停止当前采集并开始drain，目标序号 {stop_after_seq}")

                frame_cpu_total_start = time.time()
                input_start = time.time()
                if buffer_mode == 'fifo':
                    item = camera_reader.wait_next_frame(timeout_sec=task_args.get('camera_timeout_sec', 2.0))
                    if item is None:
                        if draining:
                            current_enqueued = camera_reader.get_fifo_enqueued_total()
                            if stop_after_enqueued is not None and current_enqueued >= stop_after_enqueued and camera_reader.get_queue_size() == 0:
                                break
                        continue
                    if draining and stop_after_enqueued is not None and int(item.get('fifo_enqueued_total', 0)) > stop_after_enqueued:
                        continue
                else:
                    item = camera_reader.wait_latest_after(last_seq, timeout_sec=task_args.get('camera_timeout_sec', 2.0))
                    if item is None:
                        if draining:
                            break
                        continue
                    last_seq = item['seq']
                    dropped_frames += int(item.get('skipped', 0))
                    if draining and stop_after_seq is not None and int(item['seq']) > stop_after_seq:
                        break
                input_cpu = time.time() - input_start

                y_raw = item['y']
                uv_raw = item['uv']
                if src_writer is not None:
                    src_writer.write_one_frame(y_raw, uv_raw)

                tensor_start = time.time()
                x = yuv420_frame_to_gpu_tensor(y_raw, uv_raw, str(device), use_fp16, tensorizer=tensorizer)
                tensorize_cpu = time.time() - tensor_start
                colorspace_cpu = 0.0

                pad_start = time.time()
                x_padded = replicate_pad(x, padding_b, padding_r)
                pad_cpu = time.time() - pad_start

                is_i_frame, frame_class, use_ada_i = classify_frame_type(encoded_frames, intra_period, reset_interval)
                prepare_ref_cpu = 0.0
                if (not is_i_frame) and use_ada_i == 1:
                    prep_start = time.time()
                    p_frame_net.prepare_feature_adaptor_i(last_qp)
                    prepare_ref_cpu = time.time() - prep_start

                p_profile_enabled = bool(task_args.get('profile_p_path', False))
                cpu_start, gpu_start, gpu_end = time_metrics.record_start()
                i_path_profile = {
                    'i_enc_hyper_cpu': 0.0, 'i_enc_hyper_gpu': 0.0,
                    'i_prior4x_cpu': 0.0, 'i_prior4x_gpu': 0.0,
                    'i_dec_recon_cpu': 0.0, 'i_dec_recon_gpu': 0.0,
                    'i_dec_up_cpu': 0.0, 'i_dec_up_gpu': 0.0,
                    'i_dec_stack_cpu': 0.0, 'i_dec_stack_gpu': 0.0,
                    'i_dec_tail_cpu': 0.0, 'i_dec_tail_gpu': 0.0,
                    'i_dec_shuffle_cpu': 0.0, 'i_dec_shuffle_gpu': 0.0,
                    'i_entropy_cpu': 0.0, 'i_entropy_gpu': 0.0,
                }
                p_path_profile = {
                    'p_ref_adaptor_cpu': 0.0, 'p_ref_adaptor_gpu': 0.0,
                    'p_feature_ctx_cpu': 0.0, 'p_feature_ctx_gpu': 0.0,
                'p_feat_part1_cpu': 0.0, 'p_feat_part1_gpu': 0.0,
                'p_feat_part2_cpu': 0.0, 'p_feat_part2_gpu': 0.0,
                    'p_enc_hyper_cpu': 0.0, 'p_enc_hyper_gpu': 0.0,
                    'p_prior2x_cpu': 0.0, 'p_prior2x_gpu': 0.0,
                    'p_decoder_cpu': 0.0, 'p_decoder_gpu': 0.0,
                    'p_entropy_cpu': 0.0, 'p_entropy_gpu': 0.0,
                }
                if is_i_frame or len(p_frame_net.dpb) == 0:
                    curr_qp = task_args['qp_i']
                    sps = {'sps_id': -1, 'height': pic_h, 'width': pic_w, 'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0}
                    if task_args.get('profile_i_path', False):
                        encoded, i_path_profile = compress_i_frame_profiled(i_frame_net, x_padded, curr_qp, enable_timing=True)
                    else:
                        encoded = i_frame_net.compress(x_padded, curr_qp)
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded['x_hat'])
                    frame_types.append(0)
                    is_i_frame = True
                    frame_class = 'I'
                else:
                    fa_idx = index_map[encoded_frames % len(index_map)]
                    curr_qp = p_frame_net.shift_qp(task_args['qp_p'], fa_idx)
                    last_qp = curr_qp
                    sps = {'sps_id': -1, 'height': pic_h, 'width': pic_w, 'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': use_ada_i}
                    if p_profile_enabled:
                        encoded, p_path_profile = compress_p_frame_profiled(p_frame_net, x_padded, curr_qp, enable_timing=True)
                        if all(float(p_path_profile.get(k, 0.0)) == 0.0 for k in [
                            'p_ref_adaptor_cpu','p_feature_ctx_cpu','p_feat_part1_cpu','p_feat_part2_cpu','p_enc_hyper_cpu','p_prior2x_cpu','p_decoder_cpu','p_entropy_cpu'
                        ]):
                            raise RuntimeError('P-path profiling is enabled but all six stage stats are zero; please verify this patched script is running and --profile_p_path=1 is applied.')
                    else:
                        encoded = p_frame_net.compress(x_padded, curr_qp)
                    frame_types.append(1)
                cpu_time, gpu_time = time_metrics.record_end(cpu_start, gpu_start, gpu_end)
                if cpu_time is not None:
                    enc_cpu_times.append(cpu_time)
                if gpu_time is not None:
                    enc_gpu_times.append(gpu_time)

                write_start = time.time()
                sps_id, sps_new = sps_helper.get_sps_id(sps)
                sps['sps_id'] = sps_id
                sps_bytes = write_sps(output_stream, sps) if sps_new else 0
                stream_bytes = write_ip(output_stream, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
                write_stream_cpu = time.time() - write_start
                bit_count = (stream_bytes + sps_bytes) * 8
                bits.append(bit_count)
                total_cpu = time.time() - frame_cpu_total_start

                append_encode_profile(encode_profile, {
                    'frame_idx': int(encoded_frames),
                    'frame_class': frame_class,
                    'input_cpu': _to_float(input_cpu),
                    'tensorize_cpu': _to_float(tensorize_cpu),
                    'colorspace_cpu': _to_float(colorspace_cpu),
                    'pad_cpu': _to_float(pad_cpu),
                    'prepare_ref_cpu': _to_float(prepare_ref_cpu),
                    'model_cpu': _to_float(cpu_time),
                    'model_gpu': _to_float(gpu_time),
                    'write_stream_cpu': _to_float(write_stream_cpu),
                    'total_cpu': _to_float(total_cpu),
                    'bit_count': int(bit_count),
                    'qp': int(curr_qp),
                    'camera_seq': int(item.get('seq', -1)),
                    **i_path_profile,
                    **p_path_profile,
                })
                encoded_frames += 1

                now = time.time()
                if now - status_last >= status_interval:
                    status_last = now
                    fps = encoded_frames / max(1e-9, (now - session_start))
                    gpu_ms = (sum(enc_gpu_times) / max(1, len(enc_gpu_times)) * 1000.0) if enc_gpu_times else 0.0
                    if buffer_mode == 'fifo':
                        if draining and stop_after_enqueued is not None:
                            captured_now = max(0, stop_after_enqueued - session_enqueued_start)
                        else:
                            captured_now = max(0, camera_reader.get_fifo_enqueued_total() - session_enqueued_start)
                        remain = max(0, captured_now - encoded_frames)
                        prefix = '[Drain]' if draining else '[Enc]'
                        _inline_status(f"{prefix} 采集: {captured_now} | 已编码 {encoded_frames} | 剩余: {remain} | GPU: {gpu_ms:.2f} ms/帧| 速率: {fps:.2f} fps")
                    else:
                        cap_now = max(0, camera_reader.get_captured_total() - session_capture_start)
                        remain = max(0, cap_now - encoded_frames)
                        prefix = '[Drain]' if draining else '[Enc]'
                        _inline_status(f"{prefix} 采集: {captured_now} | 已编码 {encoded_frames} | 剩余: {remain} | GPU: {gpu_ms:.2f} ms/帧| 速率: {fps:.2f} fps")

                if draining and buffer_mode == 'fifo' and stop_after_enqueued is not None:
                    target_frames = max(0, stop_after_enqueued - session_enqueued_start)
                    if camera_reader.get_queue_size() == 0 and encoded_frames >= target_frames:
                        break
    finally:
        if src_writer is not None:
            src_writer.close()

    _clear_inline_status()
    session_end = time.time()
    session_wall_time = max(1e-9, session_end - session_start)
    session_capture_end = camera_reader.get_captured_total()
    if buffer_mode == 'fifo':
        session_enqueued_end = stop_after_enqueued if stop_after_enqueued is not None else camera_reader.get_fifo_enqueued_total()
        captured_during_session = max(0, session_enqueued_end - session_enqueued_start)
        fifo_overflow_drops = max(0, camera_reader.get_fifo_overflow_drop_total() - session_fifo_drop_start)
    else:
        captured_during_session = max(0, session_capture_end - session_capture_start)
        fifo_overflow_drops = 0
    stream_data = output_stream.getvalue()
    with open(bin_path, 'wb') as f:
        f.write(stream_data)
    output_stream.close()

    avg_enc_cpu, avg_enc_gpu = time_metrics.compute_averages(enc_cpu_times, enc_gpu_times)
    profile_summary = build_encode_profile_summary(encode_profile)
    psnrs, msssims = make_zero_metrics('yuv420', len(frame_types))
    log_result = generate_log_json(len(frame_types), pic_h * pic_w, 0.0, frame_types, bits, psnrs, msssims,
                                   verbose=verbose_json, avg_encoding_time=avg_enc_gpu, avg_decoding_time=0.0)
    log_result['stage'] = 'encode'
    log_result['input_mode'] = 'camera'
    log_result['segment_name'] = segment_name
    log_result['curr_bin_path'] = bin_path
    log_result['curr_src_yuv_path'] = src_yuv_path
    log_result['time_metrics_mode'] = 'standard_sync_per_frame'
    log_result['avg_encoding_time_gpu'] = avg_enc_gpu
    log_result['avg_encoding_time_cpu'] = avg_enc_cpu
    log_result['encoding_times_gpu'] = enc_gpu_times
    log_result['encoding_times_cpu'] = enc_cpu_times
    log_result['encode_profile_per_frame'] = encode_profile['per_frame']
    log_result['encode_profile_summary'] = profile_summary
    log_result['session_wall_time'] = session_wall_time
    log_result['effective_encoded_fps'] = len(frame_types) / session_wall_time if session_wall_time > 0 else 0.0
    log_result['captured_during_session'] = captured_during_session
    log_result['captured_fps'] = captured_during_session / session_wall_time if session_wall_time > 0 else 0.0
    log_result['dropped_frames_latest_policy'] = max(0, captured_during_session - len(frame_types)) if buffer_mode != 'fifo' else 0
    log_result['skipped_frames_before_encode'] = dropped_frames
    log_result['fifo_overflow_drop_frames'] = fifo_overflow_drops
    log_result['frame_num'] = len(frame_types)
    with open(json_path, 'w') as fp:
        json.dump(log_result, fp, indent=2)
    if verbose:
        time_metrics.print_time_summary('ENC', len(frame_types), avg_enc_cpu, avg_enc_gpu)
        print_encode_profile_summary(profile_summary)
        dropped_total = log_result['dropped_frames_latest_policy'] + fifo_overflow_drops
        print(f"     {segment_name} q={task_args['qp_i']} | 会话时长={session_wall_time:.3f}s | 采集={captured_during_session}帧| 编码={len(frame_types)}帧| 丢帧={dropped_total} | 有效编码FPS={log_result['effective_encoded_fps']:.2f}")
        if buffer_mode == 'fifo' and int(task_args.get('camera_buffer_max_frames', 0)) > 0 and fifo_overflow_drops > 0:
            print(f"[WARN][FIFO] queue limit={int(task_args.get('camera_buffer_max_frames', 0))} frames, overflow dropped {fifo_overflow_drops} frames; set --camera_buffer_max_frames 0 or a larger value to preserve backlog after pressing E.")
    return log_result


def run_camera_encode_fixed_frames(p_frame_net, i_frame_net, task_args: Dict) -> Dict:
    camera_reader = get_src_reader(task_args)
    camera_reader.start_background_capture()
    device = next(i_frame_net.parameters()).device
    use_fp16 = task_args.get('precision', 'fp16') == 'fp16'
    warmup_models_for_camera(camera_reader, p_frame_net, i_frame_net, task_args, device, use_fp16)
    if str(task_args.get('camera_buffer_mode', 'fifo')).lower() == 'fifo':
        dropped = camera_reader.clear_fifo_queue()
        if dropped > 0 and task_args.get('verbose', 0):
            print(f'[camera] warmup 结束后清空 FIFO 队列 {dropped} 帧')
    result = encode_camera_segment(
        camera_reader,
        p_frame_net,
        i_frame_net,
        task_args,
        segment_name=task_args['seq'],
        bin_path=task_args['curr_bin_path'],
        json_path=task_args['curr_json_path'],
        src_yuv_path=task_args.get('curr_src_yuv_path'),
        encoded_frame_target=task_args['frame_num'],
        preview_hook=None,
    )
    camera_reader.close()
    return result


def i420_to_bgr_preview(y, uv):
    if cv2 is None:
        return None
    h = y.shape[1]
    w = y.shape[2]
    y_plane = y[0]
    u = uv[0]
    v = uv[1]
    i420 = np.concatenate([y_plane.reshape(-1), u.reshape(-1), v.reshape(-1)], axis=0)
    i420 = i420.reshape(h * 3 // 2, w)
    return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)


def build_camera_segment_paths(base_dir: str, prefix: str, seg_idx: int, qp_i: int):
    seg_name = f'{prefix}_seg{seg_idx:03d}_q{qp_i}'
    bin_path = os.path.join(base_dir, f'{seg_name}.bin')
    json_path = os.path.join(base_dir, f'{seg_name}_enc.json')
    src_yuv_path = os.path.join(base_dir, f'{seg_name}_src.yuv')
    return seg_name, bin_path, json_path, src_yuv_path


_INLINE_STATUS_LEN = 0

def _inline_status(msg: str):
    global _INLINE_STATUS_LEN
    pad = max(0, _INLINE_STATUS_LEN - len(msg))
    print('\r' + msg + (' ' * pad), end='', flush=True)
    _INLINE_STATUS_LEN = len(msg)


def _clear_inline_status():
    global _INLINE_STATUS_LEN
    if _INLINE_STATUS_LEN > 0:
        print('\r' + (' ' * _INLINE_STATUS_LEN) + '\r', end='', flush=True)
        _INLINE_STATUS_LEN = 0


def start_terminal_command_listener(cmd_queue: Queue):
    def _worker():
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                break
            except Exception:
                continue
            if cmd:
                cmd_queue.put(cmd)
            if cmd in ('q','quit','exit'):
                break
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


def drain_manual_commands(cmd_queue: Queue):
    cmds = []
    while True:
        try:
            cmds.append(cmd_queue.get_nowait())
        except Exception:
            break
    return cmds


def run_camera_manual_session_app(args: argparse.Namespace):
    if cv2 is None and args.camera_preview:
        print('[WARN] OpenCV GUI is unavailable, preview is disabled automatically.')
        args.camera_preview = False
    if args.stage != 'encode':
        raise ValueError('manual_session currently supports stage=encode only.')
    if args.rate_num != 1:
        raise ValueError('manual_session currently supports rate_num=1 only.')

    global i_frame_net_global, p_frame_net_global
    clear_prior4x_caches()
    qp_i = args.qp_i[0] if args.qp_i is not None else int(DMC.get_qp_num() // 2)
    qp_p = args.qp_p[0] if args.qp_p is not None else qp_i
    task_args = {
        'rate_idx': 0, 'qp_i': qp_i, 'qp_p': qp_p, 'force_intra': args.force_intra, 'reset_interval': args.reset_interval,
        'seq': args.camera_session_prefix, 'src_type': 'camera', 'src_height': args.camera_height, 'src_width': args.camera_width,
        'intra_period': 1 if args.force_intra else (args.force_intra_period if args.force_intra_period > 0 else 32),
        'frame_num': args.camera_frames, 'calc_ssim': False, 'dataset_path': '', 'write_stream': True, 'check_existing': False,
        'stream_path': args.stream_path, 'save_decoded_frame': False, 'ds_name': args.camera_ds_name, 'verbose': args.verbose,
        'verbose_json': args.verbose_json, 'precision': args.precision, 'min_warmup': args.min_warmup, 'rt_mode': 1, 'stage': 'encode',
        'enc_log_suffix': args.enc_log_suffix, 'dec_log_suffix': args.dec_log_suffix, 'decode_bin_override': None,
        'profile_i_path': args.profile_i_path, 'profile_p_path': args.profile_p_path, 'strict_frame_sync': True,
        'max_nonframe_nals': args.max_nonframe_nals, 'input_mode': 'camera', 'camera_device': args.camera_device,
        'optimize_yuv420_tensorize': args.optimize_yuv420_tensorize,
        'yuv420_use_pinned_memory': args.yuv420_use_pinned_memory,
        'yuv420_async_h2d': args.yuv420_async_h2d,
        'camera_format': normalize_camera_format(args.camera_format), 'camera_fps': args.camera_fps, 'camera_timeout_sec': args.camera_timeout_sec,
        'camera_v4l2_io_mode': args.camera_v4l2_io_mode, 'camera_dump_reference': args.camera_dump_reference,
        'camera_reference_yuv': args.camera_reference_yuv, 'camera_buffer_mode': args.camera_buffer_mode,
        'camera_buffer_max_frames': args.camera_buffer_max_frames, 'camera_drain_print_interval': args.camera_drain_print_interval,
    }
    base_dir = os.path.join(args.stream_path, args.camera_ds_name)
    create_folder(base_dir, True)
    camera_reader = get_src_reader(task_args)
    camera_reader.start_background_capture()
    device = next(i_frame_net_global.parameters()).device
    use_fp16 = args.precision == 'fp16'
    warmup_models_for_camera(camera_reader, p_frame_net_global, i_frame_net_global, task_args, device, use_fp16)

    print('manual_session mode: when preview=0, use terminal commands for control.')
    print('terminal commands: start/s begin segment, stop/e/end stop current segment, quit/q/exit quit app.')
    if args.camera_buffer_mode == 'fifo' and args.camera_buffer_max_frames > 0:
        print(f"[INFO][FIFO] queue limit enabled: {args.camera_buffer_max_frames} frames.")
    manifest = {'mode': 'manual_session', 'camera_device': args.camera_device, 'camera_format': normalize_camera_format(args.camera_format),
                'camera_width': args.camera_width, 'camera_height': args.camera_height, 'camera_fps': args.camera_fps, 'segments': []}
    seg_idx = 0
    cmd_queue = Queue()
    start_terminal_command_listener(cmd_queue)
    window_name = 'Jetson Camera Preview'
    if args.camera_preview and cv2 is not None:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            cmds = drain_manual_commands(cmd_queue)
            if args.camera_preview and cv2 is not None:
                latest = camera_reader.peek_latest()
                if latest is not None:
                    frame = i420_to_bgr_preview(latest['y'], latest['uv'])
                    if frame is not None:
                        cv2.putText(frame, 'IDLE', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
                        cv2.imshow(window_name, frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord(args.camera_start_key): cmds.append('start')
                elif k == ord(args.camera_stop_key): cmds.append('stop')
                elif k == ord(args.camera_quit_key): cmds.append('quit')
            else:
                time.sleep(0.05)

            if any(c in ('quit','q','exit') for c in cmds):
                print('>>> 退出 manual_session')
                break
            if not any(c in ('start','s') for c in cmds):
                continue

            dropped = camera_reader.clear_fifo_queue() if args.camera_buffer_mode == 'fifo' else 0
            if dropped > 0:
                print(f'>>> Cleared {dropped} queued frames before starting this segment.')
            camera_reader.set_fifo_accepting(True)
            seg_idx += 1
            seg_name, bin_path, json_path, src_yuv_path = build_camera_segment_paths(base_dir, args.camera_session_prefix, seg_idx, qp_i)
            print(f'>>> 开始录制并编码: {seg_name}')
            state = {'stop_announced': False, 'quit_after_segment': False}

            def preview_hook(_reason):
                local_cmds = drain_manual_commands(cmd_queue)
                if args.camera_preview and cv2 is not None:
                    latest_now = camera_reader.peek_latest()
                    if latest_now is not None:
                        frame_now = i420_to_bgr_preview(latest_now['y'], latest_now['uv'])
                        if frame_now is not None:
                            cv2.putText(frame_now, 'REC', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                            cv2.imshow(window_name, frame_now)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord(args.camera_stop_key):
                        local_cmds.append('stop')
                    elif k == ord(args.camera_quit_key):
                        local_cmds.append('quit')
                if any(c in ('quit','q','exit') for c in local_cmds):
                    state['quit_after_segment'] = True
                if any(c in ('stop','e','end','quit','q','exit') for c in local_cmds):
                    if not state['stop_announced']:
                        state['stop_announced'] = True
                        _clear_inline_status()
                        if state['quit_after_segment']:
                            print('>>> Received q; handled as stop-first policy: stop capture, drain current segment, then exit manual_session.')
                        else:
                            print('>>> Stopping capture and entering drain stage for current segment.')
                        return {'action':'stop','stop_after_seq': camera_reader.get_captured_total(), 'quit_after_segment': state['quit_after_segment']}
                return 'continue'

            result = encode_camera_segment(camera_reader, p_frame_net_global, i_frame_net_global, task_args, seg_name, bin_path, json_path, src_yuv_path if args.camera_dump_reference else None, None, preview_hook)
            manifest['segments'].append(result)
            print(f">>> 当前阶段已经全部编码: {seg_name} | 编码帧{result.get('frame_num',0)} | 队列余量={camera_reader.get_queue_size()}")
            if state.get('quit_after_segment', False):
                print('>>> q-stop flow completed, exiting manual_session.')
                break
    finally:
        if args.camera_preview and cv2 is not None:
            cv2.destroyAllWindows()
        camera_reader.close()

    with open(args.output_path, 'w') as fp:
        json.dump(manifest, fp, indent=2)
    print(f'会话清单已写入 {args.output_path}')



def encode_one_point(p_frame_net, i_frame_net, task_args: Dict) -> Dict:
    """

    """
    if task_args.get('input_mode', 'file') == 'camera':
        return run_camera_encode_fixed_frames(p_frame_net, i_frame_net, task_args)

    if not task_args['write_stream']:
        raise ValueError('encode-only 闇€瑕?--write_stream 1')
    if task_args['check_existing'] and os.path.exists(task_args['curr_json_path']) and os.path.exists(task_args['curr_bin_path']):
        with open(task_args['curr_json_path']) as f:
            log_result = json.load(f)
            if log_result.get('i_frame_num', 0) + log_result.get('p_frame_num', 0) == task_args['frame_num']:
                return log_result

    frame_num = int(task_args.get('frame_num', -1))
    decode_to_eof = frame_num <= 0
    intra_period = task_args['intra_period']
    reset_interval = task_args['reset_interval']
    verbose_json = task_args['verbose_json']
    verbose = task_args['verbose']
    device = next(i_frame_net.parameters()).device
    use_fp16 = task_args.get('precision', 'fp16') == 'fp16'
    min_warmup = int(task_args.get('min_warmup', 5))
    pic_h = task_args['src_height']
    pic_w = task_args['src_width']
    padding_r, padding_b = DMCI.get_padding_size(pic_h, pic_w, 16)
    use_two_entropy_coders = pic_h * pic_w > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    time_metrics = TimeMetrics(task_args)
    time_metrics.set_device(device)
    tensorizer = _get_yuv420_tensorizer(task_args, device, use_fp16)

    src_reader = get_src_reader(task_args)
    index_map = [0, 1, 0, 2, 0, 2, 0, 2]
    with torch.inference_mode():
        for wi in range(min_warmup):
            x, *_ = get_src_frame_fast(task_args, src_reader, device, use_fp16)
            if x is None:
                break
            x_padded = replicate_pad(x, padding_b, padding_r)
            if wi == 0 or (intra_period > 0 and wi % intra_period == 0):
                encoded = i_frame_net.compress(x_padded, task_args['qp_i'])
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
            else:
                if len(p_frame_net.dpb) == 0:
                    encoded = i_frame_net.compress(x_padded, task_args['qp_i'])
                    p_frame_net.clear_dpb()
                    p_frame_net.add_ref_frame(None, encoded['x_hat'])
                else:
                    fa_idx = index_map[wi % len(index_map)]
                    curr_qp = p_frame_net.shift_qp(task_args['qp_p'], fa_idx)
                    _ = p_frame_net.compress(x_padded, curr_qp)


    src_reader.close()
    src_reader = get_src_reader(task_args)
    p_frame_net.clear_dpb()
    p_frame_net.set_curr_poc(0)
    time_metrics.min_warmup = 0

    frame_queue = None
    reader_thread = None
    if task_args['src_type'] == 'yuv420' and not decode_to_eof:
        frame_queue = Queue(maxsize=8)
        def _reader_worker():
            for _ in range(frame_num):
                y, uv = src_reader.read_one_frame()
                if y is None:
                    break
                frame_queue.put((y, uv))
            frame_queue.put((None, None))
        reader_thread = threading.Thread(target=_reader_worker, daemon=True)
        reader_thread.start()

    frame_types = []
    bits = []
    enc_cpu_times = []
    enc_gpu_times = []
    encode_profile = init_encode_profile()
    output_stream = io.BytesIO()
    sps_helper = SPSHelper()
    last_qp = 0
    frame_idx = 0
    with torch.inference_mode():
        while True:
            if not decode_to_eof and frame_idx >= frame_num:
                break

            frame_cpu_total_start = time.time()
            input_start = time.time()
            colorspace_cpu = 0.0
            if task_args['src_type'] == 'yuv420':
                if frame_queue is not None:
                    y_raw, uv_raw = frame_queue.get()
                else:
                    y_raw, uv_raw = src_reader.read_one_frame()
                if y_raw is None:
                    break
                input_cpu = time.time() - input_start
                tensor_start = time.time()
                x = yuv420_frame_to_gpu_tensor(y_raw, uv_raw, str(device), use_fp16, tensorizer=tensorizer)
                tensorize_cpu = time.time() - tensor_start
            else:
                rgb = src_reader.read_one_frame()
                if rgb is None:
                    break
                input_cpu = time.time() - input_start
                tensor_start = time.time()
                x = torch.from_numpy(rgb).to(device=device,
                                             dtype=torch.float16 if use_fp16 else torch.float32) / 255.0
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                tensorize_cpu = time.time() - tensor_start
                colorspace_start = time.time()
                x = rgb2ycbcr(x)
                colorspace_cpu = time.time() - colorspace_start

            pad_start = time.time()
            x_padded = replicate_pad(x, padding_b, padding_r)
            pad_cpu = time.time() - pad_start

            is_i_frame, frame_class, use_ada_i = classify_frame_type(frame_idx, intra_period, reset_interval)
            prepare_ref_cpu = 0.0
            if (not is_i_frame) and use_ada_i == 1 and len(p_frame_net.dpb) > 0:
                prep_start = time.time()
                p_frame_net.prepare_feature_adaptor_i(last_qp)
                prepare_ref_cpu = time.time() - prep_start

            p_profile_enabled = bool(task_args.get('profile_p_path', False))
            cpu_start, gpu_start, gpu_end = time_metrics.record_start()
            i_path_profile = {
                'i_enc_hyper_cpu': 0.0, 'i_enc_hyper_gpu': 0.0,
                'i_prior4x_cpu': 0.0, 'i_prior4x_gpu': 0.0,
                'i_dec_recon_cpu': 0.0, 'i_dec_recon_gpu': 0.0,
                'i_dec_up_cpu': 0.0, 'i_dec_up_gpu': 0.0,
                'i_dec_stack_cpu': 0.0, 'i_dec_stack_gpu': 0.0,
                'i_dec_tail_cpu': 0.0, 'i_dec_tail_gpu': 0.0,
                'i_dec_shuffle_cpu': 0.0, 'i_dec_shuffle_gpu': 0.0,
                'i_entropy_cpu': 0.0, 'i_entropy_gpu': 0.0,
                'i_prior_paramgen_cpu': 0.0, 'i_prior_paramgen_gpu': 0.0,
                'i_prior_stage0_cpu': 0.0, 'i_prior_stage0_gpu': 0.0,
                'i_prior_stage1_cpu': 0.0, 'i_prior_stage1_gpu': 0.0,
                'i_prior_stage2_cpu': 0.0, 'i_prior_stage2_gpu': 0.0,
                'i_prior_stage3_cpu': 0.0, 'i_prior_stage3_gpu': 0.0,
            }
            p_path_profile = {
                'p_ref_adaptor_cpu': 0.0, 'p_ref_adaptor_gpu': 0.0,
                'p_feature_ctx_cpu': 0.0, 'p_feature_ctx_gpu': 0.0,
                'p_feat_part1_cpu': 0.0, 'p_feat_part1_gpu': 0.0,
                'p_feat_part2_cpu': 0.0, 'p_feat_part2_gpu': 0.0,
                'p_enc_hyper_cpu': 0.0, 'p_enc_hyper_gpu': 0.0,
                'p_prior2x_cpu': 0.0, 'p_prior2x_gpu': 0.0,
                'p_decoder_cpu': 0.0, 'p_decoder_gpu': 0.0,
                'p_entropy_cpu': 0.0, 'p_entropy_gpu': 0.0,
            }
            if is_i_frame or len(p_frame_net.dpb) == 0:
                curr_qp = task_args['qp_i']
                sps = {'sps_id': -1, 'height': pic_h, 'width': pic_w,
                       'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': 0}
                if task_args.get('profile_i_path', False):
                    encoded, i_path_profile = compress_i_frame_profiled(i_frame_net, x_padded, curr_qp, enable_timing=True)
                else:
                    encoded = i_frame_net.compress(x_padded, curr_qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, encoded['x_hat'])
                frame_types.append(0)
                is_i_frame = True
                frame_class = 'I'
            else:
                fa_idx = index_map[frame_idx % len(index_map)]
                curr_qp = p_frame_net.shift_qp(task_args['qp_p'], fa_idx)
                last_qp = curr_qp
                sps = {'sps_id': -1, 'height': pic_h, 'width': pic_w,
                       'ec_part': 1 if use_two_entropy_coders else 0, 'use_ada_i': use_ada_i}
                if p_profile_enabled:
                    encoded, p_path_profile = compress_p_frame_profiled(p_frame_net, x_padded, curr_qp, enable_timing=True)
                    if all(float(p_path_profile.get(k, 0.0)) == 0.0 for k in [
                        'p_ref_adaptor_cpu','p_feature_ctx_cpu','p_enc_hyper_cpu','p_prior2x_cpu','p_decoder_cpu','p_entropy_cpu'
                    ]):
                        raise RuntimeError('P-path profiling is enabled but all six stage stats are zero; please verify you are running this patched offline script.')
                else:
                    encoded = p_frame_net.compress(x_padded, curr_qp)
                frame_types.append(1)
            cpu_time, gpu_time = time_metrics.record_end(cpu_start, gpu_start, gpu_end)
            if cpu_time is not None:
                enc_cpu_times.append(cpu_time)
            if gpu_time is not None:
                enc_gpu_times.append(gpu_time)

            write_start = time.time()
            sps_id, sps_new = sps_helper.get_sps_id(sps)
            sps['sps_id'] = sps_id
            sps_bytes = 0
            if sps_new:
                sps_bytes = write_sps(output_stream, sps)
            stream_bytes = write_ip(output_stream, is_i_frame, sps_id, curr_qp, encoded['bit_stream'])
            write_stream_cpu = time.time() - write_start
            bit_count = (stream_bytes + sps_bytes) * 8
            bits.append(bit_count)
            total_cpu = time.time() - frame_cpu_total_start

            append_encode_profile(encode_profile, {
                'frame_idx': int(frame_idx),
                'frame_class': frame_class,
                'input_cpu': _to_float(input_cpu),
                'tensorize_cpu': _to_float(tensorize_cpu),
                'colorspace_cpu': _to_float(colorspace_cpu),
                'pad_cpu': _to_float(pad_cpu),
                'prepare_ref_cpu': _to_float(prepare_ref_cpu),
                'model_cpu': _to_float(cpu_time),
                'model_gpu': _to_float(gpu_time),
                'write_stream_cpu': _to_float(write_stream_cpu),
                'total_cpu': _to_float(total_cpu),
                'bit_count': int(bit_count),
                'qp': int(curr_qp),
                **i_path_profile,
                **p_path_profile,
            })
            frame_idx += 1

    src_reader.close()
    if reader_thread is not None:
        reader_thread.join()
    with open(task_args['curr_bin_path'], 'wb') as f:
        f.write(output_stream.getvalue())
    output_stream.close()
    avg_enc_cpu, avg_enc_gpu = time_metrics.compute_averages(enc_cpu_times, enc_gpu_times)
    profile_summary = build_encode_profile_summary(encode_profile)
    psnrs, msssims = make_zero_metrics(task_args['src_type'], len(frame_types))
    log_result = generate_log_json(
        len(frame_types), pic_h * pic_w, 0.0, frame_types, bits, psnrs, msssims,
        verbose=verbose_json, avg_encoding_time=avg_enc_gpu, avg_decoding_time=0.0,
    )
    log_result['stage'] = 'encode'
    log_result['curr_bin_path'] = task_args['curr_bin_path']
    log_result['time_metrics_mode'] = 'standard_sync_per_frame'
    log_result['avg_encoding_time_gpu'] = avg_enc_gpu
    log_result['avg_encoding_time_cpu'] = avg_enc_cpu
    log_result['encoding_times_gpu'] = enc_gpu_times
    log_result['encoding_times_cpu'] = enc_cpu_times
    log_result['encode_profile_per_frame'] = encode_profile['per_frame']
    log_result['encode_profile_summary'] = profile_summary
    with open(task_args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    if verbose:
        time_metrics.print_time_summary('ENC', len(frame_types), avg_enc_cpu, avg_enc_gpu)
        print_encode_profile_summary(profile_summary)
        print(f"     {task_args['seq']} q={task_args['qp_i']}, size={os.path.getsize(task_args['curr_bin_path']) / 1024 / 1024:.3f} MB")
    return log_result


# ============================================================
#
# ============================================================

def decode_one_point(p_frame_net, i_frame_net, task_args: Dict) -> Dict:

    if task_args.get('stage') == 'decode' and task_args.get('frame_num', -1) > 0:
        print(f"decode-only: overriding frame_num={task_args['frame_num']} to -1 (decode to EOF)")
        task_args['frame_num'] = -1

    if not os.path.exists(task_args['curr_bin_path']):
        raise FileNotFoundError(f"bin not found: {task_args['curr_bin_path']}")

    frame_num = int(task_args.get('frame_num', -1))
    decode_to_eof = frame_num <= 0
    save_decoded = task_args['save_decoded_frame'] and (task_args.get('rt_mode', 0) == 0)
    verbose_json = task_args['verbose_json']
    verbose = task_args['verbose']
    rt_mode = task_args.get('rt_mode', 0)
    device = next(i_frame_net.parameters()).device
    use_fp16 = task_args.get('precision', 'fp16') == 'fp16'
    pic_h = task_args['src_height']
    pic_w = task_args['src_width']
    use_two_entropy_coders = pic_h * pic_w > 1280 * 720
    i_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    p_frame_net.set_use_two_entropy_coders(use_two_entropy_coders)
    time_metrics = TimeMetrics(task_args)
    time_metrics.set_device(device)

    with open(task_args['curr_bin_path'], 'rb') as f:
        input_buff = io.BytesIO(f.read())

    src_reader_dec = None
    recon_writer = None
    video_writer = None
    metric_args = None


    if task_args.get('save_decoded_video', False):
        if cv2 is None:
            print("[WARN] OpenCV not available, cannot save decoded video")
        else:
            video_fps = task_args.get('decoded_video_fps', 0)
            if video_fps <= 0:
                # 灏濊瘯鑾峰彇鍘熷瑙嗛甯х巼
                if task_args.get('src_type') == 'yuv420':
                    video_fps = task_args.get('camera_fps', 30)
                else:
                    video_fps = 30
            ext = task_args.get('decoded_video_ext', 'avi')
            codec = task_args.get('decoded_video_codec', 'MJPG')
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out_video_path = task_args['curr_bin_path'].replace('.bin', f'_decoded.{ext}')
            create_folder(os.path.dirname(out_video_path), True)
            video_writer = cv2.VideoWriter(out_video_path, fourcc, video_fps, (pic_w, pic_h))
            if not video_writer.isOpened():
                print(f"[WARN] Failed to open video writer: {out_video_path}")
                video_writer = None


    if task_args.get('save_decoded_yuv', False):
        recon_writer = YUV420Writer(task_args['curr_rec_path'], pic_w, pic_h)


    if rt_mode == 0 or task_args.get('save_decoded_video') or task_args.get('save_decoded_yuv'):
        metric_args = build_metric_args(task_args)
        src_reader_dec = get_src_reader(metric_args)

    p_frame_net.clear_dpb()
    p_frame_net.set_curr_poc(0)
    sps_helper = SPSHelper()
    frame_types = []
    bits = []
    psnrs = []
    msssims = []
    dec_cpu_times = []
    dec_gpu_times = []
    decoded_frame_num = 0
    sps_overhead_bytes = 0

    with torch.inference_mode():
        nonframe_nals = 0
        ref_frame_idx = 0
        while decode_to_eof or decoded_frame_num < frame_num:
            nal_start_pos = input_buff.tell()
            try:
                header = read_header(input_buff)
            except (struct.error, EOFError):

                break
            if header is None:
                break
            while header['nal_type'] == NalType.NAL_SPS:
                sps = read_sps_remaining(input_buff, header['sps_id'])
                sps_helper.add_sps_by_id(sps)
                nal_end_pos = input_buff.tell()
                sps_overhead_bytes += (nal_end_pos - nal_start_pos)
                nal_start_pos = input_buff.tell()
                try:
                    header = read_header(input_buff)
                except (struct.error, EOFError):

                    header = None
                    break
                if header is None:
                    break
                nonframe_nals += 1
                if nonframe_nals > int(task_args.get('max_nonframe_nals', 1000)):
                    raise RuntimeError(f"Too many non-frame NALs (> {task_args.get('max_nonframe_nals')}). Bitstream may be corrupted.")
            if header is None:
                break
            if header['nal_type'] not in (NalType.NAL_I, NalType.NAL_P):
                if input_buff.tell() == nal_start_pos:
                    raise RuntimeError('Bitstream cursor did not advance on non-frame NAL.')
                continue
            cpu_start, gpu_start, gpu_end = time_metrics.record_start()
            sps_id = header['sps_id']
            sps = sps_helper.get_sps_by_id(sps_id)
            qp, bit_stream = read_ip_remaining(input_buff)
            nal_end_pos = input_buff.tell()
            nal_bytes = nal_end_pos - nal_start_pos
            frame_bits = (nal_bytes + sps_overhead_bytes) * 8
            sps_overhead_bytes = 0
            if header['nal_type'] == NalType.NAL_I:
                decoded = i_frame_net.decompress(bit_stream, sps, qp)
                p_frame_net.clear_dpb()
                p_frame_net.add_ref_frame(None, decoded['x_hat'])
                cur_frame_type = 0
            else:
                if len(p_frame_net.dpb) == 0:
                    if task_args.get('strict_frame_sync', True):
                        raise RuntimeError('Encountered P-frame when DPB is empty. Stream parsing likely desynced.')
                    else:
                        _ = time_metrics.record_end(cpu_start, gpu_start, gpu_end)
                        continue
                if sps.get('use_ada_i', 0):
                    p_frame_net.reset_ref_feature()
                decoded = p_frame_net.decompress(bit_stream, sps, qp)
                cur_frame_type = 1
            recon = decoded['x_hat']
            x_hat = recon[:, :, :pic_h, :pic_w]
            cpu_time, gpu_time = time_metrics.record_end(cpu_start, gpu_start, gpu_end)
            if cpu_time is not None:
                dec_cpu_times.append(cpu_time)
            if gpu_time is not None:
                dec_gpu_times.append(gpu_time)


            if rt_mode == 0 and src_reader_dec is not None:
                if task_args.get('strict_frame_sync', True) and ref_frame_idx != decoded_frame_num:
                    raise RuntimeError(f'Frame sync mismatch before reading ref: ref_frame_idx={ref_frame_idx}, decoded_frame_num={decoded_frame_num}')
                x, y, u, v, rgb = get_src_frame_fast(metric_args, src_reader_dec, device, use_fp16)
                if x is None:
                    raise RuntimeError('Reference video ended earlier than decoded frames.')
                curr_psnr, curr_ssim = get_distortion(metric_args, x_hat, y, u, v, rgb)
                psnrs.append(curr_psnr)
                msssims.append(curr_ssim)
                ref_frame_idx += 1


            if recon_writer is not None:
                y_rec, uv_rec = yuv_444_to_420(x_hat)
                y_rec = torch.clamp(y_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                uv_rec = torch.clamp(uv_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                recon_writer.write_one_frame(y_rec.squeeze(0).cpu().numpy(), uv_rec.squeeze(0).cpu().numpy())


            if video_writer is not None:

                rgb_rec = ycbcr2rgb(x_hat)
                rgb_rec = torch.clamp(rgb_rec * 255, 0, 255).round().to(dtype=torch.uint8)
                bgr = rgb_rec.squeeze(0).cpu().numpy().transpose(1,2,0)[:,:,::-1]
                video_writer.write(bgr)

            bits.append(frame_bits)
            frame_types.append(cur_frame_type)
            decoded_frame_num += 1

    input_buff.close()
    if src_reader_dec is not None:
        src_reader_dec.close()
    if recon_writer is not None:
        recon_writer.close()
    if video_writer is not None:
        video_writer.release()

    avg_dec_cpu, avg_dec_gpu = time_metrics.compute_averages(dec_cpu_times, dec_gpu_times)
    if rt_mode == 1:
        metric_type = 'yuv420' if task_args.get('input_mode', 'file') == 'camera' else task_args['src_type']
        psnrs, msssims = make_zero_metrics(metric_type, decoded_frame_num)
    log_result = generate_log_json(
        decoded_frame_num, pic_h * pic_w, 0.0, frame_types, bits, psnrs, msssims,
        verbose=verbose_json, avg_encoding_time=0.0, avg_decoding_time=avg_dec_gpu,
    )
    log_result['stage'] = 'decode'
    log_result['curr_bin_path'] = task_args['curr_bin_path']
    log_result['time_metrics_mode'] = 'standard_sync_per_frame'
    log_result['avg_decoding_time_gpu'] = avg_dec_gpu
    log_result['avg_decoding_time_cpu'] = avg_dec_cpu
    log_result['decoding_times_gpu'] = dec_gpu_times
    log_result['decoding_times_cpu'] = dec_cpu_times
    log_result['rt_mode'] = rt_mode
    log_result['input_mode'] = task_args.get('input_mode', 'file')
    if metric_args is not None:
        log_result['curr_src_yuv_path'] = metric_args.get('src_path')
    with open(task_args['curr_json_path'], 'w') as fp:
        json.dump(log_result, fp, indent=2)
    if verbose:
        time_metrics.print_time_summary('DEC', decoded_frame_num, avg_dec_cpu, avg_dec_gpu)
        print(f"     {task_args['seq']} q={task_args['qp_i']}, size={os.path.getsize(task_args['curr_bin_path']) / 1024 / 1024:.3f} MB, rt_mode={rt_mode}")
    return log_result


# ============================================================
# 端到端
# ============================================================

def run_one_point_both(p_frame_net, i_frame_net, task_args: Dict) -> Dict:
    total_start_time = time.time()
    if task_args.get('input_mode', 'file') == 'camera' and task_args.get('camera_control_mode') == 'manual_session':
        raise ValueError('manual_session does not support stage=both; run encode first, then decode separately.')

    if task_args.get('input_mode', 'file') == 'camera' and task_args.get('rt_mode', 0) == 0 and task_args.get('camera_dump_reference', True) is not True:
        raise ValueError('camera + both + rt_mode=0 requires camera_dump_reference=1 for strict decode PSNR alignment.')

    enc_json_path = task_args['curr_json_path']
    task_args_enc = dict(task_args)
    task_args_enc['curr_json_path'] = enc_json_path.replace('.json', task_args.get('enc_log_suffix', '_enc.json'))
    enc_log = encode_one_point(p_frame_net, i_frame_net, task_args_enc)
    task_args_dec = dict(task_args)
    task_args_dec['curr_json_path'] = enc_json_path.replace('.json', task_args.get('dec_log_suffix', '_dec.json'))
    if task_args.get('input_mode', 'file') == 'camera':
        task_args_dec['curr_src_yuv_path'] = enc_log.get('curr_src_yuv_path')
        task_args_dec['camera_reference_yuv'] = enc_log.get('curr_src_yuv_path')
    dec_log = decode_one_point(p_frame_net, i_frame_net, task_args_dec)
    total_time = time.time() - total_start_time
    merged = dec_log
    merged['stage'] = 'both'
    merged['avg_encoding_time_gpu'] = enc_log.get('avg_encoding_time_gpu', 0.0)
    merged['avg_encoding_time_cpu'] = enc_log.get('avg_encoding_time_cpu', 0.0)
    merged['encoding_times_gpu'] = enc_log.get('encoding_times_gpu', [])
    merged['encoding_times_cpu'] = enc_log.get('encoding_times_cpu', [])
    merged['encode_profile_per_frame'] = enc_log.get('encode_profile_per_frame', [])
    merged['encode_profile_summary'] = enc_log.get('encode_profile_summary', {})
    merged['avg_decoding_time_gpu'] = dec_log.get('avg_decoding_time_gpu', 0.0)
    merged['avg_decoding_time_cpu'] = dec_log.get('avg_decoding_time_cpu', 0.0)
    merged['decoding_times_gpu'] = dec_log.get('decoding_times_gpu', [])
    merged['decoding_times_cpu'] = dec_log.get('decoding_times_cpu', [])
    merged['total_end_to_end_time'] = total_time
    merged['frames_processed'] = dec_log.get('frame_num', 0)
    merged['avg_total_time_per_frame'] = total_time / max(1, merged['frames_processed'])
    merged['avg_frame_encoding_time'] = enc_log.get('avg_encoding_time_gpu', 0.0)
    merged['avg_frame_decoding_time'] = dec_log.get('avg_decoding_time_gpu', 0.0)
    merged['curr_bin_path'] = task_args['curr_bin_path']
    merged['curr_src_yuv_path'] = enc_log.get('curr_src_yuv_path')
    merged['enc_log_path'] = task_args_enc['curr_json_path']
    merged['dec_log_path'] = task_args_dec['curr_json_path']
    with open(enc_json_path, 'w') as fp:
        json.dump(merged, fp, indent=2)
    return merged


# ============================================================
# worker构造task_args /路径 / 调用阶段函数
# ============================================================

def worker_wrapper(task_args: Dict) -> Dict:
    global i_frame_net_global, p_frame_net_global
    clear_prior4x_caches()

    bin_folder = os.path.join(task_args['stream_path'], task_args['ds_name'])
    create_folder(bin_folder, True)
    if task_args.get('input_mode', 'file') == 'camera':
        task_args['src_path'] = None
        task_args['bin_folder'] = bin_folder
        task_args['curr_bin_path'] = os.path.join(bin_folder, f"{task_args['seq']}_q{task_args['qp_i']}.bin")
        task_args['curr_rec_path'] = task_args['curr_bin_path'].replace('.bin', '.yuv')
        task_args['curr_src_yuv_path'] = task_args.get('camera_reference_yuv') or task_args['curr_bin_path'].replace('.bin', '_src.yuv')
    else:
        sub_dir_name = task_args['seq']
        task_args['src_path'] = os.path.join(task_args['dataset_path'], sub_dir_name)
        task_args['bin_folder'] = bin_folder
        task_args['curr_bin_path'] = os.path.join(bin_folder, f"{task_args['seq']}_q{task_args['qp_i']}.bin")
        task_args['curr_rec_path'] = task_args['curr_bin_path'].replace('.bin', '.yuv')
    base_json_path = task_args['curr_bin_path'].replace('.bin', '.json')
    stage = task_args.get('stage', 'both')
    if stage == 'encode':
        task_args['curr_json_path'] = task_args['curr_bin_path'].replace('.bin', task_args.get('enc_log_suffix', '_enc.json'))
        result = encode_one_point(p_frame_net_global, i_frame_net_global, task_args)
    elif stage == 'decode':
        task_args['curr_json_path'] = task_args['curr_bin_path'].replace('.bin', task_args.get('dec_log_suffix', '_dec.json'))
        if task_args.get('decode_bin_override'):
            task_args['curr_bin_path'] = task_args['decode_bin_override']
            if task_args.get('input_mode', 'file') == 'camera' and task_args.get('camera_reference_yuv') is None:
                task_args['curr_src_yuv_path'] = task_args['curr_bin_path'].replace('.bin', '_src.yuv')
            task_args['curr_rec_path'] = task_args['curr_bin_path'].replace('.bin', '_decoded.yuv')
        result = decode_one_point(p_frame_net_global, i_frame_net_global, task_args)
    else:
        task_args['curr_json_path'] = base_json_path
        result = run_one_point_both(p_frame_net_global, i_frame_net_global, task_args)
    result['ds_name'] = task_args['ds_name']
    result['seq'] = task_args['seq']
    result['rate_idx'] = task_args['rate_idx']
    result['qp_i'] = task_args['qp_i']
    result['qp_p'] = task_args.get('qp_p', task_args['qp_i'])
    result['input_mode'] = task_args.get('input_mode', 'file')
    return result


def build_camera_tasks(args: argparse.Namespace, qp_i: List[int], qp_p: List[int]) -> List[Dict]:
    if args.camera_width != 1280 or args.camera_height != 720:
        print('[WARN] Current implementation is tuned for 720p60. Other resolutions may work but need camera-side validation.')
    if args.camera_fps != 60:
        print('[WARN] camera_fps is not 60. If your 720p camera is fixed at 60fps, keep --camera_fps 60.')
    if args.rate_num > 1:
        print('[WARN] camera mode with rate_num>1 samples different content per QP; for strict RD comparison, dump I420 and run file mode.')

    frame_num = args.force_frame_num if args.force_frame_num > 0 else args.camera_frames
    intra_period = 1 if args.force_intra else (args.force_intra_period if args.force_intra_period > 0 else 32)
    seq_name = f"{args.camera_session_prefix}_{normalize_camera_format(args.camera_format).lower()}_{args.camera_width}x{args.camera_height}_{args.camera_fps}fps"
    tasks = []
    for rate_idx in range(args.rate_num):
        cur_args = {
            'rate_idx': rate_idx,
            'qp_i': qp_i[rate_idx],
            'qp_p': qp_p[rate_idx],
            'force_intra': args.force_intra,
            'reset_interval': args.reset_interval,
            'seq': seq_name,
            'src_type': 'camera',
            'src_height': args.camera_height,
            'src_width': args.camera_width,
            'intra_period': intra_period,
            'frame_num': frame_num,
            'calc_ssim': args.calc_ssim,
            'dataset_path': '',
            'write_stream': args.write_stream,
            'check_existing': args.check_existing,
            'stream_path': args.stream_path,
            'save_decoded_frame': args.save_decoded_frame,
            'ds_name': args.camera_ds_name,
            'verbose': args.verbose,
            'verbose_json': args.verbose_json,
            'precision': args.precision,
            'min_warmup': args.min_warmup,
            'rt_mode': args.rt_mode,
            'stage': args.stage,
            'enc_log_suffix': args.enc_log_suffix,
            'dec_log_suffix': args.dec_log_suffix,
            'decode_bin_override': args.decode_bin_override,
            'profile_i_path': args.profile_i_path,
            'profile_p_path': args.profile_p_path,
            'strict_frame_sync': args.strict_frame_sync,
            'max_nonframe_nals': args.max_nonframe_nals,
            'optimize_yuv420_tensorize': args.optimize_yuv420_tensorize,
            'yuv420_use_pinned_memory': args.yuv420_use_pinned_memory,
            'yuv420_async_h2d': args.yuv420_async_h2d,
            'input_mode': 'camera',
            'camera_device': args.camera_device,
            'camera_format': normalize_camera_format(args.camera_format),
            'camera_fps': args.camera_fps,
            'camera_timeout_sec': args.camera_timeout_sec,
            'camera_v4l2_io_mode': args.camera_v4l2_io_mode,
            'camera_dump_reference': args.camera_dump_reference,
            'camera_reference_yuv': args.camera_reference_yuv,
            'camera_control_mode': args.camera_control_mode,
            'camera_buffer_mode': args.camera_buffer_mode,
            'camera_buffer_max_frames': args.camera_buffer_max_frames,
            'camera_drain_print_interval': args.camera_drain_print_interval,
            'save_decoded_yuv': args.save_decoded_yuv,
            'save_decoded_video': args.save_decoded_video,
            'decoded_video_ext': args.decoded_video_ext,
            'decoded_video_codec': args.decoded_video_codec,
            'decoded_video_fps': args.decoded_video_fps,
        }
        tasks.append(cur_args)
    return tasks


def build_file_tasks(args: argparse.Namespace, config, root_path, qp_i: List[int], qp_p: List[int]) -> List[Dict]:
    all_tasks = []
    for ds_name in config:
        if config[ds_name].get('test', 0) == 0:
            continue
        for seq in config[ds_name]['sequences']:
            for rate_idx in range(args.rate_num):
                cur_args = {}
                cur_args['rate_idx'] = rate_idx
                cur_args['qp_i'] = qp_i[rate_idx]
                if not args.force_intra:
                    cur_args['qp_p'] = qp_p[rate_idx]
                else:
                    cur_args['qp_p'] = qp_i[rate_idx]
                cur_args['force_intra'] = args.force_intra
                cur_args['reset_interval'] = args.reset_interval
                cur_args['seq'] = seq
                cur_args['src_type'] = config[ds_name]['src_type']
                cur_args['src_height'] = config[ds_name]['sequences'][seq]['height']
                cur_args['src_width'] = config[ds_name]['sequences'][seq]['width']
                cur_args['intra_period'] = config[ds_name]['sequences'][seq]['intra_period']
                if args.force_intra:
                    cur_args['intra_period'] = 1
                if args.force_intra_period > 0:
                    cur_args['intra_period'] = args.force_intra_period
                cur_args['frame_num'] = config[ds_name]['sequences'][seq]['frames']
                if args.force_frame_num > 0:
                    cur_args['frame_num'] = args.force_frame_num
                cur_args['calc_ssim'] = args.calc_ssim
                cur_args['dataset_path'] = os.path.join(root_path, config[ds_name]['base_path'])
                cur_args['write_stream'] = args.write_stream
                cur_args['check_existing'] = args.check_existing
                cur_args['stream_path'] = args.stream_path
                cur_args['save_decoded_frame'] = args.save_decoded_frame
                cur_args['ds_name'] = ds_name
                cur_args['verbose'] = args.verbose
                cur_args['verbose_json'] = args.verbose_json
                cur_args['precision'] = args.precision
                cur_args['min_warmup'] = args.min_warmup
                cur_args['rt_mode'] = args.rt_mode
                cur_args['stage'] = args.stage
                cur_args['enc_log_suffix'] = args.enc_log_suffix
                cur_args['dec_log_suffix'] = args.dec_log_suffix
                cur_args['decode_bin_override'] = args.decode_bin_override
                cur_args['profile_i_path'] = args.profile_i_path
                cur_args['profile_p_path'] = args.profile_p_path
                cur_args['strict_frame_sync'] = args.strict_frame_sync
                cur_args['max_nonframe_nals'] = args.max_nonframe_nals
                cur_args['optimize_yuv420_tensorize'] = args.optimize_yuv420_tensorize
                cur_args['yuv420_use_pinned_memory'] = args.yuv420_use_pinned_memory
                cur_args['yuv420_async_h2d'] = args.yuv420_async_h2d
                cur_args['input_mode'] = 'file'
                cur_args['save_decoded_yuv'] = args.save_decoded_yuv
                cur_args['save_decoded_video'] = args.save_decoded_video
                cur_args['decoded_video_ext'] = args.decoded_video_ext
                cur_args['decoded_video_codec'] = args.decoded_video_codec
                cur_args['decoded_video_fps'] = args.decoded_video_fps
                all_tasks.append(cur_args)
    return all_tasks


def main():
    begin_time = time.time()
    args = parse_args()
    global ENABLE_PRIOR4X_WORKSPACE_OPT
    ENABLE_PRIOR4X_WORKSPACE_OPT = bool(args.optimize_prior4x_workspace)
    if args.force_zero_thres is not None and args.force_zero_thres < 0:
        args.force_zero_thres = None
    if args.cuda_idx is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(s) for s in args.cuda_idx])

    init_models(args)

    rate_num = args.rate_num
    if args.qp_i is not None:
        assert len(args.qp_i) == rate_num
        qp_i = args.qp_i
    else:
        assert 1 <= rate_num <= DMC.get_qp_num()
        if rate_num == 1:
            qp_i = [DMC.get_qp_num() // 2]
        else:
            qp_i = [int(i + 0.5) for i in np.linspace(0, DMC.get_qp_num() - 1, num=rate_num)]
    if not args.force_intra:
        if args.qp_p is not None:
            assert len(args.qp_p) == rate_num
            qp_p = args.qp_p
        else:
            qp_p = qp_i
    else:
        qp_p = qp_i

    if args.input_mode == 'camera' and args.camera_control_mode == 'manual_session':
        run_camera_manual_session_app(args)
        total_minutes = (time.time() - begin_time) / 60
        print('测试完成')
        print(f'总耗时: {total_minutes:.1f} 分钟')
        return

    print(f"input_mode={args.input_mode}, stage={args.stage}, rate_num={rate_num}, qp_i={qp_i}")
    print("时间统计模式: standard_sync_per_frame (fixed)")
    print(f"P路径细分profiling: {args.profile_p_path}")
    print(f"P-prior2x workspace/cat reuse : {args.optimize_p_prior2x_workspace}")
    print(f"P-resprior workspace/cat reuse : {args.optimize_p_resprior_workspace}")
    print(f"P-encoder-cat workspace/cat reuse : {args.optimize_p_encoder_cat_workspace}")
    print(f"YUV420 tensorize : {args.optimize_yuv420_tensorize} (pinned={args.yuv420_use_pinned_memory}, async_h2d={args.yuv420_async_h2d})")
    print(f"prior_4x workspace/cache/in-place optimization: {'ON' if ENABLE_PRIOR4X_WORKSPACE_OPT else 'OFF'}")
    if bool(args.profile_i_path) or bool(args.profile_p_path):
        print("[WARN] profile_i_path/profile_p_path=1 会显著增加同步与计时开销，不能作为纯吞吐基线。")

    if args.input_mode == 'camera':
        if args.stage in ('both', 'decode') and args.rt_mode == 0 and not args.camera_dump_reference and args.camera_reference_yuv is None:
            raise ValueError('camera rt_mode=0 with decode/metrics requires reference I420: enable --camera_dump_reference or pass --camera_reference_yuv.')
        all_tasks = build_camera_tasks(args, qp_i, qp_p)
        config = {args.camera_ds_name: {'test': 1, 'sequences': {all_tasks[0]['seq']: {}}}}
    else:
        with open(args.test_config) as f:
            cfg = json.load(f)
        root_path = args.force_root_path if args.force_root_path is not None else cfg['root_path']
        config = cfg['test_classes']
        all_tasks = build_file_tasks(args, config, root_path, qp_i, qp_p)

    results = []
    for task_args in tqdm(all_tasks, desc='处理序列'):
        res = worker_wrapper(task_args)
        results.append(res)

    log_result = {}
    if args.input_mode == 'camera':
        log_result[args.camera_ds_name] = {all_tasks[0]['seq']: {}}
    else:
        for ds_name in config:
            if config[ds_name].get('test', 0) == 0:
                continue
            log_result[ds_name] = {}
            for seq in config[ds_name]['sequences']:
                log_result[ds_name][seq] = {}
    for res in results:
        log_result.setdefault(res['ds_name'], {}).setdefault(res['seq'], {})[f"{res['rate_idx']:03d}"] = res

    out_json_dir = os.path.dirname(args.output_path)
    if len(out_json_dir) > 0:
        create_folder(out_json_dir, True)
    with open(args.output_path, 'w') as fp:
        dump_json(log_result, fp, float_digits=6, indent=2)
    total_minutes = (time.time() - begin_time) / 60
    print('测试完成')
    print(f'总耗时: {total_minutes:.1f} 分钟')


if __name__ == '__main__':
    main()

