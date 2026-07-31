import os
import subprocess
import tempfile
import time

import requests

from mc_translator.config import ConfigManager
from mc_translator.constants import KOBOLD_MODELS_URL
from mc_translator.utils.hardware_detect import detect_vulkan_discrete_device_index


class AiLauncher:
    def __init__(self, config: ConfigManager) -> None:
        self.config = config
        self.process: subprocess.Popen | None = None

    def is_alive(self) -> bool:
        try:
            return requests.get(KOBOLD_MODELS_URL, timeout=1).status_code == 200
        except requests.RequestException:
            return False

    def ensure_running(self, should_continue, on_status, on_log) -> bool:
        if self.is_alive():
            on_log("✅ ИИ уже работает", "green")
            return True

        exe = self.config.get("AI", "exe_path")
        model = self.config.get("AI", "model_path")
        gpu = self.config.get("AI", "gpu_layers")
        if not gpu.isdigit():
            gpu = "99"
        backend = self.config.get("AI", "gpu_backend") or "cuda"

        on_log("🤖 Запуск ИИ...", "cyan")
        argv = [
            exe,
            model,
            "--port",
            "5001",
            "--quiet",
            "--contextsize",
            "8192",
            "--gpulayers",
            gpu,
            # Cheap, low-risk: nudges CPU scheduling priority up for the
            # layers that run off-GPU, so other apps on the same machine
            # compete with it less.
            "--highpriority",
        ]
        # Backend-specific flags -- gated so a koboldcpp-nocuda.exe build
        # (downloaded by the "auto-setup local AI" flow for non-NVIDIA
        # hardware, see runtime/ai_catalog.py) is never launched with
        # --usecublas, which it has no CUDA support for. "cuda" stays the
        # default for every install that predates gpu_backend existing, so
        # this doesn't change behavior for anyone already running --usecublas.
        if backend == "cuda":
            argv += [
                "--usecublas",
                # Quantizing the KV cache frees a meaningful chunk of
                # VRAM/RAM on tight hardware (q8_0 is near-lossless) without
                # touching --contextsize, so it doesn't risk truncating a
                # legitimately long "context mode" prompt. cuBLAS-specific.
                "--quantkv",
                "q8_0",
            ]
        elif backend == "vulkan":
            # A bare "--usevulkan" (no index) is NOT "pick the strongest
            # GPU" -- it's "let raw Vulkan device enumeration order decide",
            # which real reports show can land on an integrated GPU instead
            # of a discrete one on hybrid systems (see
            # hardware_detect.detect_vulkan_discrete_device_index's
            # docstring). Pin the index explicitly when it can be resolved;
            # fall back to the blank flag (today's behavior) otherwise.
            device_index = detect_vulkan_discrete_device_index()
            if device_index is not None:
                argv += ["--usevulkan", str(device_index)]
            else:
                argv.append("--usevulkan")
            # Flash attention is default-on in modern koboldcpp, but on most
            # non-NVIDIA Vulkan GPUs (no coopmat2 support) it silently falls
            # back to CPU compute instead of erroring -- real reports of
            # ~75% token-gen throughput loss on AMD
            # (github.com/ggml-org/llama.cpp/discussions/12629). That's
            # exactly the "launch succeeds, performance silently collapses"
            # failure mode this project has already been burned by once
            # (see the 14B/6-layers incident in ai_catalog.py's module
            # docstring) -- --noflashattention avoids it.
            argv.append("--noflashattention")
            # Deliberately NOT adding --quantkv here even though it's
            # mechanically backend-agnostic in koboldcpp (unlike the cuda
            # branch above): there's an unresolved, driver-blamed koboldcpp
            # report of q8_0 KV-cache corruption specifically on AMD
            # (github.com/LostRuins/koboldcpp/issues/1459, RX 6900XT) with
            # no fix as of this writing, and this path can't be verified on
            # real AMD hardware -- the VRAM saved isn't worth that risk.
        # backend == "cpu": no GPU flag at all.

        # stderr goes to a temp file (not subprocess.PIPE) so a chatty child
        # can't fill the OS pipe buffer and deadlock while we're not
        # actively draining it -- we only read this file after detecting the
        # process has exited, purely for a diagnostic message.
        stderr_fd, stderr_path = tempfile.mkstemp(prefix="koboldcpp_stderr_", suffix=".log")
        try:
            with os.fdopen(stderr_fd, "wb") as stderr_file:
                self.process = subprocess.Popen(
                    argv,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                )
        except OSError as exc:
            on_log(f"❌ Ошибка запуска ИИ: {exc}", "red")
            self._cleanup_stderr_log(stderr_path)
            return False

        for i in range(180):
            if not should_continue():
                return False
            # A dead child (bad/missing model path, port already in use,
            # --usecublas without CUDA, etc.) used to be indistinguishable
            # from one still warming up until the full 180-iteration budget
            # elapsed -- check poll() so a fast failure is reported fast.
            exit_code = self.process.poll()
            if exit_code is not None:
                detail = self._read_stderr_tail(stderr_path)
                on_log(f"❌ Процесс ИИ завершился сразу (код {exit_code}){detail}", "red")
                return False
            on_status(f"Прогрев нейросети... ({i}/180 сек)")
            if self.is_alive():
                on_log("✅ ИИ успешно запущен!\n", "green")
                self._cleanup_stderr_log(stderr_path)
                return True
            time.sleep(1)

        on_log("❌ Сервер ИИ не отвечает.", "red")
        return False

    @staticmethod
    def _read_stderr_tail(path: str, max_chars: int = 500) -> str:
        result = ""
        try:
            with open(path, "rb") as f:
                content = f.read().decode("utf-8", errors="replace").strip()
            if content:
                result = f" — {content[-max_chars:]}"
        except OSError:
            pass
        AiLauncher._cleanup_stderr_log(path)
        return result

    @staticmethod
    def _cleanup_stderr_log(path: str) -> None:
        try:
            os.remove(path)
        except OSError:
            pass

    def terminate(self, on_log=None) -> None:
        if not self.process:
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.process = None
        # Verify at the HTTP level (not just the Popen handle) -- Windows can
        # detach a GPU-accelerated child in ways that leave the tracked
        # handle pointing at nothing useful, silently no-opping terminate().
        # Left unnoticed, the stale server keeps holding VRAM and the next
        # run's ensure_running() would happily reuse it (is_alive() sees it
        # as healthy) instead of the fresh process the user just asked for.
        if self.is_alive() and on_log is not None:
            on_log(
                "⚠ Не удалось остановить процесс ИИ (koboldcpp.exe) — он "
                "продолжает работать и занимает видеопамять. При необходимости "
                "закройте его вручную через диспетчер задач.",
                "yellow",
            )
