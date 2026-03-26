import atexit
import os
import socket
import subprocess
import time
from pathlib import Path

import streamlit as st

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import openai

SERVER_PROCESS = None
OPENAI_CLIENT = None
PORT = None
SERVER_CONFIG = None
CONTEXT_WINDOW = 4096
SERVER_START_TIMEOUT_SECONDS = 40
DEFAULT_MODEL_ALIAS = "qwen-2.5-0.5b"


class _LegacyChatCompletions:
    def create(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)


class _LegacyChat:
    def __init__(self):
        self.completions = _LegacyChatCompletions()


class _LegacyOpenAIClient:
    def __init__(self):
        self.chat = _LegacyChat()


def _build_openai_compatible_client(base_url, api_key, timeout_seconds):
    if OpenAI is not None:
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    # Compatibility path for older openai package versions.
    openai.api_base = base_url
    openai.api_key = api_key
    if hasattr(openai, "timeout"):
        openai.timeout = timeout_seconds
    return _LegacyOpenAIClient()


def _extract_chunk_fields(chunk):
    """
    Normalize streamed chunk payloads for both openai>=1 and openai<1 clients.
    """
    if chunk is None:
        return None, None

    choices = None
    if hasattr(chunk, "choices"):
        choices = getattr(chunk, "choices", None)
    elif isinstance(chunk, dict):
        choices = chunk.get("choices")

    if not choices:
        return None, None

    first_choice = choices[0]
    delta = None
    if hasattr(first_choice, "delta"):
        delta = getattr(first_choice, "delta", None)
    elif isinstance(first_choice, dict):
        delta = first_choice.get("delta")

    if delta is None:
        return None, None

    if isinstance(delta, dict):
        reasoning_text = delta.get("reasoning_content")
        text = delta.get("content")
    else:
        reasoning_text = getattr(delta, "reasoning_content", None)
        text = getattr(delta, "content", None)

    return reasoning_text, text


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def wait_for_server(port, timeout=SERVER_START_TIMEOUT_SECONDS):
    start_time = time.time()
    while (time.time() - start_time) < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def _resolve_runtime_paths():
    base_dir = Path(__file__).resolve().parent
    executable_path = base_dir / "llama-bin" / "llama-server.exe"
    # Reverted back to the original model since the Hugging Face reasoning model is corrupted
    model_path = base_dir / "qwen2.5-0.5b-instruct-q8_0.gguf"
    return executable_path, model_path


def _cpu_threads():
    cpu_count = os.cpu_count() or 2
    return max(1, cpu_count // 2)


def _is_server_alive():
    if SERVER_PROCESS is None or PORT is None:
        return False
    if SERVER_PROCESS.poll() is not None:
        return False
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=1):
            return True
    except OSError:
        return False


def _stop_server():
    global SERVER_PROCESS
    if SERVER_PROCESS is None:
        return
    if SERVER_PROCESS.poll() is None:
        SERVER_PROCESS.terminate()
        try:
            SERVER_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            SERVER_PROCESS.kill()
    SERVER_PROCESS = None


atexit.register(_stop_server)


def _sanitize_messages(messages, max_messages):
    if not isinstance(messages, list):
        raise ValueError("Messages must be provided as a list.")

    valid_roles = {"system", "user", "assistant"}
    cleaned_messages = []

    for message in messages[-max_messages:]:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in valid_roles or content is None:
            continue
        text = str(content).strip()
        if not text:
            continue
        cleaned_messages.append({"role": role, "content": text})

    if not cleaned_messages:
        raise ValueError("No valid messages available for inference.")

    return cleaned_messages


def load_model_and_tokenizer(
    model_id: str,
    reasoning_mode: str = "off",
    reasoning_format: str = "deepseek-legacy",
    reasoning_budget: int = 256,
):
    """
    Start llama-server.exe if needed and return an OpenAI-compatible client.
    """
    del model_id
    global SERVER_PROCESS, OPENAI_CLIENT, PORT, SERVER_CONFIG

    executable_path, model_path = _resolve_runtime_paths()
    if not executable_path.exists():
        raise FileNotFoundError(f"Could not find {executable_path}.")
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find {model_path}.")

    normalized_mode = str(reasoning_mode).lower().strip()
    if normalized_mode not in {"on", "off", "auto"}:
        normalized_mode = "off"

    current_config = {
        "model_path": str(model_path),
        "reasoning_mode": normalized_mode,
        "reasoning_format": str(reasoning_format).strip(),
        "reasoning_budget": int(reasoning_budget),
    }

    if (
        OPENAI_CLIENT is not None
        and _is_server_alive()
        and SERVER_CONFIG == current_config
    ):
        return OPENAI_CLIENT, SERVER_PROCESS, "Your privacy, Our concern"

    _stop_server()
    PORT = get_free_port()

    cmd = [
        str(executable_path),
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(PORT),
        "-c",
        str(CONTEXT_WINDOW),
        "-t",
        str(_cpu_threads()),
    ]
    cmd.extend(["--reasoning", normalized_mode])
    if normalized_mode != "off":
        cmd.extend(["--reasoning-format", current_config["reasoning_format"]])
        cmd.extend(["--reasoning-budget", str(current_config["reasoning_budget"])])

    startupinfo = None
    creationflags = 0
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        creationflags = subprocess.CREATE_NO_WINDOW

    SERVER_PROCESS = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=creationflags,
    )

    if not wait_for_server(PORT):
        _stop_server()
        raise RuntimeError("llama-server failed to start within timeout.")

    OPENAI_CLIENT = _build_openai_compatible_client(
        base_url=f"http://127.0.0.1:{PORT}/v1",
        api_key="llama.cpp",
        timeout_seconds=120.0,
    )
    SERVER_CONFIG = current_config

    return OPENAI_CLIENT, SERVER_PROCESS, "Your privacy, Our concern"


def generate_response_stream(
    client,
    model_alias,
    messages,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    max_history_messages=12,
):
    """
    Stream AI response chunks as a generator consumable by Streamlit.
    """
    if client is None:
        raise RuntimeError("OpenAI client is not initialized.")

    prepared_messages = _sanitize_messages(messages, max_history_messages)
    model_name = model_alias or DEFAULT_MODEL_ALIAS

    response = client.chat.completions.create(
        model=model_name,
        messages=prepared_messages,
        max_tokens=max(1, int(max_new_tokens)),
        temperature=max(0.0, float(temperature)),
        top_p=min(1.0, max(0.0, float(top_p))),
        stream=True,
    )

    def stream_generator():
        emitted_content = False
        in_reasoning = False
        for chunk in response:
            reasoning_text, text = _extract_chunk_fields(chunk)
            if reasoning_text:
                if not in_reasoning:
                    yield "<think>\n"
                    in_reasoning = True
                emitted_content = True
                yield reasoning_text
                continue

            if text:
                if in_reasoning:
                    yield "\n</think>\n\n"
                    in_reasoning = False
                emitted_content = True
                yield text

        if in_reasoning:
            yield "\n</think>\n"

        if not emitted_content:
            yield "No response was generated. Please try again."

    return stream_generator()
