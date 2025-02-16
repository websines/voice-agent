import argparse
import hashlib
from pathlib import Path
import asyncio
import sys

import requests
import sounddevice as sd  # type: ignore
import yaml

from .engine import Glados, GladosConfig
from .TTS import tts_glados
from .utils import spoken_text_converter as stc
from .Telephony.telnyx_client import TelnyxClient, TelnyxConfig
from .Telephony.call_manager import CallManager
from .Telephony.webhook_server import initialize_webhook_server, start_webhook_server

DEFAULT_CONFIG = Path("configs/glados_config.yaml")

MODEL_CHECKSUMS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "313705ff6f897696ddbe0d92b5ffadad7429a47d2ddeef370e6f59248b1e8fb5",
    "models/ASR/silero_vad_v5.onnx": "6b99cbfd39246b6706f98ec13c7c50c6b299181f2474fa05cbc8046acc274396",
    "models/TTS/glados.onnx": "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095",
    "models/TTS/kokoro-v1.0.fp16.onnx": "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    "models/TTS/kokoro-voices-v1.0.bin": "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88",
    "models/TTS/phomenizer_en.onnx": "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0",
}

MODEL_URLS = {
    "models/ASR/nemo-parakeet_tdt_ctc_110m.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/nemo-parakeet_tdt_ctc_110m.onnx",
    "models/ASR/silero_vad_v5.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/silero_vad_v5.onnx",
    "models/TTS/glados.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx",
    "models/TTS/kokoro-v1.0.fp16.onnx": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx",
    "models/TTS/kokoro-voices-v1.0.bin": "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin",
    "models/TTS/phomenizer_en.onnx": "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx",
}


assert MODEL_CHECKSUMS.keys() == MODEL_URLS.keys()


def verify_checksums() -> dict[str, bool]:
    """
    Verify the integrity of model files by comparing their SHA-256 checksums against expected values.

    This function checks each model file specified in MODEL_CHECKSUMS to ensure it exists
    and has the correct checksum. Files that are missing or have incorrect checksums are
    marked as invalid.

    Parameters:
        None

    Returns:
        dict[str, bool]: A dictionary where keys are model file paths and values are
                         boolean indicators of checksum validity (True if valid, False if
                         missing or checksum mismatch)

    Example:
        >>> verify_checksums()
        {
            'models/tts_model.pth': True,
            'models/asr_model.bin': False
        }
    """
    results = {}
    for path, expected in MODEL_CHECKSUMS.items():
        model_path = Path(path)
        if not model_path.exists():
            results[path] = False
            continue

        sha = hashlib.sha256()
        sha.update(model_path.read_bytes())
        results[path] = sha.hexdigest() == expected

    return results


def download_models() -> None:
    """Download model files required for GLaDOS with robust error handling and progress tracking.

    This function downloads the required model files from predefined URLs if they are missing
    or have invalid checksums. It includes several reliability features:

    Features:
        - Progress tracking with percentage display
        - Temporary file usage to prevent partial downloads
        - SHA256 checksum verification during download
        - Retry logic with exponential backoff (max 3 retries)
        - Network timeout handling (30 seconds)
        - Automatic cleanup of temporary files on failure
        - Creation of parent directories if they don't exist

    The function downloads the following models:
        - ASR model: nemo-parakeet_tdt_ctc_110m.onnx
        - VAD model: silero_vad.onnx
        - TTS model: glados.onnx
        - Phonemizer model: phomenizer_en.onnx

    Raises:
        requests.RequestException: If network errors occur during download
        IOError: If file system operations fail
        ValueError: If downloaded file has incorrect checksum
        SystemExit: If any model download fails after max retries

    Example:
        >>> from glados.cli import download_models
        >>> download_models()  # Downloads all required models with progress tracking
    """
    import sys
    from time import sleep
    from typing import BinaryIO

    def calculate_sha256(file: BinaryIO) -> str:
        """Calculate SHA256 hash of a file-like object."""
        sha = hashlib.sha256()
        file.seek(0)
        for chunk in iter(lambda: file.read(8192), b""):
            sha.update(chunk)
        file.seek(0)
        return sha.hexdigest()

    def download_with_progress(url: str, path: Path, expected_hash: str, max_retries: int = 3) -> None:
        """Download a file with progress tracking and retry logic."""
        retry_count = 0
        temp_path = path.with_suffix(path.suffix + ".tmp")

        while retry_count < max_retries:
            try:
                # Create parent directories if they don't exist
                path.parent.mkdir(parents=True, exist_ok=True)

                # Start download with stream enabled
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                # Open temporary file for writing
                with open(temp_path, "wb") as f:
                    if total_size == 0:
                        print(f"\nDownloading {path.name} (size unknown)")
                    else:
                        print(f"\nDownloading {path.name} ({total_size / 1024 / 1024:.1f} MB)")

                    downloaded_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                sys.stdout.write(f"\rProgress: {progress:.1f}%")
                                sys.stdout.flush()

                # Verify checksum
                with open(temp_path, "rb") as f:
                    if calculate_sha256(f) != expected_hash:
                        raise ValueError("Downloaded file has incorrect checksum")

                # Rename temporary file to final path
                if path.exists():
                    path.unlink()
                temp_path.rename(path)
                print(f"\nSuccessfully downloaded {path.name}")
                return

            except (OSError, requests.RequestException, ValueError) as e:
                retry_count += 1
                if temp_path.exists():
                    temp_path.unlink()

                if retry_count < max_retries:
                    wait_time = 2**retry_count  # Exponential backoff
                    print(f"\nError downloading {path.name}: {e!s}")
                    print(f"Retrying in {wait_time} seconds... (Attempt {retry_count + 1}/{max_retries})")
                    sleep(wait_time)
                else:
                    print(f"\nFailed to download {path.name} after {max_retries} attempts: {e!s}")
                    raise

    # Download each model file
    checksums = verify_checksums()
    for path, is_valid in checksums.items():
        if not is_valid:
            try:
                download_with_progress(MODEL_URLS[path], Path(path), MODEL_CHECKSUMS[path])
            except Exception as e:
                print(f"Error: Failed to download {path}: {e!s}")
                sys.exit(1)


def say(text: str, config_path: str | Path = "configs/glados_config.yaml") -> None:
    """
    Converts text to speech using the GLaDOS text-to-speech system and plays the generated audio.

    Parameters:
        text (str): The text to be spoken by the GLaDOS voice assistant.
        config_path (str | Path, optional): Path to the configuration YAML file.
            Defaults to "glados_config.yaml".

    Notes:
        - Uses a text-to-speech synthesizer to generate audio
        - Converts input text to a spoken format before synthesis
        - Plays the generated audio using the system's default sound device
        - Blocks execution until audio playback is complete

    Example:
        say("Hello, world!")  # Speaks the text using GLaDOS voice
    """
    glados_tts = tts_glados.Synthesizer()
    converter = stc.SpokenTextConverter()
    converted_text = converter.text_to_spoken(text)
    # Generate the audio to from the text
    audio = glados_tts.generate_speech_audio(converted_text)

    # Play the audio
    sd.play(audio, glados_tts.sample_rate)
    sd.wait()


def start(config_path: str | Path = "configs/glados_config.yaml") -> None:
    """Start GLaDOS with the specified configuration."""
    if not models_valid():
        print("Required models are missing or invalid. Please run 'glados download' first.")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    glados = Glados.from_config(GladosConfig(**config["Glados"]))
    
    # Initialize Telnyx client if config exists
    telnyx_client = None
    try:
        with open("configs/telnyx_config.yaml") as f:
            telnyx_config = yaml.safe_load(f)
            telnyx_client = TelnyxClient(TelnyxConfig(**telnyx_config))
            # Initialize webhook server with the client
            initialize_webhook_server(telnyx_client)
            # Start webhook server in a separate thread
            import threading
            webhook_thread = threading.Thread(
                target=start_webhook_server,
                kwargs={"host": "0.0.0.0", "port": 8000},
                daemon=True
            )
            webhook_thread.start()
            print("Webhook server started on http://0.0.0.0:8000")

    except FileNotFoundError:
        print("No Telnyx configuration found, running without telephony support")
    except Exception as e:
        print(f"Error initializing Telnyx client: {str(e)}")

    glados.start_listen_event_loop()


def models_valid() -> bool:
    """
    Check the validity of all model files for the GLaDOS voice assistant.

    Verifies the integrity of model files by computing their checksums and comparing them against expected values.

    Returns:
        bool: True if all model files are valid and present, False otherwise. When False, prints a message
              instructing the user to download the required model files.
    """
    results = verify_checksums()
    if not all(results.values()):
        print("Some model files are missing or invalid. Please run 'uv run glados download'")
        return False
    return True


async def make_call(phone_number: str, config_path: str | Path = "configs/telnyx_config.yaml") -> None:
    """
    Make an outbound call using the Telnyx integration.
    
    Args:
        phone_number: The phone number to call in E.164 format
        config_path: Path to the Telnyx configuration file
    """
    try:
        # Load Telnyx configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        telnyx_config = TelnyxConfig(**config)
        
        # Initialize Telnyx client and call manager
        client = TelnyxClient(telnyx_config)
        manager = CallManager(client)
        
        # Start the call
        call_id = await manager.start_call(phone_number)
        if call_id:
            print(f"Call initiated with ID: {call_id}")
            # Keep the process alive to handle call events
            while True:
                await asyncio.sleep(1)
        else:
            print("Failed to initiate call")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nCall terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error making call: {e}")
        sys.exit(1)


def main() -> None:
    """Main entry point for the GLaDOS CLI."""
    parser = argparse.ArgumentParser(description="GLaDOS CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser("download", help="Download required models")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start GLaDOS")
    start_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to configuration file",
    )

    # Say command
    say_parser = subparsers.add_parser("say", help="Make GLaDOS say something")
    say_parser.add_argument("text", type=str, help="Text to say")
    say_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to configuration file",
    )

    # Call command
    call_parser = subparsers.add_parser("call", help="Make a phone call")
    call_parser.add_argument("phone_number", type=str, help="Phone number to call")
    call_parser.add_argument(
        "--config",
        type=str,
        default="configs/telnyx_config.yaml",
        help="Path to Telnyx configuration file",
    )

    # Webhook command
    webhook_parser = subparsers.add_parser("webhook", help="Start the webhook server")
    webhook_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the webhook server to",
    )
    webhook_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the webhook server on",
    )
    webhook_parser.add_argument(
        "--config",
        type=str,
        default="configs/telnyx_config.yaml",
        help="Path to Telnyx configuration file",
    )

    args = parser.parse_args()

    if args.command == "download":
        download_models()
    elif args.command == "start":
        start(args.config)
    elif args.command == "say":
        say(args.text, args.config)
    elif args.command == "call":
        asyncio.run(make_call(args.phone_number, args.config))
    elif args.command == "webhook":
        try:
            with open(args.config) as f:
                telnyx_config = yaml.safe_load(f)
                telnyx_client = TelnyxClient(TelnyxConfig(**telnyx_config))
                initialize_webhook_server(telnyx_client)
                print(f"Starting webhook server on http://{args.host}:{args.port}")
                start_webhook_server(host=args.host, port=args.port)
        except Exception as e:
            print(f"Error starting webhook server: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
