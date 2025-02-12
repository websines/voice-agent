"""
GLaDOS Telephony Module - Handles SIP-based voice calls using Telnyx.

This module provides integration with Telnyx for making and receiving phone calls,
allowing GLaDOS to interact with callers using its TTS and ASR capabilities.
"""

from .telnyx_client import TelnyxClient
from .call_manager import CallManager
from .audio_bridge import AudioBridge

__all__ = ["TelnyxClient", "CallManager", "AudioBridge"] 