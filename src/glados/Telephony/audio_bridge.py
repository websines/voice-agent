"""
AudioBridge handles real-time audio streaming between Telnyx and GLaDOS components.
"""

import asyncio
from typing import Optional, Dict, Any
from loguru import logger
import sounddevice as sd
import numpy as np
from queue import Queue

class AudioBridge:
    """
    Handles real-time audio streaming between Telnyx and GLaDOS components.
    
    This class manages the bidirectional audio stream between the Telnyx call
    and GLaDOS's TTS/ASR components, handling audio format conversion and buffering.
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the AudioBridge.
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.is_running = False
        self.tts_paused = False
        self._stream: Optional[sd.Stream] = None
        
    async def start(self) -> None:
        """Start the audio bridge and initialize audio streams."""
        if self.is_running:
            return
            
        def audio_callback(indata: np.ndarray, outdata: np.ndarray, 
                          frames: int, time: Any, status: Any) -> None:
            """Callback for handling audio stream data."""
            if status:
                logger.warning(f"Audio callback status: {status}")
                
            # Handle incoming audio (from call to ASR)
            if not self.input_queue.full():
                self.input_queue.put(indata.copy())
                
            # Handle outgoing audio (from TTS to call)
            try:
                if not self.tts_paused and not self.output_queue.empty():
                    outdata[:] = self.output_queue.get_nowait()
                else:
                    outdata.fill(0)
            except Exception as e:
                logger.error(f"Error in audio output handling: {e}")
                outdata.fill(0)
        
        try:
            self._stream = sd.Stream(
                channels=self.channels,
                samplerate=self.sample_rate,
                callback=audio_callback,
                dtype=np.float32
            )
            self._stream.start()
            self.is_running = True
            logger.info("Audio bridge started successfully")
        except Exception as e:
            logger.error(f"Failed to start audio bridge: {e}")
            raise
            
    async def stop(self) -> None:
        """Stop the audio bridge and clean up resources."""
        if not self.is_running:
            return
            
        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
            self.is_running = False
            # Clear queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
            logger.info("Audio bridge stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping audio bridge: {e}")
            raise
            
    async def send_audio(self, audio_data: np.ndarray) -> None:
        """
        Send audio data to the call (TTS output).
        
        Args:
            audio_data: Numpy array containing audio samples
        """
        if not self.is_running:
            return
            
        try:
            # Ensure audio data matches expected format
            if audio_data.shape[1] != self.channels:
                audio_data = self._convert_channels(audio_data)
            
            # Split into chunks that match the stream buffer size
            chunk_size = int(self.sample_rate * 0.02)  # 20ms chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    chunk = np.pad(chunk, ((0, chunk_size - len(chunk)), (0, 0)))
                self.output_queue.put(chunk)
        except Exception as e:
            logger.error(f"Error sending audio data: {e}")
            
    async def get_audio(self) -> Optional[np.ndarray]:
        """
        Get received audio data from the call (for ASR).
        
        Returns:
            Optional[np.ndarray]: Audio data if available, None otherwise
        """
        if not self.is_running or self.input_queue.empty():
            return None
            
        try:
            return self.input_queue.get_nowait()
        except Exception as e:
            logger.error(f"Error getting audio data: {e}")
            return None
            
    def _convert_channels(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio data to the required number of channels."""
        if audio_data.shape[1] == self.channels:
            return audio_data
        elif audio_data.shape[1] == 1 and self.channels == 2:
            return np.repeat(audio_data, 2, axis=1)
        elif audio_data.shape[1] == 2 and self.channels == 1:
            return np.mean(audio_data, axis=1, keepdims=True)
        else:
            raise ValueError(f"Unsupported channel conversion from {audio_data.shape[1]} to {self.channels}")
            
    async def pause_tts(self) -> None:
        """Pause TTS output (when caller is speaking)."""
        self.tts_paused = True
        
    async def resume_tts(self) -> None:
        """Resume TTS output (when caller stops speaking)."""
        self.tts_paused = False 