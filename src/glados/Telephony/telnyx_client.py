"""
TelnyxClient handles all interactions with the Telnyx API for voice calls.
"""

from typing import Optional, Callable, Dict, Any
import asyncio
import base64
import json
from loguru import logger
import telnyx
import websockets
from pydantic import BaseModel, Field
import numpy as np


class StreamConfig(BaseModel):
    """Configuration for media streaming."""
    stream_url: str = Field(..., description="WebSocket URL for media streaming")
    stream_track: str = Field(default="both_tracks", description="Track identifier for media streaming")
    stream_bidirectional: bool = Field(default=True, description="Enable bidirectional streaming")
    codec: str = Field(default="PCMU", description="Audio codec (PCMU or OPUS)")
    sampling_rate: int = Field(default=8000, description="Audio sampling rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")


class TelnyxConfig(BaseModel):
    """Configuration for Telnyx client."""
    api_key: str
    sip_connection_id: str
    from_number: str
    stream_config: StreamConfig


class TelnyxClient:
    """
    Handles Telnyx API interactions for voice calls with media streaming.
    """
    
    def __init__(self, config: TelnyxConfig):
        """Initialize the Telnyx client."""
        self.config = config
        telnyx.api_key = config.api_key
        self._active_calls: Dict[str, Any] = {}
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._audio_queues: Dict[str, asyncio.Queue] = {}
        self._call_handlers: Dict[str, Callable] = {}  # Add handler registry
        
    def register_call_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for call events."""
        self._call_handlers[event_type] = handler

    async def make_call(self, to_number: str) -> Optional[str]:
        """Make an outbound call with media streaming."""
        try:
            # Create the call with media streaming enabled
            call = telnyx.Call.create(
                connection_id=self.config.sip_connection_id,
                to=to_number,
                from_=self.config.from_number,
                record_audio="dual",
                media_url=self.config.stream_config.stream_url,
                media_streaming_track=self.config.stream_config.stream_track,
                media_streaming_codec=self.config.stream_config.codec,
                media_streaming_sample_rate=self.config.stream_config.sampling_rate,
                media_streaming_channels=self.config.stream_config.channels
            )
            
            call_id = call.call_control_id
            self._active_calls[call_id] = call
            self._audio_queues[call_id] = asyncio.Queue()
            
            logger.info(f"Initiated call to {to_number} with ID: {call_id}")
            return call_id
            
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            return None
            
    async def _setup_websocket(self, call_id: str) -> None:
        """Set up WebSocket connection for media streaming."""
        try:
            ws_url = self.config.stream_config.stream_url
            logger.info(f"Setting up WebSocket for call {call_id} at URL: {ws_url}")
            
            # Add proper authentication headers
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Telnyx-Call-Control-ID": call_id,
                "User-Agent": "GLaDOS-Telephony/1.0",
                "Content-Type": "application/json"
            }

            # Connect to WebSocket with authentication headers
            ws = await websockets.connect(
                ws_url,
                subprotocols=["telnyx-media-v2"],
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=300
            )
            logger.info(f"WebSocket connected for call {call_id}")
            
            # Send initial media configuration
            start_message = {
                "event": "media_start",
                "call_control_id": call_id,
                "media": {
                    "track": self.config.stream_config.stream_track,
                    "codec": self.config.stream_config.codec,
                    "sampling_rate": self.config.stream_config.sampling_rate,
                    "channels": self.config.stream_config.channels
                }
            }
            await ws.send(json.dumps(start_message))
            logger.info(f"Sent media_start message: {json.dumps(start_message, indent=2)}")
            
            self._ws_connections[call_id] = ws
            
            # Start WebSocket handler tasks
            asyncio.create_task(self._handle_websocket_messages(call_id, ws))
            asyncio.create_task(self._handle_audio_queue(call_id, ws))
            
            logger.info(f"WebSocket handlers started for call {call_id}")
        except Exception as e:
            logger.error(f"Failed to setup WebSocket for call {call_id}: {e}")
            logger.exception(e)  # Log full traceback
            await self.end_call(call_id)

    async def _handle_websocket_messages(self, call_id: str, ws: websockets.WebSocketClientProtocol) -> None:
        """Handle incoming WebSocket messages."""
        try:
            logger.info(f"Started WebSocket message handler for call {call_id}")
            async for message in ws:
                if isinstance(message, bytes):
                    # Handle binary message (audio data)
                    logger.info(f"[TELNYX AUDIO] Received binary audio data for call {call_id}, size: {len(message)} bytes")
                    await self._handle_incoming_audio(call_id, message)
                else:
                    # Handle text message (control messages)
                    try:
                        data = json.loads(message)
                        logger.info(f"[TELNYX MSG] {json.dumps(data, indent=2)}")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in WebSocket message for call {call_id}")
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for call {call_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket message handler for call {call_id}: {e}")
        finally:
            # Clean up when the connection is closed
            if call_id in self._ws_connections:
                del self._ws_connections[call_id]
            logger.info(f"WebSocket handler ended for call {call_id}")

    async def _handle_incoming_audio(self, call_id: str, audio_data: bytes) -> None:
        """Handle incoming audio data from the call."""
        try:
            # Convert from bytes to numpy array with int16 type
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert from int16 to float32 in the range [-1, 1] for our audio processing
            audio_array = audio_array.astype(np.float32) / 32767.0
            
            # Reshape if needed for multi-channel audio
            if self.config.stream_config.channels > 1:
                audio_array = audio_array.reshape(-1, self.config.stream_config.channels)
            
            # Send the processed audio to the audio bridge (ASR pipeline)
            if call := self._active_calls.get(call_id):
                if hasattr(call, 'audio_bridge') and call.audio_bridge:
                    await call.audio_bridge.put_audio(audio_array)
                    logger.debug(f"Sent {len(audio_array)} samples to audio bridge for call {call_id}")
                else:
                    logger.warning(f"No audio bridge available for call {call_id}")
        except Exception as e:
            logger.error(f"Error processing incoming audio for call {call_id}: {e}")

    async def _handle_audio_queue(self, call_id: str, ws: websockets.WebSocketClientProtocol) -> None:
        """Handle outgoing audio queue."""
        try:
            while True:
                if call_id not in self._audio_queues:
                    break
                    
                audio_data = await self._audio_queues[call_id].get()
                if audio_data is None:  # Sentinel value to stop the queue
                    break
                    
                # Convert audio data to the correct format for Telnyx
                if isinstance(audio_data, np.ndarray):
                    # Convert from float32 [-1, 1] to int16 for PCMU
                    audio_data = (audio_data * 32767).astype(np.int16).tobytes()
                
                # Send audio data over WebSocket
                message = {
                    "event": "media",
                    "call_control_id": call_id,
                    "media": {
                        "payload": base64.b64encode(audio_data).decode(),
                        "sampling_rate": self.config.stream_config.sampling_rate,
                        "channels": self.config.stream_config.channels,
                        "codec": self.config.stream_config.codec
                    }
                }
                await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error in audio queue handler for call {call_id}: {e}")
            await self._cleanup_websocket(call_id)

    async def send_audio(self, call_id: str, audio_data: np.ndarray) -> None:
        """Send audio data to the call."""
        if call_id in self._audio_queues:
            # Ensure audio is in the correct format (float32 [-1, 1])
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / 32767.0  # Normalize if in int16 range
                
            logger.debug(f"Sending {len(audio_data)} samples of audio to call {call_id}")
            await self._audio_queues[call_id].put(audio_data)

    async def _cleanup_websocket(self, call_id: str) -> None:
        """Clean up WebSocket connection and related resources."""
        try:
            if ws := self._ws_connections.pop(call_id, None):
                await ws.close()
            if call_id in self._audio_queues:
                await self._audio_queues[call_id].put(None)  # Signal to stop
                del self._audio_queues[call_id]
            logger.info(f"Cleaned up WebSocket resources for call {call_id}")
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket for call {call_id}: {e}")

    async def end_call(self, call_id: str) -> bool:
        """End an active call."""
        try:
            if call := self._active_calls.get(call_id):
                call.hangup()
                await self._cleanup_websocket(call_id)
                del self._active_calls[call_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Error ending call {call_id}: {e}")
            return False

    async def handle_webhook(self, payload: Dict[str, Any]) -> None:
        """Handle incoming webhook events from Telnyx."""
        try:
            event_data = payload.get("data", {})
            event_type = event_data.get("event_type")
            payload_data = event_data.get("payload", {})
            call_id = payload_data.get("call_control_id")

            # Log the full webhook payload for debugging
            logger.info(f"[TELNYX WEBHOOK] Received event: {json.dumps(payload, indent=2)}")

            # Properly extract call_control_id for different event types
            if not call_id:
                call_id = event_data.get("id")  # Some events nest ID differently
            
            logger.info(f"Handling {event_type} for call {call_id}")

            # Auto-register call for any relevant event types
            if call_id and call_id not in self._active_calls:
                if event_type in ["call.initiated", "call.answered"]:
                    logger.info(f"Auto-registering call {call_id} from {event_type}")
                    call = await self._create_dummy_call(call_id)
                    self._active_calls[call_id] = call
                    self._audio_queues[call_id] = asyncio.Queue()
                    logger.debug(f"Active calls: {list(self._active_calls.keys())}")

            # Call the registered handler if exists
            if handler := self._call_handlers.get(event_type):
                await handler(payload)
                logger.debug(f"Handler executed for {event_type}")
            else:
                logger.debug(f"No handler registered for event type: {event_type}")
            
            # Handle media streaming setup
            if event_type == "call.answered":
                if call_id and call_id in self._active_calls:
                    logger.info(f"Setting up media streaming for answered call: {call_id}")
                    call = self._active_calls[call_id]
                    
                    # First, ensure we have the media URL
                    if not self.config.stream_config.stream_url:
                        logger.error("No media streaming URL configured!")
                        return
                        
                    logger.info(f"Using media URL: {self.config.stream_config.stream_url}")
                    
                    # Set up media streaming
                    await call.answer_media_streaming(
                        media_url=self.config.stream_config.stream_url,
                        media_streaming_track=self.config.stream_config.stream_track,
                        media_streaming_codec=self.config.stream_config.codec,
                        media_streaming_sample_rate=self.config.stream_config.sampling_rate,
                        media_streaming_channels=self.config.stream_config.channels
                    )
                    logger.info(f"Media streaming started for call: {call_id}")
                    
                    # Now we set up the WebSocket connection
                    await self._setup_websocket(call_id)
                else:
                    logger.warning(f"Call {call_id} not found in active calls for media setup")
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            logger.exception(e)  # Log full traceback

    async def _dummy_answer_media_streaming(self, **kwargs) -> None:
        """Dummy method for auto-registered calls."""
        logger.info("Using dummy answer_media_streaming for auto-registered call")
        return

    async def _create_dummy_call(self, call_id: str) -> Any:
        """Create a minimal call object for inbound calls."""
        call = type('Call', (), {
            'call_control_id': call_id,
            'answer_media_streaming': self._dummy_answer_media_streaming,
            'hangup': lambda *args, **kwargs: logger.info(f"Dummy hangup for {call_id}"),
            'send_audio': lambda x, *args, **kwargs: logger.debug(f"Dummy audio sent to {call_id}")
        })()
        return call