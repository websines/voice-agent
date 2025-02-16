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
                record_audio="dual",  # Enable recording for media streaming
                media_url=self.config.stream_config.stream_url,
                media_streaming_track=self.config.stream_config.stream_track,
                media_streaming_codec=self.config.stream_config.codec,
                media_streaming_sample_rate=self.config.stream_config.sampling_rate,
                media_streaming_channels=self.config.stream_config.channels
            )
            
            call_id = call.call_control_id
            self._active_calls[call_id] = call
            self._audio_queues[call_id] = asyncio.Queue()
            
            # Set up WebSocket connection for media streaming
            await self._setup_websocket(call_id)
            
            logger.info(f"Initiated call to {to_number} with ID: {call_id}")
            return call_id
            
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            return None
            
    async def _setup_websocket(self, call_id: str) -> None:
        """Set up WebSocket connection for media streaming."""
        try:
            # Construct WebSocket URL with authentication
            ws_url = f"{self.config.stream_config.stream_url}"
            
            # Set up headers with bearer token authentication
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Connect to WebSocket with headers
            ws = await websockets.connect(
                ws_url,
                additional_headers=headers,
                subprotocols=["telnyx-media-v2"]
            )
            
            # Send initial media configuration
            await ws.send(json.dumps({
                "event": "media_start",
                "call_control_id": call_id,
                "media": {
                    "track": self.config.stream_config.stream_track,
                    "codec": self.config.stream_config.codec,
                    "sampling_rate": self.config.stream_config.sampling_rate,
                    "channels": self.config.stream_config.channels
                }
            }))
            
            self._ws_connections[call_id] = ws
            
            # Start WebSocket handler tasks
            asyncio.create_task(self._handle_websocket_messages(call_id, ws))
            asyncio.create_task(self._handle_audio_queue(call_id, ws))
            
            logger.info(f"WebSocket connection established for call {call_id}")
        except Exception as e:
            logger.error(f"Failed to setup WebSocket for call {call_id}: {e}")
            await self.end_call(call_id)

    async def _handle_websocket_messages(self, call_id: str, ws: websockets.WebSocketClientProtocol) -> None:
        """Handle incoming WebSocket messages."""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == "media":
                        # Handle incoming audio data
                        audio_data = base64.b64decode(data["media"]["payload"])
                        await self._handle_incoming_audio(call_id, audio_data)
                    elif event_type == "media_start_success":
                        logger.info(f"Media streaming started successfully for call {call_id}")
                    elif event_type == "error":
                        logger.error(f"WebSocket error for call {call_id}: {data.get('error', 'Unknown error')}")
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received for call {call_id}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message for call {call_id}: {e}")
        except Exception as e:
            logger.error(f"WebSocket connection error for call {call_id}: {e}")
            await self._cleanup_websocket(call_id)

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
            await self._audio_queues[call_id].put(audio_data)

    async def _handle_incoming_audio(self, call_id: str, audio_data: bytes) -> None:
        """Handle incoming audio data from the call."""
        try:
            # Convert from bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Convert from int16 to float32 [-1, 1] for our audio processing
            audio_array = audio_array.astype(np.float32) / 32767.0
            
            # Reshape if needed for channels
            if self.config.stream_config.channels > 1:
                audio_array = audio_array.reshape(-1, self.config.stream_config.channels)
            
            # Store in active calls for processing
            if call_id in self._active_calls:
                self._active_calls[call_id].audio_data = audio_array
                logger.debug(f"Processed {len(audio_data)} bytes of audio for call {call_id}")
        except Exception as e:
            logger.error(f"Error processing incoming audio for call {call_id}: {e}")

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
            event_type = payload.get("event_type")
            logger.debug(f"Received webhook event: {event_type}")
            
            # Call the registered handler if exists
            if handler := self._call_handlers.get(event_type):
                await handler(payload)
            
            # Handle media streaming setup
            if event_type == "call.answered":
                call_id = payload.get("payload", {}).get("call_control_id")
                if call_id and call_id in self._active_calls:
                    # Start media streaming when call is answered
                    call = self._active_calls[call_id]
                    await call.answer_media_streaming(
                        media_url=self.config.stream_config.stream_url,
                        media_streaming_track=self.config.stream_config.stream_track,
                        media_streaming_codec=self.config.stream_config.codec,
                        media_streaming_sample_rate=self.config.stream_config.sampling_rate,
                        media_streaming_channels=self.config.stream_config.channels
                    )
                    logger.info(f"Media streaming started for call: {call_id}")
                    
                    # Set up WebSocket after media streaming is started
                    await self._setup_websocket(call_id)
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")