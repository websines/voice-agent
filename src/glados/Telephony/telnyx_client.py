"""
TelnyxClient handles all interactions with the Telnyx API for voice calls.
"""

from typing import Optional, Callable, Dict, Any, List
import asyncio
import base64
import json
from loguru import logger
import telnyx
import websockets
from pydantic import BaseModel, Field
import numpy as np


class DTMFCommand(BaseModel):
    """Configuration for a DTMF command."""
    digits: str = Field(..., description="The DTMF digits sequence")
    description: str = Field(..., description="Description of what this command does")
    handler: Optional[str] = Field(None, description="Name of the handler function")


class StreamConfig(BaseModel):
    """Configuration for media streaming."""
    stream_url: str = Field(..., description="WebSocket URL for media streaming")
    stream_track: str = Field(default="inbound_track", description="Track identifier for the media stream")
    stream_bidirectional: bool = Field(default=True, description="Enable bidirectional streaming")
    codec: str = Field(default="PCMU", description="Audio codec (PCMU or OPUS)")
    sampling_rate: int = Field(default=8000, description="Audio sampling rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")


class TelnyxConfig(BaseModel):
    """Configuration for Telnyx client."""
    api_key: str
    sip_connection_id: str
    from_number: str
    webhook_url: Optional[str] = None
    stream_config: StreamConfig
    dtmf_commands: Optional[Dict[str, DTMFCommand]] = Field(
        default_factory=dict,
        description="DTMF command configurations"
    )


class TelnyxClient:
    """
    Handles all Telnyx API interactions for voice calls.
    
    This class manages the connection to Telnyx's API, handles incoming webhooks,
    and provides methods for making outbound calls with media streaming support.
    """
    
    def __init__(self, config: TelnyxConfig):
        """
        Initialize the Telnyx client.
        
        Args:
            config: TelnyxConfig object containing API credentials and settings
        """
        self.config = config
        telnyx.api_key = config.api_key
        self._call_handlers: Dict[str, Callable] = {}
        self._active_calls: Dict[str, Any] = {}
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._audio_queues: Dict[str, asyncio.Queue] = {}
        self._dtmf_buffers: Dict[str, str] = {}  # Store DTMF sequences per call
        self._dtmf_timeouts: Dict[str, asyncio.Task] = {}  # DTMF timeout tasks
        self._setup_default_handlers()
        
    def _setup_default_handlers(self) -> None:
        """Setup default handlers for common call events."""
        self.register_call_handler("call.initiated", self._handle_call_initiated)
        self.register_call_handler("call.answered", self._handle_call_answered)
        self.register_call_handler("call.hangup", self._handle_call_hangup)
        self.register_call_handler("call.media.streaming.started", self._handle_stream_started)
        self.register_call_handler("call.media.streaming.stopped", self._handle_stream_stopped)
        
    async def make_call(self, to_number: str) -> Optional[str]:
        """
        Initiate an outbound call with media streaming.
        
        Args:
            to_number: The phone number to call in E.164 format
            
        Returns:
            call_control_id: The unique identifier for the call, or None if failed
        """
        try:
            stream_config = self.config.stream_config
            call = telnyx.Call.create(
                connection_id=self.config.sip_connection_id,
                to=to_number,
                from_=self.config.from_number,
                webhook_url=self.config.webhook_url,
                # Media streaming configuration
                stream_url=stream_config.stream_url,
                stream_track=stream_config.stream_track,
                stream_bidirectional=stream_config.stream_bidirectional,
                audio_codec=stream_config.codec,
                sampling_rate=stream_config.sampling_rate,
                channels=stream_config.channels
            )
            call_id = call.call_control_id
            self._active_calls[call_id] = call
            self._audio_queues[call_id] = asyncio.Queue()
            
            # Start WebSocket connection for media streaming
            await self._setup_websocket(call_id)
            
            logger.info(f"Initiated call to {to_number} with ID: {call_id}")
            return call_id
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            return None
            
    async def _setup_websocket(self, call_id: str) -> None:
        """
        Set up WebSocket connection for media streaming.
        
        Args:
            call_id: The call control ID
        """
        try:
            ws = await websockets.connect(self.config.stream_config.stream_url)
            self._ws_connections[call_id] = ws
            
            # Start WebSocket handler tasks
            asyncio.create_task(self._handle_websocket_messages(call_id, ws))
            asyncio.create_task(self._handle_audio_queue(call_id, ws))
            
            logger.info(f"WebSocket connection established for call {call_id}")
        except Exception as e:
            logger.error(f"Failed to setup WebSocket for call {call_id}: {e}")
            await self.end_call(call_id)

    async def _handle_websocket_messages(self, call_id: str, ws: websockets.WebSocketClientProtocol) -> None:
        """
        Handle incoming WebSocket messages.
        
        Args:
            call_id: The call control ID
            ws: WebSocket connection
        """
        try:
            async for message in ws:
                try:
                    data = json.loads(message)
                    event_type = data.get("event")
                    
                    if event_type == "connected":
                        logger.info(f"Media WebSocket connected for call {call_id}")
                    elif event_type == "start":
                        logger.info(f"Media streaming started for call {call_id}")
                    elif event_type == "media":
                        # Handle incoming audio data
                        audio_data = base64.b64decode(data["media"]["payload"])
                        await self._handle_incoming_audio(call_id, audio_data)
                    elif event_type == "dtmf":
                        digit = data["dtmf"]["digit"]
                        logger.info(f"DTMF received for call {call_id}: {digit}")
                        await self._handle_dtmf(call_id, digit)
                    elif event_type == "error":
                        logger.error(f"WebSocket error for call {call_id}: {data['payload']}")
                    elif event_type == "stop":
                        logger.info(f"Media streaming stopped for call {call_id}")
                        await self._cleanup_websocket(call_id)
                        break
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received for call {call_id}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message for call {call_id}: {e}")
        except Exception as e:
            logger.error(f"WebSocket connection error for call {call_id}: {e}")
            await self._cleanup_websocket(call_id)

    async def _handle_audio_queue(self, call_id: str, ws: websockets.WebSocketClientProtocol) -> None:
        """
        Handle outgoing audio queue.
        
        Args:
            call_id: The call control ID
            ws: WebSocket connection
        """
        try:
            while True:
                if call_id not in self._audio_queues:
                    break
                    
                audio_data = await self._audio_queues[call_id].get()
                if audio_data is None:  # Sentinel value to stop the queue
                    break
                    
                # Send audio data over WebSocket
                message = {
                    "event": "media",
                    "media": {
                        "payload": base64.b64encode(audio_data).decode()
                    }
                }
                await ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error in audio queue handler for call {call_id}: {e}")
            await self._cleanup_websocket(call_id)

    async def send_audio(self, call_id: str, audio_data: np.ndarray) -> None:
        """
        Send audio data to the call.
        
        Args:
            call_id: The call control ID
            audio_data: Audio data as numpy array
        """
        if call_id in self._audio_queues:
            await self._audio_queues[call_id].put(audio_data.tobytes())

    async def _handle_incoming_audio(self, call_id: str, audio_data: bytes) -> None:
        """
        Handle incoming audio data from the call.
        
        Args:
            call_id: The call control ID
            audio_data: Raw audio data
        """
        # Convert to numpy array for processing
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # TODO: Process audio data (pass to ASR, etc.)
        # For now, just log receipt
        logger.debug(f"Received {len(audio_data)} bytes of audio for call {call_id}")

    async def _handle_dtmf(self, call_id: str, digit: str) -> None:
        """
        Handle incoming DTMF digit and check for command matches.
        
        Args:
            call_id: The call control ID
            digit: The DTMF digit received
        """
        # Initialize or update DTMF buffer for this call
        if call_id not in self._dtmf_buffers:
            self._dtmf_buffers[call_id] = ""
        
        # Add digit to buffer
        self._dtmf_buffers[call_id] += digit
        current_sequence = self._dtmf_buffers[call_id]
        
        # Cancel existing timeout task if any
        if call_id in self._dtmf_timeouts:
            self._dtmf_timeouts[call_id].cancel()
        
        # Set new timeout to clear buffer
        async def clear_dtmf_buffer():
            await asyncio.sleep(2.0)  # Wait 2 seconds for more digits
            if call_id in self._dtmf_buffers:
                del self._dtmf_buffers[call_id]
                del self._dtmf_timeouts[call_id]
        
        self._dtmf_timeouts[call_id] = asyncio.create_task(clear_dtmf_buffer())
        
        # Check for command matches
        for cmd_digits, command in self.config.dtmf_commands.items():
            if current_sequence.endswith(cmd_digits):
                # Found a match
                logger.info(f"DTMF command matched: {cmd_digits} ({command.description})")
                if handler := self._call_handlers.get(f"dtmf_{cmd_digits}"):
                    try:
                        await handler(call_id)
                        # Clear buffer after successful command
                        del self._dtmf_buffers[call_id]
                        if call_id in self._dtmf_timeouts:
                            self._dtmf_timeouts[call_id].cancel()
                            del self._dtmf_timeouts[call_id]
                    except Exception as e:
                        logger.error(f"Error executing DTMF command handler: {e}")

    async def _cleanup_websocket(self, call_id: str) -> None:
        """
        Clean up WebSocket resources for a call.
        
        Args:
            call_id: The call control ID
        """
        try:
            if ws := self._ws_connections.pop(call_id, None):
                await ws.close()
            
            if call_id in self._audio_queues:
                await self._audio_queues[call_id].put(None)  # Signal queue to stop
                del self._audio_queues[call_id]
            
            # Clean up DTMF resources
            if call_id in self._dtmf_buffers:
                del self._dtmf_buffers[call_id]
            if call_id in self._dtmf_timeouts:
                self._dtmf_timeouts[call_id].cancel()
                del self._dtmf_timeouts[call_id]
                
            logger.info(f"WebSocket cleanup completed for call {call_id}")
        except Exception as e:
            logger.error(f"Error during WebSocket cleanup for call {call_id}: {e}")

    async def end_call(self, call_id: str) -> bool:
        """
        End an active call.
        
        Args:
            call_id: The call control ID
            
        Returns:
            bool: True if call was ended successfully, False otherwise
        """
        try:
            if call := self._active_calls.get(call_id):
                # Clean up WebSocket first
                await self._cleanup_websocket(call_id)
                
                # End the call
                call.hangup()
                del self._active_calls[call_id]
                logger.info(f"Ended call: {call_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to end call: {e}")
            return False
            
    def register_call_handler(self, event: str, handler: Callable) -> None:
        """
        Register a handler for specific call events.
        
        Args:
            event: The event type to handle (e.g., 'answered', 'hangup')
            handler: Callback function to handle the event
        """
        self._call_handlers[event] = handler
        
    async def handle_webhook(self, payload: Dict[str, Any]) -> None:
        """
        Handle incoming webhooks from Telnyx.
        
        Args:
            payload: The webhook payload from Telnyx
        """
        event_type = payload.get("data", {}).get("event_type")
        call_control_id = payload.get("data", {}).get("payload", {}).get("call_control_id")
        
        if event_type and call_control_id:
            if handler := self._call_handlers.get(event_type):
                try:
                    await handler(payload)
                except Exception as e:
                    logger.error(f"Error in call handler for {event_type}: {e}")
            else:
                logger.debug(f"No handler registered for event: {event_type}")
                
    async def _handle_call_initiated(self, payload: Dict[str, Any]) -> None:
        """Handle call.initiated event."""
        call_id = payload["data"]["payload"]["call_control_id"]
        logger.info(f"Call initiated: {call_id}")
        
    async def _handle_call_answered(self, payload: Dict[str, Any]) -> None:
        """Handle call.answered event."""
        call_id = payload["data"]["payload"]["call_control_id"]
        logger.info(f"Call answered: {call_id}")
        
    async def _handle_call_hangup(self, payload: Dict[str, Any]) -> None:
        """Handle call.hangup event."""
        call_id = payload["data"]["payload"]["call_control_id"]
        logger.info(f"Call hung up: {call_id}")
        
    async def _handle_stream_started(self, payload: Dict[str, Any]) -> None:
        """Handle call.media.streaming.started event."""
        call_id = payload["data"]["payload"]["call_control_id"]
        logger.info(f"Media streaming started for call: {call_id}")
        
    async def _handle_stream_stopped(self, payload: Dict[str, Any]) -> None:
        """Handle call.media.streaming.stopped event."""
        call_id = payload["data"]["payload"]["call_control_id"]
        await self._cleanup_websocket(call_id)
        logger.info(f"Media streaming stopped for call: {call_id}") 