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


# In telnyx_client.py
class StreamConfig(BaseModel):
    """Configuration for media streaming."""
    stream_url: str = Field(..., description="WebSocket URL for media streaming")
    stream_track: str = Field(default="inbound_track", 
                             description="Track to stream (inbound_track|outbound_track|both_tracks)")
    stream_bidirectional: bool = Field(default=True, description="Enable bidirectional streaming")
    codec: str = Field(default="PCMU", description="Audio codec (PCMU, PCMA, G722)")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    max_duration: int = Field(default=3600, description="Maximum streaming duration in seconds")


class TelnyxConfig(BaseModel):
    """Configuration for Telnyx client."""
    api_key: str
    public_key: str
    sip_connection_id: str
    from_number: str
    webhook_url: str
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
        # Validate streaming config
        if config.stream_config.stream_bidirectional and not config.stream_config.stream_url:
            raise ValueError("Bidirectional streaming requires a valid stream_url")
        if config.stream_config.stream_track not in ['inbound_track', 'outbound_track', 'both_tracks']:
            raise ValueError("Invalid stream_track value - must be 'inbound_track', 'outbound_track' or 'both_tracks'")
        
        self.config = config
        telnyx.api_key = config.api_key
        self._call_handlers: Dict[str, Callable] = {}
        self._media_handlers: Dict[str, Callable] = {}
        self._active_calls: Dict[str, Any] = {}
        self._ws_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._audio_queues: Dict[str, asyncio.Queue] = {}
        self._dtmf_buffers: Dict[str, str] = {}  # Store DTMF sequences per call
        self._dtmf_timeouts: Dict[str, asyncio.Task] = {}  # DTMF timeout tasks
        self._setup_default_handlers()
        self._setup_media_handlers()
        
    def _setup_default_handlers(self) -> None:
        """Setup default handlers for common call events."""
        self.register_call_handler("call.initiated", self._handle_call_initiated)
        self.register_call_handler("call.answered", self._handle_call_answered)
        self.register_call_handler("call.hangup", self._handle_call_hangup)
        self.register_call_handler("call.media.streaming.started", self._handle_stream_started)
        self.register_call_handler("call.media.streaming.stopped", self._handle_stream_stopped)
        
    def _setup_media_handlers(self):
        self.register_media_handler("media", self._handle_media_packet)
        self.register_media_handler("dtmf", self._handle_dtmf_event)
        self.register_media_handler("mark", self._handle_mark_event)
        self.register_media_handler("error", self._handle_stream_error)

    async def make_call(self, to_number: str) -> Optional[str]:
        """
        Initiate an outbound call with media streaming.
        
        Args:
            to_number: The phone number to call in E.164 format
            
        Returns:
            call_control_id: The unique identifier for the call, or None if failed
        """
        try:
            call = telnyx.Call.create(
                connection_id=self.config.sip_connection_id,
                to=to_number,
                from_=self.config.from_number,
                webhook_url=self.config.webhook_url,
                client_state=base64.b64encode(json.dumps({
                    "app": "glados",
                    "type": "outbound_call"
                }).encode()).decode(),
                stream_url=self.config.stream_config.stream_url,
                stream_track=self.config.stream_config.stream_track,
                stream_bidirectional_mode="rtp",
                codec=self.config.stream_config.codec
            )
            call_id = call.call_control_id
            self._active_calls[call_id] = call
            self._audio_queues[call_id] = asyncio.Queue()
            
            logger.info(f"Initiated call to {to_number} with ID: {call_id}")
            return call_id
        except Exception as e:
            logger.error(f"Failed to initiate call: {e}")
            return None
            
    async def _setup_websocket(self, call_id: str):
        try:
            async with websockets.connect(
                self.config.stream_config.stream_url,
                extra_headers={'Call-ID': call_id}
            ) as ws:
                self._ws_connections[call_id] = ws
                self._audio_queues[call_id] = asyncio.Queue()
                await self._handle_websocket_messages(call_id, ws)
        except Exception as e:
            logger.error(f"WebSocket connection failed: {str(e)}")
            await self._cleanup_websocket(call_id)

    async def _handle_websocket_messages(self, call_id: str, ws: websockets.WebSocketClientProtocol):
        try:
            async for message in ws:
                try:
                    event = json.loads(message)
                    await self._process_media_event(call_id, event)
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON received for call {call_id}")
        except websockets.ConnectionClosed:
            logger.info(f"WebSocket connection closed for call {call_id}")
            await self._cleanup_websocket(call_id)

    async def _process_media_event(self, call_id: str, event: dict):
        event_type = event.get("event")
        handler = self._media_handlers.get(event_type)
        if handler:
            await handler(call_id, event)
        else:
            logger.debug(f"No handler for media event type: {event_type}")

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
                    
                # Convert audio data to the correct format for Telnyx
                if isinstance(audio_data, np.ndarray):
                    # Convert from float32 [-1, 1] to int16 for PCMU/OPUS
                    audio_data = (audio_data * 32767).astype(np.int16).tobytes()
                
                # Send audio data over WebSocket
                message = {
                    "event": "media",
                    "call_control_id": call_id,
                    "track": self.config.stream_config.stream_track,
                    "media": {
                        "payload": base64.b64encode(audio_data).decode(),
                        "sampling_rate": self.config.stream_config.sample_rate,
                        "channels": self.config.stream_config.channels
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
        """Clean up WebSocket connection and related resources."""
        try:
            if call_id in self._ws_connections:
                await self._ws_connections[call_id].close()
                del self._ws_connections[call_id]
            if call_id in self._audio_queues:
                del self._audio_queues[call_id]
            if call_id in self._dtmf_buffers:
                del self._dtmf_buffers[call_id]
            logger.info(f"Cleaned up WebSocket resources for call {call_id}")
        except Exception as e:
            logger.error(f"Error cleaning up WebSocket for call {call_id}: {e}")

    async def end_call(self, call_control_id: str) -> bool:
        """End an active call."""
        try:
            if call := self._active_calls.get(call_control_id):
                call.hangup()
                await self._cleanup_websocket(call_control_id)
                del self._active_calls[call_control_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Error ending call {call_control_id}: {e}")
            return False
            
    def register_call_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific call event type."""
        self._call_handlers[event_type] = handler

    def register_media_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for a specific media event type."""
        self._media_handlers[event_type] = handler
        
    async def handle_webhook(self, payload: str, signature_header: str, timestamp: str) -> None:
        """
        Handle incoming webhook events from Telnyx.
        
        Args:
            payload: Webhook payload
            signature_header: Signature header from the webhook request
            timestamp: Timestamp from the webhook request
        """
        try:
            # Convert base64 public key to PEM format
            pem_key = (
        "-----BEGIN PUBLIC KEY-----\n" +
        "\n".join([self.config.public_key[i:i+64] for i in range(0, len(self.config.public_key), 64)]) + 
        "\n-----END PUBLIC KEY-----"
    )
            event = telnyx.Webhook.construct_event(
                payload,
                signature_header,
                timestamp,
                pem_key
            )
            
            required_fields = ['id', 'created_at', 'event_type', 'data']
            missing = [field for field in required_fields 
                     if not hasattr(event, field)]
            
            if missing:
                logger.error("Invalid_webhook_payload",
                           missing_fields=missing,
                           payload=payload)
                return {"status": "invalid_payload"}
                
            logger.info("Webhook_received",
                      event_id=event.id,
                      event_type=event.event_type,
                      created_at=event.created_at)
            
            event_type = event.event_type
            logger.info(f"Processing webhook event: {event_type}")
            
            if handler := self._call_handlers.get(event_type):
                await handler(event.payload)
            else:
                logger.debug(f"No handler for event type: {event_type}")
            
            # Special handling for media streaming events
            if event_type == "call.answered":
                call_control_id = event.payload.get("call_control_id")
                if call_control_id:
                    await self._setup_websocket(call_control_id)
                
        except telnyx.error.SignatureVerificationError as e:
            logger.critical("Invalid_webhook_signature",
                          error=str(e),
                          received_signature=signature_header,
                          expected_public_key=self.config.public_key)
            raise
        except telnyx.error.TelnyxError as e:
            logger.error("Telnyx_api_error",
                       error=str(e),
                       http_status=e.http_status,
                       exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
                
    async def _handle_call_initiated(self, payload: Dict[str, Any]) -> None:
        """Handle call.initiated event."""
        call_id = payload["call_control_id"]
        logger.info(f"Call initiated: {call_id}")
        
    async def _handle_call_answered(self, payload: Dict[str, Any]) -> None:
        """Handle call.answered event."""
        try:
            call_id = payload.get("call_control_id")
            if not call_id:
                logger.error("No call_control_id in call.answered event")
                return
            
            logger.info(f"Call answered: {call_id}")
            
            # Start media streaming
            if call := self._active_calls.get(call_id):
                # Request media streaming
                await call.answer_media_streaming(
                    media_url=self.config.stream_config.stream_url,
                    media_streaming_track=self.config.stream_config.stream_track,
                    media_codec=self.config.stream_config.codec,
                    media_sample_rate=self.config.stream_config.sample_rate,
                    media_channels=self.config.stream_config.channels
                )
                logger.info(f"Media streaming requested for call: {call_id}")
                
                # Set up WebSocket after requesting media streaming
                await self._setup_websocket(call_id)
        except Exception as e:
            logger.error(f"Error handling call answered event: {e}")

    async def _handle_call_hangup(self, payload: Dict[str, Any]) -> None:
        """Handle call.hangup event."""
        try:
            call_id = payload["call_control_id"]
            logger.info(f"Call hung up: {call_id}")
            await self._cleanup_websocket(call_id)
            if call_id in self._active_calls:
                del self._active_calls[call_id]
        except Exception as e:
            logger.error(f"Error handling call hangup event: {e}")

    async def _handle_stream_started(self, payload: Dict[str, Any]) -> None:
        """Handle call.media.streaming.started event."""
        try:
            call_id = payload["call_control_id"]
            logger.info(f"Media streaming started for call: {call_id}")
        except Exception as e:
            logger.error(f"Error handling stream started event: {e}")

    async def _handle_stream_stopped(self, payload: Dict[str, Any]) -> None:
        """Handle call.media.streaming.stopped event."""
        try:
            call_id = payload["call_control_id"]
            logger.info(f"Media streaming stopped for call: {call_id}")
            await self._cleanup_websocket(call_id)
        except Exception as e:
            logger.error(f"Error handling stream stopped event: {e}")

    async def handle_media_stream(self, data: str):
        """
        Process incoming media stream data from WebSocket connection
        Args:
            data: Base64 encoded audio payload
        """
        try:
            # Decode and process audio data
            audio_data = base64.b64decode(data)
            await self.audio_bridge.process_incoming_audio(audio_data)
        except Exception as e:
            logger.error(f"Error processing media stream: {str(e)}")

    async def send_media_stream(self, websocket, audio_data: bytes):
        """
        Send audio data back through the WebSocket connection
        """
        try:
            encoded_data = base64.b64encode(audio_data).decode()
            await websocket.send_text(encoded_data)
        except Exception as e:
            logger.error(f"Error sending media stream: {str(e)}")

    async def _handle_media_packet(self, call_id: str, event: dict) -> None:
        """
        Handle incoming media packet event.
        
        Args:
            call_id: The call control ID
            event: The media packet event
        """
        try:
            media = event.get('media', {})
            if not (payload := media.get('payload')):
                return

            # Decode base64 audio
            raw_audio = base64.b64decode(payload)
            audio_array = np.frombuffer(raw_audio, dtype=np.int16)

            # Process audio through pipeline
            if call_id in self._audio_queues:
                await self._audio_queues[call_id].put(audio_array)
                
            logger.debug(f"Processed media packet for {call_id} ({len(raw_audio)} bytes)")

        except Exception as e:
            logger.error(f"Media processing error: {str(e)}")

    async def _handle_dtmf_event(self, call_id: str, event: dict) -> None:
        """
        Handle incoming DTMF event.
        
        Args:
            call_id: The call control ID
            event: The DTMF event
        """
        try:
            digit = event.get('dtmf', {}).get('digit')
            if digit and call_id in self._dtmf_buffers:
                self._dtmf_buffers[call_id] += digit
                logger.info(f"DTMF input for {call_id}: {digit}")
                
                # Reset timeout
                if task := self._dtmf_timeouts.get(call_id):
                    task.cancel()
                self._dtmf_timeouts[call_id] = asyncio.create_task(
                    self._clear_dtmf_buffer(call_id)
                )

        except Exception as e:
            logger.error(f"DTMF handling error: {str(e)}")

    async def _handle_mark_event(self, call_id: str, event: dict) -> None:
        """
        Handle incoming mark event.
        
        Args:
            call_id: The call control ID
            event: The mark event
        """
        try:
            # Process mark event
            logger.info(f"Received mark event for call {call_id}")
        except Exception as e:
            logger.error(f"Error handling mark event: {e}")

    async def _handle_stream_error(self, call_id: str, event: dict) -> None:
        """
        Handle incoming stream error event.
        
        Args:
            call_id: The call control ID
            event: The stream error event
        """
        try:
            # Process stream error event
            logger.error(f"Received stream error event for call {call_id}")
        except Exception as e:
            logger.error(f"Error handling stream error event: {e}")

    async def send_media(self, call_id: str, audio_data: bytes) -> None:
        if call_id not in self._ws_connections:
            logger.error(f"No active WebSocket for {call_id}")
            return

        try:
            media_payload = {
                'event': 'media',
                'media': {
                    'payload': base64.b64encode(audio_data).decode('utf-8'),
                    'contentType': 'audio/x-raw',
                    'sampleRate': self.config.stream_config.sample_rate
                }
            }
            await self._ws_connections[call_id].send(json.dumps(media_payload))
        except Exception as e:
            logger.error(f"Media send error: {str(e)}")

    async def _clear_dtmf_buffer(self, call_id: str) -> None:
        await asyncio.sleep(5)  # 5 second timeout
        if call_id in self._dtmf_buffers:
            logger.info(f"Clearing DTMF buffer for call {call_id}")
            self._dtmf_buffers[call_id] = ""

    async def process_audio_stream(self, call_id: str) -> None:
        while call_id in self._active_calls:
            try:
                audio_data = await self._audio_queues[call_id].get()
                # Process audio through pipeline
                processed_audio = await self._audio_pipeline.process(audio_data)
                await self.send_media(call_id, processed_audio)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio processing error: {str(e)}")