# GLaDOS Telephony Module

This module provides SIP-based voice call capabilities to GLaDOS using the Telnyx API. It enables GLaDOS to make outbound calls and interact with callers using its TTS and ASR capabilities through real-time media streaming.

## Features

- Outbound call support via Telnyx SIP
- Real-time bidirectional audio streaming
- Event-based call handling
- Automatic audio format conversion
- Bidirectional audio bridging with GLaDOS TTS/ASR
- Support for both PCMU and OPUS codecs
- Configurable audio settings (sample rate, channels)

## Configuration

Create `configs/telnyx_config.yaml` with your settings:

```yaml
# Telnyx API credentials
api_key: "YOUR_API_KEY"  # Replace with your Telnyx API key
sip_connection_id: "YOUR_CONNECTION_ID"  # Replace with your Telnyx SIP connection ID
from_number: "+1234567890"  # Your Telnyx phone number

# Webhook settings
webhook_url: "https://glados-telnyx.your-domain.com/telnyx/webhook"  # Your Cloudflare Tunnel URL

# Media streaming configuration
stream_config:
  stream_url: "wss://glados-telnyx.your-domain.com/media"  # WebSocket URL for media streaming
  stream_track: "inbound_track"  # Track identifier
  stream_bidirectional: true  # Enable two-way audio
  codec: "PCMU"  # Audio codec (PCMU or OPUS)
  sampling_rate: 8000  # Audio sampling rate in Hz
  channels: 1  # Number of audio channels (mono)

# Call handling
max_concurrent_calls: 1  # Maximum number of concurrent calls allowed
call_timeout: 300  # Call timeout in seconds
```

## Usage

⚠️ Make sure your Cloudflare Tunnel is running before making calls!

### Command Line

Make an outbound call:
```bash
uv glados call "+1234567890"  # Replace with the target phone number in E.164 format
```

Use a custom config file:
```bash
uv glados call "+1234567890" --config /path/to/custom/telnyx_config.yaml
```

### Programmatic Usage

```python
from glados.Telephony import TelnyxClient, CallManager
from glados.Telephony.telnyx_client import TelnyxConfig, StreamConfig
import yaml

# Load config
with open("configs/telnyx_config.yaml", "r") as f:
    config = yaml.safe_load(f)
telnyx_config = TelnyxConfig(**config)

# Initialize components
client = TelnyxClient(telnyx_config)
manager = CallManager(client)

# Make a call
call_id = await manager.start_call("+1234567890")
```

## Module Structure

- `telnyx_client.py`: Handles Telnyx API interactions, call control, and media streaming
- `call_manager.py`: Manages active call sessions and coordinates components
- `audio_bridge.py`: Handles real-time audio streaming between Telnyx and GLaDOS

## Requirements

- Telnyx account with:
  - Valid API key
  - SIP connection configured
  - Phone number for outbound calling
- Cloudflare Tunnel (required for both outbound calls and webhooks)
- Python packages:
  - telnyx>=2.0.0
  - sounddevice>=0.5.1
  - numpy
  - pydantic

## Cloudflare Tunnel Setup (Required)

A Cloudflare Tunnel is required to:
1. Connect GLaDOS to Telnyx's Voice API for outbound calls and media streaming
2. Receive call events via webhooks (answered, hangup, etc.)

Here's how to set it up:

1. Install cloudflared:
   ```bash
   # macOS
   brew install cloudflared

   # Linux
   curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   sudo dpkg -i cloudflared.deb

   # Windows (using scoop)
   scoop install cloudflared
   ```

2. Log in to Cloudflare:
   ```bash
   cloudflared tunnel login
   ```

3. Create a tunnel:
   ```bash
   cloudflared tunnel create glados-telnyx
   ```

4. Configure the tunnel (create `~/.cloudflared/config.yml`):
   ```yaml
   tunnel: <YOUR-TUNNEL-ID>
   credentials-file: /Users/<username>/.cloudflared/<TUNNEL-ID>.json

   ingress:
     - hostname: glados-telnyx.your-domain.com
       service: http://localhost:8000  # Your local webhook server port
     - service: http_status:404
   ```

5. Start the tunnel:
   ```bash
   cloudflared tunnel run glados-telnyx
   ```

6. In your Telnyx Portal:
   - Go to Voice > Outbound Settings
   - Set Connection Type to "Webhooks"
   - Set your Webhook URL to your tunnel URL (e.g., "https://glados-telnyx.your-domain.com")
   - Under Media Streaming, enable "Stream Audio to External Service"
   - Set the Media Format to match your config (default: PCMU, 8000 Hz)

## Error Handling

The module includes comprehensive error handling for:
- Failed call attempts
- Audio streaming issues
- Network connectivity problems
- Invalid configurations
- Media streaming errors
- WebSocket connection issues

Errors are logged using the `loguru` logger and can be monitored in your application logs.

## Contributing

When contributing to this module:
1. Follow the existing code structure and style
2. Add comprehensive docstrings and type hints
3. Include error handling for all external interactions
4. Update tests if adding new functionality
5. Test media streaming with both PCMU and OPUS codecs
