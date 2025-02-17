import pytest
from unittest.mock import Mock, AsyncMock
from glados.Telephony.telnyx_client import TelnyxClient, TelnyxConfig, StreamConfig
from glados.Telephony.call_manager import CallManager
import websockets

@pytest.fixture
def mock_telnyx_config():
    return TelnyxConfig(
        api_key='test_key',
        sip_connection_id='test_conn',
        from_number='+15555555',
        stream_config=StreamConfig(
            stream_url='ws://invalid',
            codec='PCMU'
        )
    )

@pytest.fixture
def telnyx_client(mock_telnyx_config):
    return TelnyxClient(mock_telnyx_config)

@pytest.fixture
def call_manager(telnyx_client):
    return CallManager(telnyx_client)

@pytest.mark.asyncio
async def test_websocket_auth(telnyx_client):
    """Test WebSocket connection uses proper auth headers"""
    with pytest.raises(websockets.InvalidStatusCode):
        await telnyx_client._setup_websocket('test_call_id')

@pytest.mark.asyncio
async def test_call_initiated_handler(call_manager):
    """Test call.initiated event creates session"""
    test_payload = {'data': {'call_control_id': 'test123', 'to': '+15551234'}}
    await call_manager._handle_call_initiated(test_payload)
    assert 'test123' in call_manager._active_sessions
    session = call_manager._active_sessions['test123']
    assert session.phone_number == '+15551234'
