"""
Webhook server for handling Telnyx events.
"""

from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
from .telnyx_client import TelnyxClient

app = FastAPI()

# Global variable to store the TelnyxClient instance
telnyx_client: Optional[TelnyxClient] = None


def initialize_webhook_server(client: TelnyxClient) -> None:
    """
    Initialize the webhook server with a TelnyxClient instance.
    
    Args:
        client: The TelnyxClient instance to handle webhooks
    """
    global telnyx_client
    telnyx_client = client
    logger.info("Webhook server initialized with TelnyxClient")


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle incoming Telnyx webhooks.
    
    This endpoint receives webhooks from Telnyx and forwards them to the TelnyxClient
    for processing.
    """
    if not telnyx_client:
        logger.error("TelnyxClient not initialized")
        return JSONResponse(status_code=500, content={"error": "Server not initialized"})
    
    try:
        # Get the webhook payload
        payload = await request.json()
        
        # Log the full payload for debugging
        logger.debug(f"Received webhook payload: {payload}")
        
        # Extract event details
        data = payload.get("data", {})
        event_type = data.get("event_type")
        call_control_id = data.get("payload", {}).get("call_control_id")
        
        logger.info(f"Processing webhook: {event_type} for call {call_control_id}")
        
        # Forward to TelnyxClient for processing
        await telnyx_client.handle_webhook(data)
        
        return JSONResponse(content={"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return JSONResponse(status_code=400, content={"error": str(e)})


def start_webhook_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the webhook server.
    
    Args:
        host: The host to bind to
        port: The port to listen on
    """
    logger.info(f"Starting webhook server on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port) 