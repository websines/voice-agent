"""
Webhook server for handling Telnyx events.
"""

from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
import telnyx
import os
from .telnyx_client import TelnyxClient
from telnyx.error import SignatureVerificationError

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


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    Handle incoming Telnyx webhooks.
    
    This endpoint receives webhooks from Telnyx and forwards them to the TelnyxClient
    for processing.
    """
    if not telnyx_client:
        raise HTTPException(status_code=500, detail="TelnyxClient not initialized")
    
    try:
        # Get the raw body for signature verification
        body = await request.body()
        
        # Verify the webhook signature
        signature = request.headers.get("telnyx-signature-ed25519", "")
        timestamp = request.headers.get("telnyx-timestamp", "")
        
        telnyx.public_key = "jLZ5RdA4O1WS38Svtc5Y5eBcWYOwWFAMr4+oSqQwDJo="
        
        try:
            event = telnyx.Webhook.construct_event(
                body,
                signature,
                timestamp,
                tolerance=300
            )
        except ValueError as e:
            logger.error(f"Invalid payload: {str(e)}")
            raise HTTPException(status_code=400)
        except SignatureVerificationError as e:
            logger.error(f"Invalid signature: {str(e)}")
            raise HTTPException(status_code=400)
        
        # Forward processed event
        await telnyx_client.handle_webhook(event)
        
        return JSONResponse(content={"status": "success"})
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


def start_webhook_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the webhook server.
    
    Args:
        host: The host to bind to
        port: The port to listen on
    """
    uvicorn.run(app, host=host, port=port) 