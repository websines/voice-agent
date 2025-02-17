"""
Webhook server for handling Telnyx events.
"""

from typing import Optional
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger
from .telnyx_client import TelnyxClient
import asyncio
import json

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


async def process_webhook_background(payload: dict):
    """Process webhook in background."""
    try:
        if not telnyx_client:
            logger.error("TelnyxClient not initialized")
            return

        # Extract event details
        data = payload.get("data", {})
        event_type = data.get("event_type")
        call_control_id = data.get("payload", {}).get("call_control_id")
        
        logger.info(f"[TELNYX WEBHOOK] Processing {event_type} for call {call_control_id}")
        logger.info(f"[TELNYX WEBHOOK] Full payload: {json.dumps(payload, indent=2)}")
        
        # Forward to TelnyxClient for processing
        await telnyx_client.handle_webhook(payload)
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        logger.exception(e)


@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handle incoming Telnyx webhooks.
    
    This endpoint receives webhooks from Telnyx and processes them in the background
    to avoid timeouts.
    """
    try:
        # Get raw body first
        body = await request.body()
        
        try:
            # Try to parse JSON
            payload = json.loads(body)
            logger.info("[TELNYX WEBHOOK] Received webhook request")
            
            # Add to background tasks
            background_tasks.add_task(process_webhook_background, payload)
            
            # Respond immediately
            return JSONResponse(
                status_code=200,
                content={"message": "Webhook received"}
            )
            
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in webhook body: {body}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON"}
            )
            
    except Exception as e:
        logger.error(f"Error in webhook handler: {e}")
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


def start_webhook_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the webhook server.
    
    Args:
        host: The host to bind to
        port: The port to listen on
    """
    logger.info(f"Starting webhook server on http://{host}:{port}")
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=30,
        limit_concurrency=100,
        backlog=2048
    )
    server = uvicorn.Server(config)
    server.run()