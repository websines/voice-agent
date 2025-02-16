import logging
import structlog
from pathlib import Path
import yaml

def configure_logging():
    """Initialize structured logging configuration"""
    config_path = Path(__file__).parent / 'logging.yaml'
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)
    
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()
