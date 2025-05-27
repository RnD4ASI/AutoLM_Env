from loguru import logger

def get_logger(name: str):
    """ 
    Returns a configured logger instance with custom formatting.
    
    Parameters:
        name (str): The name of the logger.
    
    Returns:
        logger: Configured loguru logger instance.
    """
    # Configure the logger with custom format
    format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    
    # Remove default handler and add custom handler
    logger.remove()
    logger.add(sink=lambda msg: print(msg, end=""), format=format_string, level="DEBUG")
    
    # Create a contextualized logger with the module name
    contextualized_logger = logger.bind(name=name)
    
    return contextualized_logger
