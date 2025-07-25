from .metrics import MetricsManager
from .logger import get_logger, configure_logger

# Lazy import helpers to avoid transformers dependency at init
def get_text_helper():
    from .helpers import TextHelper
    return TextHelper

def get_document_helper():
    from .helpers import DocumentHelper
    return DocumentHelper

def get_file_helper():
    from .helpers import FileHelper
    return FileHelper

def get_validation_helper():
    from .helpers import ValidationHelper
    return ValidationHelper

__all__ = [
    'MetricsManager',
    'get_text_helper',
    'get_document_helper', 
    'get_file_helper',
    'get_validation_helper',
    'get_logger',
    'configure_logger'
]
