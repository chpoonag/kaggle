from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseCFG(ABC):
    """Abstract base class for ML experiment configurations."""
    
    @classmethod
    @abstractmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert class attributes to serializable dict."""
        return {k: v for k, v in cls.__dict__.items() 
                if not (k.startswith('_') or callable(v))}
