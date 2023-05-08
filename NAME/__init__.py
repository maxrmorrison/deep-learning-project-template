###############################################################################
# Configuration
###############################################################################


# Default configuration parameters to be modified
from .config import defaults

# Modify configuration
import yapecs
yapecs.configure('NAME', defaults)

# Import configuration parameters
from .config.defaults import *
try:
    from .config.secrets import *
except ImportError:
    pass
from .config.static import *


###############################################################################
# Module imports
###############################################################################


from .core import *
from .model import Model
from . import checkpoint
from . import data
from . import evaluate
from . import load
from . import partition
from . import time
from . import train
from . import write
