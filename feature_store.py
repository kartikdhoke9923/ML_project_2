# Handles all feature engineering for inference
# Key design decisions:
# - C-columns: approximated from transaction history (card/address/email counts)
# - D-columns: computed from timestamps in history
# - M-columns: passed directly from payment processor input
# - All other features: computed from the raw transaction fields

import numpy as np
import pandas as pd
import time
from typing import Optional

RISKY_DOMAINS = {
    "anonymous.com", "guerrillamail.com", "mailnull.com",
    "suremail.info", "spambog.com", "trashmail.com"
}
