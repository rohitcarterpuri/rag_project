import logging

import nltk

from safe_output import fail, install

# Sanitize console output before anything else can print or raise
install()

# NLTK's own error logs can include the local data directory path
logging.getLogger("nltk").setLevel(logging.CRITICAL)

results = [nltk.download("punkt"), nltk.download("punkt_tab")]

if False in results:
    fail("Could not download the NLTK 'punkt' data. Check your network connection.")
