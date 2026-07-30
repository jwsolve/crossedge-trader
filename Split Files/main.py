#!/usr/bin/env python3
"""
Auxo trading bot – entry point.
"""

import os
import logging
from pathlib import Path

from bot import PaperBot
from web_server import BotRequestHandler, ThreadingHTTPServer

BASE_DIR = Path(__file__).resolve().parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auxo.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('auxo')

def main():
    port = int(os.environ.get("PORT", "8080"))
    bot = PaperBot()
    BotRequestHandler.bot = bot

    server = ThreadingHTTPServer(("0.0.0.0", port), BotRequestHandler)
    logger.info(f"Auxo running at http://localhost:{port}")
    print(f"Auxo running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        bot.stop()
        logger.info("Auxo stopped")
        print("\nStopped.")

if __name__ == "__main__":
    main()
