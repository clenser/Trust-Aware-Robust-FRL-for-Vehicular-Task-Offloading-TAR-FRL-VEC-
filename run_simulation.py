"""
Launcher for the Trust-Aware Federated Reinforcement Learning VANET Simulator.
Starts the FastAPI Web and WebSocket server on http://localhost:8000
"""

import uvicorn
import webbrowser
import os

if __name__ == "__main__":
    print("=" * 70)
    print(" VANNET: TRUST-AWARE FEDERATED RL FOR VEHICULAR EDGE OFFLOADING")
    print(" Simulation console and research results dashboard")
    print("=" * 70)
    print(" Starting Web Server on http://localhost:8000 ...")
    print(" Open your web browser to view the console.")
    print("=" * 70)
    
    uvicorn.run("web.app:app", host="127.0.0.1", port=8000, reload=False, log_level="info")
