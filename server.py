#!/usr/bin/env python3
"""
Simple server to serve the static frontend files and run the API
"""

import os
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from flask import Flask, send_from_directory

# Import the API
from api import app as api_app

class CustomHTTPRequestHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve index.html for all routes"""
    
    def do_GET(self):
        # If the path is /api/*, let Flask handle it
        if self.path.startswith('/api/'):
            return super().do_GET()
        
        # For all other paths, serve index.html (SPA behavior)
        if self.path == '/' or not os.path.exists(self.path[1:]):
            self.path = '/index.html'
        
        return super().do_GET()

def run_static_server():
    """Run the static file server"""
    port = 8000
    server = HTTPServer(('localhost', port), CustomHTTPRequestHandler)
    print(f"Static server running at http://localhost:{port}")
    server.serve_forever()

def run_api_server():
    """Run the Flask API server"""
    print("Starting API server...")
    api_app.run(host='localhost', port=5000, debug=False, use_reloader=False)

def main():
    """Main function to start both servers"""
    print("Starting 4th Down Decision Tool...")
    print("=" * 50)
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Wait a moment for API to start
    time.sleep(2)
    
    # Start static server
    print("Opening browser...")
    webbrowser.open('http://localhost:8000')
    
    try:
        run_static_server()
    except KeyboardInterrupt:
        print("\nShutting down servers...")

if __name__ == '__main__':
    main()
