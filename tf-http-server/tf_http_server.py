"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
import http.server

class TFHttpRequestHandler(http.server.BaseHTTPRequestHandler):
    """Defines handlers for specific HTTP request codes."""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write('{"response": "Hello!"}'.encode('utf8'))


def run():
    """Starts a server that will handle  HTTP requests to use TensorFlow."""
    server_address = ('localhost', 8765)
    httpd = http.server.HTTPServer(server_address, TFHttpRequestHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    run()
