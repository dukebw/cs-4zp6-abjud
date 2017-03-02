"""This module implements the TensorFlow HTTP server, which is a standalone
server that is able to handle HTTP requests to do TensorFlow operations.
"""
import http.server

class TFHttpRequestHandler(http.server.BaseHTTPRequestHandler):
    """Defines handlers for specific HTTP request codes."""
    def do_GET(self):
        print('GET!')

    def do_POST(self):
        print('POST!')


def run():
    """Starts a server that will handle  HTTP requests to use TensorFlow."""
    server_address = ('localhost', 8765)
    httpd = http.server.HTTPServer(server_address, TFHttpRequestHandler)
    httpd.serve_forever()


if __name__ == "__main__":
    run()
