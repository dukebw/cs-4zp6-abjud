"""This module tests the interfaces exposed by `tf_http_server`."""
import requests
import urllib

def run_tests():
    """Runs all TF HTTP server unit tests."""
    img_url = {'image_url': 'http://www.ccdf.ca/ccdf/NewCoach/english/newimages/Module%20D%20Grapics/D4%20Man%20waving.jpg'}
    response = requests.get('http://localhost:8765', img_url)
    print(response.json())

if __name__ == "__main__":
    run_tests()
