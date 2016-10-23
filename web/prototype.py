import tornado.ioloop
import tornado.web

class SearchHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('textInput.html')
    def post(self):
        self.render('response.html', msg = self.get_body_argument("query"))

class TutorialHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('help.html')

class SkeltonOverlayHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('imgUpload.html')
    def post(self):
        self.render('uploadResult.html')


def main():

    settings = {
        "template_path": 'views/',
        "static_path": 'static/',
    }
    return tornado.web.Application([
        (r"/", SearchHandler),
        (r"/search", SearchHandler),
        (r"/help", TutorialHandler),
        (r"/skOverlay", SkeltonOverlayHandler)

    ], **settings)

if __name__ == "__main__":
    app = main()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()
