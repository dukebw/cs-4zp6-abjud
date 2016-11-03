import tornado.ioloop
import tornado.web
import os, uuid

__UPLOADS__ = 'uploads/'

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
        fileInfo = self.request.files['image'][0]
        #you can also print the binary representation of the image
        #print fileInfo
        fileName = fileInfo['filename']
        extn = os.path.splitext(fileName)[1]
        #uuid just produces a unique id string + .<ext> (ex. png, jpg, mp4)
        uniqueName = str(uuid.uuid4()) + extn
        #writes the file to the 'uploads/' directory
        fileHandler = open(__UPLOADS__ + uniqueName, 'w')
        fileHandler.write(fileInfo['body'])
        self.finish(uniqueName + ' is uploaded')


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
