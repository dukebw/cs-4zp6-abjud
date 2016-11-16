from ctypes import cdll

#loads the library and stores it into lib, this path is hardcoded as of now
lib = cdll.LoadLibrary('./libtest.so')

class Test(object):
    def __init__(self):
        self.obj = lib.Test_new()#calling the functions defined in the 'extern C part of the cpp file'

    def hello(self):
        lib.Test_hello(self.obj)#note the names, i think they don't follow any strict naming rules

    def cosine(self):
        lib.Test_cosine(self.obj)

def main():

    simpleTest = Test()
    simpleTest.hello()

    degrees = input('Please enter an angle (no error checks in place)')
    simpleTest.cosine()


if __name__ == "__main__":
    main()
