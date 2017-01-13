from ctypes import *

#loads the library and stores it into lib, this path is hardcoded as of now
lib = cdll.LoadLibrary('./libtest.so')

class Test(object):
    def __init__(self):
        self.obj = lib.Test_new()#calling the functions defined in the 'extern C part of the cpp file'

    def hello(self):
        lib.Test_hello(self.obj)#note the names, i think they don't follow any strict naming rules

    def cosine(self, param):
        lib.Test_cosine(self.obj, param)

    def LoadRandomVals(self, pointer, arraySize):
        lib.Test_load(self.obj, pointer, arraySize)

    def loadVoidArr(self, arrayToLoad, pointer, arraySize):
        lib.Test_voidLoad(self.obj, arrayToLoad, pointer, arraySize)

def main():

    simpleTest = Test()
    simpleTest.hello()

    #need to cast the input string as a float (raw_input returns a string representation)
    #*******IMPORTANT************
    #NEED to cast the FLOAT as a 'c_double' type before passing it into the C function, which strictly expects a double

    #all of your python vars have to be wrapped before sending them to a C function or you will get a TypeError
    #CTYPES TYPE WRAPPERS (copy/pasted from ctypes documenation):

        #c_bool     c_char      c_wchar   c_byte   c_short   c_int   c_long
        #c_float    c_double    c_char_p (null terminated string) ...

    degrees = c_double(float(raw_input('Please enter an angle (no error checks in place):\n')))
    simpleTest.cosine(degrees)


    #simple 1D int array --> let the user decide what the size is
    arraySize = int(raw_input('\nPlease enter an array size to be created and loaded (no error checks):\n'))


    pointerToArr = (c_uint*arraySize)()
    print '\n'
    print pointerToArr
    #this cast function is from the ctypes library, and can allow the variable to represent a C pointer
    cast(pointerToArr, POINTER(c_uint*arraySize))

    print '\nPre-Randomized string repr\n'
    string = ''
    for arrayElement in pointerToArr[0:arraySize-1]:
        string +=  "{0} ".format(arrayElement)
    print string

    #note that we must cast arraySize as a ctype type before sending it to the shared object file
    simpleTest.LoadRandomVals(pointerToArr, c_uint(arraySize))

    print '\nPost-Randomized string repr\n'
    string = ''
    for arrayElement in pointerToArr[0:arraySize-1]:
        string += "{0} ".format(arrayElement)
    print string + '\n'

    #loading the values of the randomized array above into a void pointer
    voidPointer = (c_void_p*arraySize)()
    print voidPointer
    cast(voidPointer, POINTER(c_void_p*arraySize))

    simpleTest.loadVoidArr(voidPointer, pointerToArr, c_uint(arraySize))



if __name__ == "__main__":
    main()
