Background info:
    =>ctypes can export with/to cdll, windll, oell


Test to check the ctypes wrapper - Simple Program

Given a simple C++ class/file => test.cpp:

    Since ctypes can only talk to C functions, we need to provide function definitions for C using extern "C" in the cpp file

    NOTE: a constructor function also needs to be defined for C even though one is not defined in the c++ class

just type the following while in this current directory
    >make

The test.cpp file will be compiled to make an object (.o) file and a shared object (.so) file
The makefile does this using g++ and some -options --> see makefile for details


the shared object will be used by the python code

In the python file => wrapper.py

    you need to import ctypes:
    >>from ctypes import cdll

    then ctypes can load a shared object as a library:
    >>lib = cdll.LoadLibrary('<path to shared object file>')

    after this, you can call the functions that you defined with extern "C" in the cpp file. For the test file, I created a class and a main function but all you'd need to do to access the function is:
    >>lib.functionName(params)
