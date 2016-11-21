#include <iostream>
#include <stdio.h>
#include <math.h>// math.h = c standard lib, could've used #include <cmath> as well => c++ standard lib

#define PI 3.14159265

//both std and printf methods work with the wrapper (iostream and stdio)
class Test{
    public:
        void hello(){
            std::cout << "Hello" << std::endl;
        }

        void cosine(double param){
            double res;
            res = cos (param * PI/180.0);
            printf("\nThe cosine of %f degrees is %f.\n", param, res);
        }
};

extern "C" {
    Test* Test_new(){ return new Test(); }
    void Test_hello(Test* t){ t->hello(); }
    void Test_cosine(Test* t, double param){ t->cosine(param); }
}
