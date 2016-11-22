#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>// math.h = c standard lib, could've used #include <cmath> as well => c++ standard lib
#include <stdint.h>

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

        void loadArray(uint32_t *pointer){
            for(int i=0; i < 5; i++){
                pointer[i] = rand() % 1000;
            }
        }

        void loadVoidArr(void *arrayToLoad, uint32_t *data){
            int *tmp = (int*)arrayToLoad;
            for(int i = 0; i < 5; i++){
                tmp[i] = data[i];
                printf("\n %i", tmp[i]);
            }
            printf("\n");
        }
};

extern "C" {
    Test* Test_new(){ return new Test(); }
    void Test_hello(Test* t){ t->hello(); }
    void Test_cosine(Test* t, double param){ t->cosine(param); }
    void Test_load(Test* t, uint32_t *pointer ){ t->loadArray(pointer); }
    void Test_voidLoad(Test *t, void *arrayToLoad, uint32_t *data){ t->loadVoidArr(arrayToLoad, data); }
}
