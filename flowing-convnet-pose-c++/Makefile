# Use "P=<program name> make"

INCLUDES=-I../caffe/include -I../caffe/build/include -I/usr/include/openblas
CXXFLAGS=-ggdb3 -Wall -Wextra -O3 -DCPU_ONLY=1 -DBLAS=open # -Werror
LDLIBS=-L../caffe/build/lib -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lboost_system -lglog
LDFLAGS=-Wl,-R../caffe/build/lib
CXX=g++
SOURCE=src/$(P).cpp
EXECUTABLE=bin/$(P)

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $< -o $@ $(LDLIBS) $(LDFLAGS)

clean:
	$(RM) $(EXECUTABLE)
