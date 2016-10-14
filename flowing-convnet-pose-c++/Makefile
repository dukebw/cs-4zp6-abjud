P=estimate_pose

INCLUDES=-I../caffe/include -I../caffe/build/include -I/usr/include/openblas
CXXFLAGS=-ggdb3 -Wall -Wextra -Werror -O0 -DCPU_ONLY=1 -DBLAS=open
LDLIBS=-L../caffe/build/lib -lcaffe -lopencv_core -lopencv_highgui -lboost_system
LDFLAGS=-Wl,-R../caffe/build/lib
CXX=g++
SOURCE=src/$(P).cpp
EXECUTABLE=bin/$(P)

all: $(EXECUTABLE)

$(EXECUTABLE): $(SOURCE)
	$(CXX) $(INCLUDES) $(CXXFLAGS) $< -o $@ $(LDLIBS) $(LDFLAGS)

clean:
	$(RM) $(EXECUTABLE)
