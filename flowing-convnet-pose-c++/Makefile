P=video_pose_overlay

INCLUDES=-I../caffe/include -I../caffe/build/include -I/usr/include/openblas -I./inc
CXXFLAGS=-ggdb3 -Wall -Wextra -O3 -DCPU_ONLY=1 -DBLAS=open -std=gnu++14 -fcolor-diagnostics $(INCLUDES) # -Werror
LDLIBS=-L../caffe/build/lib -lcaffe -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video -lboost_system -lglog
LDFLAGS=-Wl,-R../caffe/build/lib
CXX=clang++
SOURCE=src/$(P).cpp src/estimate_pose.cpp
OBJECTS=$(patsubst %.cpp,obj/%.o,$(notdir $(SOURCE)))
EXECUTABLE=bin/$(P)

all: $(EXECUTABLE)

obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJECTS) $(LDLIBS) 

depend: .depend

.depend: $(SOURCE)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJECTS)

dist-clean: clean
	$(RM) *~ .depend

include .depend
