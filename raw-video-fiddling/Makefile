# NOTE(brendan): set P using "export P=program_name" from shell
P=read_frame

OBJECTS=
STD_FLAGS=c99
CFLAGS=-ggdb3 -std=$(STD_FLAGS) -Wall -Wextra -Werror -O0 -I/usr/include/ffmpeg
LDLIBS=-lavutil -lavformat -lavcodec -lz -lavutil -lm -lswscale
CC=gcc
CXX=g++

all: $(P)

$(P):
