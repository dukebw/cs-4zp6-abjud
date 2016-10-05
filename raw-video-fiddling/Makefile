# NOTE(brendan): set P using "export P=program_name" from shell
P=read_frame

OBJECTS=
STD_FLAGS=c99
CFLAGS=-ggdb3 -std=$(STD_FLAGS) -Wall -Wextra -Werror -O0 -I/usr/local/include
LDLIBS=-L/usr/local/lib -lavformat -lavcodec -lswscale -lavutil -lswresample -lm -lz -pthread -ldl
CC=gcc
CXX=g++

all: $(P)

$(P):
