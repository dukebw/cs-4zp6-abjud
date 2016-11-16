#simple makefile
CC=g++
SRCFILE=test
LIBFILE=libtest


all:
	$(CC) -c -fPIC $(SRCFILE).cpp -o $(SRCFILE).o
	$(CC) -shared -Wl,-soname,$(LIBFILE).so -o $(LIBFILE).so $(SRCFILE).o

clean:
	rm $(SRCFILE).o $(LIBFILE).so
