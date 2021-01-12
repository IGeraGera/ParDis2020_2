CC=gcc
CFLAGS=-O3

default: all

all: main

main:
	$(CC) $(CFLAGS) -o main main.c -lopenblas -lm 

test:
	./main ../Datasets/BreastCancerCoimbra.csv 

.PHONY: clean

clean:
	-rm -f main
