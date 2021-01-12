CC=gcc
MPICC=mpicc
CFLAGS=-O3

default: all

all: mainSerial main

main:
	$(MPICC) $(CFLAGS) -o main main.c -lm 

mainSerial:
	$(CC) $(CFLAGS) -o mainSerial mainSerial.c -lm 

testSerial:
	./mainSerial ../Datasets/BreastCancerCoimbra.csv 
test:
	mpiexec -n 4 ./main ../Datasets/BreastCancerCoimbra.csv

.PHONY: clean

clean:
	-rm -f mainSerial
	-rm -f main
