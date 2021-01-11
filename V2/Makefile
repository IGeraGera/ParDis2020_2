CC=gcc
MPICC=mpicc
CFLAGS=-O3

default: all

all: mainSerial main

main:
	$(MPICC) $(CFLAGS) -o main main.c -lopenblas -lm 

mainSerial:
	$(CC) $(CFLAGS) -o mainSerial mainSerial.c -lopenblas -lm 

testSerial:
	./mainSerial ../Datasets/Container_Crane_Controller_Data_Set.csv 
test:
	mpiexec -n 4 ./main ../Datasets/Container_Crane_Controller_Data_Set.csv 

.PHONY: clean

clean:
	-rm -f mainSerial
	-rm -f main
