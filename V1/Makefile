MPICC=mpicc
CFLAGS=-O3

default: all

all: main

main:
	$(MPICC) $(CFLAGS) -o main main.c -lopenblas -lm

test:
	mpiexec -n 4 ./main ../Datasets/Container_Crane_Controller_Data_Set.csv

.PHONY: clean

clean:
	-rm -f main
