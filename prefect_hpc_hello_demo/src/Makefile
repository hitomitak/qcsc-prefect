CC ?= mpicc
CFLAGS ?= -O2 -Wall
TARGET := hello_mpi

all: $(TARGET)

$(TARGET): hello_mpi.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGET)
