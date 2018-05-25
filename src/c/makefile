CC = gcc -Wall -Wextra -DUSE_OPENBLAS -DDEBUG -g

OPT = -O2
PROFILE = -lprofiler

LIB_FLAGS = -ldl -lpthread -lm

UTILS = ./
TEST = ./
BIN=./build/


key.o: $(UTILS)key.c 
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

regionLinearKNN.o: $(UTILS)regionLinearKNN.c
	$(CC) $(CCFLAGS) -c $< -o $(BIN)$@

key_test.o: $(TEST)key_test.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

key_test: key_test.o key.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)key.o -o $@ $(LIB_FLAGS)

regionLinearKNN_test.o: $(TEST)regionLinearKNN_test.c
	$(CC) $(CXXFLAGS) -c $< -o $(BIN)$@

regionLinearKNN_test: regionLinearKNN_test.o regionLinearKNN.o key.o
	$(CC) $(CCFLAGS) $(BIN)$< $(BIN)regionLinearKNN.o $(BIN)key.o -o $@ $(LIB_FLAGS)