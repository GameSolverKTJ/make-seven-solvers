CC = gcc
CFLAGS = -O3 -std=c2x -Wall -Wextra -fopenmp
TARGET = Make7DFPNv9_8_3

.PHONY: all release test clean
all: $(TARGET)
release: $(TARGET)

$(TARGET): dfpn_solver_v9_8_3.c
	$(CC) $(CFLAGS) $< -o $@

test: $(TARGET)
	./$(TARGET) --self-test

clean:
	rm -f $(TARGET)
