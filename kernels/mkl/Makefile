# Compiler
CC = g++

# MKL Root Directory (adjust if necessary)
MKLROOT = /home/elliotliu/miniconda

# Update LD_LIBRARY_PATH to include MKL libraries
export LD_LIBRARY_PATH := $(MKLROOT)/lib:$(LD_LIBRARY_PATH)

# Compiler Flags
CFLAGS = -O3 -Wall -pthread -std=c++17 -m64 -DMKL_ILP64 -DUSE_MKL -DMKL_ILP64 -mavx2 -mfma -ffast-math -DUSE_INT8_INT4_PRODUCT -DQM_x86

# Include Directories
INCLUDE_DIRS = -I.. -I../../llm/half-2.2.0/include -I$(MKLROOT)/include

# Library Flags
LDFLAGS = -Wl,--start-group \
          /home/elliotliu/miniconda/lib/libmkl_intel_ilp64.so \
          /home/elliotliu/miniconda/lib/libmkl_gnu_thread.so \
          /home/elliotliu/miniconda/lib/libmkl_core.so \
          -Wl,--end-group -lgomp -lpthread -lm -ldl

# Source Files
SRC = matmul_mkl_test.cc matmul_mkl_int8.cc
#  ../matmul_int8.cc

# Object Files
OBJ = $(SRC:.cc=.o)

# Target Executable
TARGET = matmul_mkl_test

all: $(TARGET)
	@echo "Running matmul_mkl_test..."
	./$(TARGET)

# Compile .cc files into .o files
%.o: %.cc
	$(CC) $(CFLAGS) $(INCLUDE_DIRS) -c $< -o $@

# Link Objects to Create Executable
$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $(TARGET) $(LDFLAGS)

# Clean Build Files
clean:
	rm -f $(OBJ) $(TARGET)
