# Compiler settings
CXX = g++
DEBUGFLAGS = -g -DEIGEN_INITIALIZE_MATRICES_BY_ZERO

# Directory paths
ROOT_DIR := .
SRC_DIR := $(ROOT_DIR)/src
OBJ_DIR := $(ROOT_DIR)/obj
INC_DIR := $(ROOT_DIR)/include
DATA_DIR := $(ROOT_DIR)/data

# External library paths
EIGEN_PATH = /usr/include/eigen3
LAPACK_PATH = /usr/lib/lapack
LIB_DIR = /usr/lib

# Libraries
LIBS = -L$(LIB_DIR) -llapacke -lopenblas -lgfortran -lm

# Include paths
INCLUDES = -I$(INC_DIR) -I/usr/include -I/usr/include/eigen3 -I/usr/lib/lapack

CXXFLAGS = -std=c++17 $(INCLUDES) -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE -DLOG_LEVEL=$(if $(LOG_LEVEL),$(LOG_LEVEL),3)

CXXFLAGS += -Wall 
CXXFLAGS += -Wno-shadow -Wno-unused-parameter -Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-comment -Wno-deprecated-declarations

ifdef DEBUG
    CXXFLAGS += $(DEBUGFLAGS)
endif

# Define executables
EXECUTABLES = abo_predict dd_test

# Define sources for each executable
abo_predict_SOURCES = $(SRC_DIR)/main.cc $(SRC_DIR)/QR_RLS.cpp $(SRC_DIR)/QR_decomposition.cpp \
                      $(SRC_DIR)/pseudo_inverse.cpp $(SRC_DIR)/last_row_givens.cpp \
                      $(SRC_DIR)/add_row_col.cpp $(SRC_DIR)/gau_rff.cpp $(SRC_DIR)/read_csv_func.cpp

dd_test_SOURCES = $(SRC_DIR)/double_descent_test/dd_test_lags_ewm.cc $(SRC_DIR)/QR_RLS.cpp $(SRC_DIR)/QR_decomposition.cpp \
                      $(SRC_DIR)/pseudo_inverse.cpp $(SRC_DIR)/last_row_givens.cpp \
                      $(SRC_DIR)/add_row_col.cpp $(SRC_DIR)/gau_rff.cpp $(SRC_DIR)/read_csv_func.cpp


# Generate object files
abo_predict_OBJECTS = $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(abo_predict_SOURCES)))
dd_test_OBJECTS = $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(dd_test_SOURCES)))

# Ensure build directories exist
#$(shell mkdir -p $(OBJ_DIR))
$(shell mkdir -p $(OBJ_DIR) $(OBJ_DIR)/double_descent_test)

# Default target: Build all executables
all: $(EXECUTABLES)

# Build each executable
abo_predict: $(abo_predict_OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $(abo_predict_OBJECTS) -o $@ $(LIBS)
	@echo "Build successful: $@"

dd_test: $(dd_test_OBJECTS)
	@echo "Linking $@..."
	$(CXX) $(CXXFLAGS) $(dd_test_OBJECTS) -o $@ $(LIBS)
	@echo "Build successful: $@"

# Compile .cc files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@echo "Compiling CC $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully!"

$(OBJ_DIR)/double_descent_test/%.o: $(SRC_DIR)/double_descent_test/%.cc
	@echo "Compiling CC $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully!"


# Compile .cpp files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling CPP $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully!"

# Clean rule
clean:
	rm -f $(OBJ_DIR)/*.o $(EXECUTABLES)

# Debug build
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: LOG_LEVEL = 4
debug: all
#debug: dd_test
	@echo "Debug build complete"
