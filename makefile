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
##LIBS = -L$(LIB_DIR)  -llapacke -llapack -lblas -lcblas -lgfortran -lm
LIBS = -L$(LIB_DIR) -llapacke -lopenblas -lgfortran -lm

# Include paths - both project and external
##INCLUDES = -I$(INC_DIR) I/usr/include -I$(EIGEN_PATH) -I$(LAPACK_PATH)
INCLUDES = -I$(INC_DIR) -I/usr/include -I/usr/include/eigen3 -I/usr/lib/lapack

CXXFLAGS = -std=c++17 $(INCLUDES) -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE  -DLOG_LEVEL=$(if $(LOG_LEVEL),$(LOG_LEVEL),3)

CXXFLAGS += -Wall 
CXXFLAGS += -Wno-shadow                # Suppress variable shadowing warnings
CXXFLAGS += -Wno-unused-parameter      # Suppress unused parameter warnings
CXXFLAGS += -Wno-sign-compare         # Suppress signed/unsigned comparison warnings
CXXFLAGS += -Wno-unused-variable      # Suppress unused variable warnings
CXXFLAGS += -Wno-reorder              # Suppress member initialization reorder warnings
CXXFLAGS += -Wno-comment              # Suppress comment warnings
CXXFLAGS += -Wno-deprecated-declarations # Suppress deprecated declaration warnings
# Add debug flags if DEBUG is set
ifdef DEBUG
    CXXFLAGS += $(DEBUGFLAGS)
endif

# Source files (with full paths)
#SOURCES = $(SRC_DIR)/main.cc \
#         $(SRC_DIR)/gau_rff.cpp \
#         $(SRC_DIR)/QR_RLS.cpp \
#         $(SRC_DIR)/read_csv_func.cpp
# Source files (with full paths)
SOURCES = $(SRC_DIR)/main.cc \
          $(SRC_DIR)/QR_RLS.cpp \
          $(SRC_DIR)/QR_decomposition.cpp \
          $(SRC_DIR)/pseudo_inverse.cpp \
          $(SRC_DIR)/last_row_givens.cpp \
          $(SRC_DIR)/add_row_col.cpp \
          $(SRC_DIR)/gau_rff.cpp \
          $(SRC_DIR)/read_csv_func.cpp

# Headers (now in include directory)
HEADERS = $(INC_DIR)/QR_RLS.h \
          $(INC_DIR)/gau_rff.h

# Generate object file names in obj directory
OBJECTS = $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES)))

# Target executable in root directory
TARGET = $(ROOT_DIR)/abo_predict

# Ensure build directories exist
$(shell mkdir -p $(OBJ_DIR))

# Build rules
all: $(TARGET)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	@echo "Linking object files..."
	@echo "CXXFLAGS: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET) $(LIBS)
	@echo "Build successful!"

# Compile .cc files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc $(HEADERS)
	@echo "Compiling CC $<..."
	@echo "INCLUDES: $(INCLUDES)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "Compiled $< successfully!"

# Compile .cpp files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	@echo "Compiling CPP $<..."
	@echo "INCLUDES: $(INCLUDES)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo "Compiled $< successfully!"

# Clean rule
clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

# Print variables for debugging
print-%:
	@echo $* = $($*)

# Debug build with symbols and verbose logging
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: LOG_LEVEL = 4
debug: all
	@echo "Debug build complete with LOG_LEVEL=4 and debug symbols enabled"
	@echo "Run with: gdb $(TARGET)"

.PHONY: all clean print-% debug
