# --- Toolchain ---------------------------------------------------------------
CXX       ?= g++
CPPFLAGS  ?= -Iinclude -MMD -MP

INCLUDES = -I$(INC_DIR) -I/usr/include -I/usr/include/eigen3 -I/usr/lib/lapack
CXXFLAGS  = -std=c++17 $(INCLUDES) -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE \
			-Wall -Wno-shadow \
			-Wno-unused-parameter -Wno-sign-compare -Wno-unused-variable \
			-Wno-reorder -Wno-comment -Wno-deprecated-declarations

LDFLAGS   ?= -Llibs/lib
LDLIBS    ?= -lopenblas -llapacke -lgfortran -lm
DEBUGFLAGS = -g -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
ifdef DEBUG
  CXXFLAGS += $(DEBUGFLAGS) -DLOG_LEVEL=4
else
  CXXFLAGS += -DLOG_LEVEL=3
endif

# --- Layout ------------------------------------------------------------------
SRC_DIR := src
OBJ_DIR := obj
EXP_DIR := experiments
BENCH_DIR := benchmarks
BIN_DIR := bin

# --- Object rule helper ------------------------------------------------------
make-objs = \
  $(patsubst %.cpp,%.o, \
    $(patsubst %.cc,%.o, \
      $(patsubst $(SRC_DIR)/%,$(OBJ_DIR)/%, \
        $(patsubst $(EXP_DIR)/%,$(OBJ_DIR)/%, \
          $(patsubst $(BENCH_DIR)/%,$(OBJ_DIR)/%,$(1)) \
        ) \
      ) \
    ) \
  )

# --- Compile rules -----------------------------------------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(EXP_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
	
$(OBJ_DIR)/%.o: $(BENCH_DIR)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR):
	@mkdir -p $@
