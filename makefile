# --- Toolchain ---------------------------------------------------------------
CXX       ?= g++
CPPFLAGS  ?= -Iinclude -MMD -MP                          # preprocessor flags (includes + dep gen)
#CPPFLAGS  ?= -Iinclude -Ilibs/eigen -MMD -MP
INCLUDES = -I$(INC_DIR) -I/usr/include -I/usr/include/eigen3 -I/usr/lib/lapack
CXXFLAGS = -std=c++17 $(INCLUDES) -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_STRUCTURE \
			-Wall -Wno-shadow \
			-Wno-unused-parameter -Wno-sign-compare -Wno-unused-variable \
			-Wno-reorder -Wno-comment -Wno-deprecated-declarations
LDFLAGS   ?= -Llibs/lib
LDLIBS    ?= -lopenblas -llapacke -lgfortran -lm
DEBUGFLAGS = -g -DEIGEN_INITIALIZE_MATRICES_BY_ZERO
# enable with: `make DEBUG=1`
ifdef DEBUG
  CXXFLAGS += $(DEBUGFLAGS) -DLOG_LEVEL=4
else
  CXXFLAGS += -DLOG_LEVEL=3
endif

# --- Google Benchmark (submodule) -------------------------------------------
BENCH_DIR     := libs/benchmark
BENCH_INC     := $(BENCH_DIR)/include
BENCH_LIBDIR  := $(BENCH_DIR)/build/src
BENCH_AR      := $(BENCH_LIBDIR)/libbenchmark.a

# Benchmark-only flags (kept out of normal targets)
BENCH_CPPFLAGS := -I$(BENCH_INC)
BENCH_LDFLAGS  := -L$(BENCH_LIBDIR)
BENCH_LDLIBS   := -lbenchmark -lpthread

# Auto-build libbenchmark.a if missing

$(BENCH_AR):
	@cmake -S $(BENCH_DIR) -B $(BENCH_DIR)/build \
		-DCMAKE_BUILD_TYPE=Release \
		-DBENCHMARK_ENABLE_TESTING=OFF \
		-DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON \
		-DCMAKE_CXX_COMPILER=$(CXX)
	@cmake --build $(BENCH_DIR)/build -j


# --- Layout ------------------------------------------------------------------
SRC_DIR := src
OBJ_DIR := obj

# Common sources shared by all executables
COMMON_SRCS := \
  $(SRC_DIR)/QR_RLS.cpp \
  $(SRC_DIR)/QR_decomposition.cpp \
  $(SRC_DIR)/pseudo_inverse.cpp \
  $(SRC_DIR)/last_row_givens.cpp \
  $(SRC_DIR)/add_row_col.cpp \
  $(SRC_DIR)/gau_rff.cpp \
  $(SRC_DIR)/read_csv_func.cpp

# Per-target main files
dd_test_MAIN     := $(SRC_DIR)/double_descent_test/dd_test_non_linear.cc
timing_test_MAIN := $(SRC_DIR)/timing_test/timing_test_non_linear_ts.cc
real_cond_num_test_MAIN := $(SRC_DIR)/stability_plots/real_cond_num.cc

# Targets
PROGS := dd_test timing_test real_cond_num_test

# Helper: turn a list of .cc/.cpp under SRC_DIR into objs under OBJ_DIR (mirrors tree)
make-objs = $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o, \
            $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(1)))

# Define per-target sources/objects

dd_test_SRCS     := $(dd_test_MAIN)     $(COMMON_SRCS)
timing_test_SRCS := $(timing_test_MAIN) $(COMMON_SRCS)
real_cond_num_test_SRCS := $(real_cond_num_test_MAIN) $(COMMON_SRCS)

dd_test_OBJS     := $(call make-objs,$(dd_test_SRCS))
timing_test_OBJS := $(call make-objs,$(timing_test_SRCS))
real_cond_num_test_OBJS := $(call make-objs,$(real_cond_num_test_SRCS))

$(timing_test_OBJS): CPPFLAGS += $(BENCH_CPPFLAGS)

# --- Default -----------------------------------------------------------------
.PHONY: all
all: $(PROGS)

# --- Link rules --------------------------------------------------------------

dd_test: $(dd_test_OBJS)
	@echo "Linking $@…"
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Build successful: $@"
	
timing_test: $(BENCH_AR) $(timing_test_OBJS)
	@echo "Linking $@ as benchmark…"
	$(CXX) $(LDFLAGS) $(BENCH_LDFLAGS) $(timing_test_OBJS) $(LDLIBS) $(BENCH_LDLIBS) -o $@
	@echo "Build successful: $@"

real_cond_num_test: $(real_cond_num_test_OBJS)
	@echo "Linking $@…"
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@
	@echo "Build successful: $@"

# --- Compile rules (auto-mkdir + deps) --------------------------------------
# .cpp
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling CPP $<…"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully!"

# .cc
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	@mkdir -p $(@D)
	@echo "Compiling CC  $<…"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled $< successfully!"

# --- Utilities ---------------------------------------------------------------
.PHONY: clean debug
clean:
	$(RM) -r $(OBJ_DIR) $(PROGS)

.PHONY: debug
debug:
	$(MAKE) clean
	$(MAKE) all DEBUG=1

# Include auto-generated dependency files
DEPS := $(dd_test_OBJS:.o=.d) $(timing_test_OBJS:.o=.d)
-include $(DEPS)


