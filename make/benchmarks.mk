# Benchmarks (google benchmark flags only on benchmark objects)

BENCH_HEADER := libs/benchmark/include/benchmark/benchmark.h

ifneq ($(wildcard $(BENCH_HEADER)),)

BENCH_CPPFLAGS := -Ilibs/benchmark/include
BENCH_LDFLAGS  := -Llibs/benchmark/build/src
BENCH_LDLIBS   := -lbenchmark -lpthread

timing_test_MAIN := $(BENCH_DIR)/timing_test/timing_test_non_linear_ts.cpp
timing_test_OBJS := $(call make-objs,$(timing_test_MAIN))

# Apply include path to the actual benchmark objs (obj/timing_test/...)
$(timing_test_OBJS): CPPFLAGS += $(BENCH_CPPFLAGS)

timing_test: $(timing_test_OBJS) libcore.a
	$(CXX) $(LDFLAGS) $(BENCH_LDFLAGS) $^ $(LDLIBS) $(BENCH_LDLIBS) -o $@

BENCHMARK_PROGS := timing_test

else

BENCHMARK_PROGS :=

.PHONY: timing_test
timing_test:
	@echo "Skipping timing_test: Google Benchmark not found (libs/benchmark/)"

endif

.PHONY: clean-benchmarks
clean-benchmarks:
	$(RM) timing_test $(BASELINE_RLS_OBJS)