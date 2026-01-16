# Top-level Makefile

include make/common.mk
include make/libs.mk
include make/baselines.mk
include make/experiments.mk
include make/benchmarks.mk

.PHONY: all experiments benchmarks libs baselines clean debug


all: experiments benchmarks baselines

# Convenience umbrella targets (also nice for `make experiments`)
experiments: $(EXPERIMENT_PROGS)

benchmarks: $(BENCHMARK_PROGS)

libs: libcore.a
baselines: $(BASELINE_PROGS)

debug:
	$(MAKE) DEBUG=1 all
	

clean: clean-experiments clean-benchmarks clean-libs clean-baselines
	$(RM) -r $(OBJ_DIR)
