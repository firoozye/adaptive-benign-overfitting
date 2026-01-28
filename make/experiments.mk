include make/common.mk

dd_test_MAIN     := $(EXP_DIR)/double_descent/dd_test_non_linear.cpp
EURUSD_test_MAIN := $(EXP_DIR)/EURUSD/test_EURUSD.cpp
elect_test_MAIN  := $(EXP_DIR)/electricity/test_elect.cpp
gridsearch_test_MAIN := $(EXP_DIR)/gridsearch/electricity/test_elect.cpp
gridsearch_test_best_MAIN := $(EXP_DIR)/gridsearch/electricity/test_elect_best_hyperpar.cpp
gridsearch_eurusd_test_MAIN := $(EXP_DIR)/gridsearch/EURUSD/test_eurusd.cpp
gridsearch_eurusd_test_best_MAIN := $(EXP_DIR)/gridsearch/EURUSD/test_eurusd_best_hyperpar.cpp
frontier_test_MAIN := $(EXP_DIR)/gridsearch/EURUSD/frontier_test_eurusd.cpp

dd_test_OBJS     := $(call make-objs,$(dd_test_MAIN))
EURUSD_test_OBJS := $(call make-objs,$(EURUSD_test_MAIN))
elect_test_OBJS  := $(call make-objs,$(elect_test_MAIN))
gridsearch_test_OBJS := $(call make-objs,$(gridsearch_test_MAIN))
gridsearch_test_best_OBJS := $(call make-objs,$(gridsearch_test_best_MAIN))
gridsearch_eurusd_test_OBJS := $(call make-objs,$(gridsearch_eurusd_test_MAIN))
gridsearch_eurusd_test_best_OBJS := $(call make-objs,$(gridsearch_eurusd_test_best_MAIN))
frontier_test_OBJS := $(call make-objs,$(frontier_test_MAIN))

# --- Real binaries go to bin/ -----------------------------------------------

$(BIN_DIR)/dd_test: $(dd_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/EURUSD_test: $(EURUSD_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/elect_test: $(elect_test_OBJS) libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_test: $(gridsearch_test_OBJS) libcore.a libcore_baseline.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_test_best: $(gridsearch_test_best_OBJS) libcore.a libcore_baseline.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_eurusd_test: $(gridsearch_eurusd_test_OBJS) libcore.a libcore_baseline.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/gridsearch_eurusd_test_best: $(gridsearch_eurusd_test_best_OBJS) libcore.a libcore_baseline.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/frontier_test: $(frontier_test_OBJS) libcore.a libcore_baseline.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

# --- Convenience aliases so "make dd_test" still works ----------------------

.PHONY: dd_test EURUSD_test elect_test gridsearch_test gridsearch_test_best gridsearch_eurusd_test gridsearch_eurusd_test_best frontier_test

dd_test: $(BIN_DIR)/dd_test
EURUSD_test: $(BIN_DIR)/EURUSD_test
elect_test: $(BIN_DIR)/elect_test
gridsearch_test: $(BIN_DIR)/gridsearch_test
gridsearch_test_best: $(BIN_DIR)/gridsearch_test_best
gridsearch_eurusd_test: $(BIN_DIR)/gridsearch_eurusd_test
gridsearch_eurusd_test_best: $(BIN_DIR)/gridsearch_eurusd_test_best
frontier_test: $(BIN_DIR)/frontier_test

# --- Umbrella list used by top-level Makefile --------------------------------

EXPERIMENT_PROGS := $(BIN_DIR)/dd_test $(BIN_DIR)/EURUSD_test $(BIN_DIR)/elect_test \
                    $(BIN_DIR)/gridsearch_test $(BIN_DIR)/gridsearch_test_best \
                    $(BIN_DIR)/gridsearch_eurusd_test $(BIN_DIR)/gridsearch_eurusd_test_best \
					$(BIN_DIR)/frontier_test

.PHONY: clean-experiments
clean-experiments:
	$(RM) $(EXPERIMENT_PROGS) $(dd_test_OBJS) $(EURUSD_test_OBJS) $(elect_test_OBJS) $(gridsearch_test_OBJS) \
	$(gridsearch_test_best_OBJS) $(gridsearch_eurusd_test_OBJS) $(gridsearch_eurusd_test_best_OBJS) $(frontier_test_OBJS)
