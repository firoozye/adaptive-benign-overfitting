
BASELINE_EURUSD_QRDRLS_SRCS := $(EXP_DIR)/baselines/QRD_RLS/EURUSD/qrd_rls_test.cpp
BASELINE_EURUSD_QRDRLS_OBJS := $(call make-objs,$(BASELINE_EURUSD_QRDRLS_SRCS))

BASELINE_ELECT_QRDRLS_SRCS := $(EXP_DIR)/baselines/QRD_RLS/electricity/qrd_rls_test.cpp
BASELINE_ELECT_QRDRLS_OBJS := $(call make-objs,$(BASELINE_ELECT_QRDRLS_SRCS))

BASELINE_EURUSD_KRLS_SRCS := $(EXP_DIR)/baselines/KRLS_RBF/EURUSD/krls_rbf_test.cpp
BASELINE_EURUSD_KRLS_OBJS := $(call make-objs,$(BASELINE_EURUSD_KRLS_SRCS))

BASELINE_ELECT_KRLS_SRCS := $(EXP_DIR)/baselines/KRLS_RBF/electricity/krls_rbf_test.cpp
BASELINE_ELECT_KRLS_OBJS := $(call make-objs,$(BASELINE_ELECT_KRLS_SRCS))


$(BIN_DIR)/baseline_qrd_rls_eurusd: $(BASELINE_EURUSD_QRDRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_qrd_rls_elect: $(BASELINE_ELECT_QRDRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_k_rls_eurusd: $(BASELINE_EURUSD_KRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_k_rls_elect: $(BASELINE_ELECT_KRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@


.PHONY: baseline_qrd_rls_eurusd baseline_qrd_rls_elect baseline_k_rls_eurusd baseline_k_rls_elect

baseline_qrd_rls_eurusd: $(BIN_DIR)/baseline_qrd_rls_eurusd
baseline_qrd_rls_elect:  $(BIN_DIR)/baseline_qrd_rls_elect
baseline_k_rls_eurusd:   $(BIN_DIR)/baseline_k_rls_eurusd
baseline_k_rls_elect:    $(BIN_DIR)/baseline_k_rls_elect


.PHONY: clean-baselines clean-libs
.PHONY: clean-baselines
clean-baselines:
	$(RM) $(BASELINE_PROGS) $(BASELINE_EURUSD_QRDRLS_OBJS) $(BASELINE_ELECT_QRDRLS_OBJS) \
	      $(BASELINE_EURUSD_KRLS_OBJS)  $(BASELINE_ELECT_KRLS_OBJS)

