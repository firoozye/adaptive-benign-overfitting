
BASELINE_EURUSD_QRDRLS_SRCS := $(EXP_DIR)/baselines/QRD_RLS/EURUSD/qrd_rls_test.cpp
BASELINE_EURUSD_QRDRLS_OBJS := $(call make-objs,$(BASELINE_EURUSD_QRDRLS_SRCS))

BASELINE_ELECT_QRDRLS_SRCS := $(EXP_DIR)/baselines/QRD_RLS/electricity/qrd_rls_test.cpp
BASELINE_ELECT_QRDRLS_OBJS := $(call make-objs,$(BASELINE_ELECT_QRDRLS_SRCS))

BASELINE_EURUSD_KRLS_SRCS := $(EXP_DIR)/baselines/KRLS_RBF/EURUSD/krls_rbf_test.cpp
BASELINE_EURUSD_KRLS_OBJS := $(call make-objs,$(BASELINE_EURUSD_KRLS_SRCS))

BASELINE_ELECT_KRLS_SRCS := $(EXP_DIR)/baselines/KRLS_RBF/electricity/krls_rbf_test.cpp
BASELINE_ELECT_KRLS_OBJS := $(call make-objs,$(BASELINE_ELECT_KRLS_SRCS))

BASELINE_EURUSD_SWKRLS_SRCS := $(EXP_DIR)/baselines/SWKRLS/EURUSD/swkrls_test.cpp
BASELINE_EURUSD_SWKRLS_OBJS := $(call make-objs,$(BASELINE_EURUSD_SWKRLS_SRCS))

BASELINE_ELECT_SWKRLS_SRCS := $(EXP_DIR)/baselines/SWKRLS/electricity/swkrls_test.cpp
BASELINE_ELECT_SWKRLS_OBJS := $(call make-objs,$(BASELINE_ELECT_SWKRLS_SRCS))


BASELINE_PROGS := $(BIN_DIR)/baseline_qrd_rls_eurusd $(BIN_DIR)/baseline_qrd_rls_elect \
                  $(BIN_DIR)/baseline_k_rls_eurusd  $(BIN_DIR)/baseline_k_rls_elect \
                  $(BIN_DIR)/swkrls_eurusd          $(BIN_DIR)/swkrls_elect

$(BIN_DIR)/baseline_qrd_rls_eurusd: $(BASELINE_EURUSD_QRDRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_qrd_rls_elect: $(BASELINE_ELECT_QRDRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_k_rls_eurusd: $(BASELINE_EURUSD_KRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/baseline_k_rls_elect: $(BASELINE_ELECT_KRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/swkrls_eurusd: $(BASELINE_EURUSD_SWKRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BIN_DIR)/swkrls_elect: $(BASELINE_ELECT_SWKRLS_OBJS) libcore_baseline.a libcore.a | $(BIN_DIR)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@


.PHONY: baseline_qrd_rls_eurusd baseline_qrd_rls_elect baseline_k_rls_eurusd baseline_k_rls_elect
.PHONY: swkrls_eurusd swkrls_elect

baseline_qrd_rls_eurusd: $(BIN_DIR)/baseline_qrd_rls_eurusd
baseline_qrd_rls_elect:  $(BIN_DIR)/baseline_qrd_rls_elect
baseline_k_rls_eurusd:   $(BIN_DIR)/baseline_k_rls_eurusd
baseline_k_rls_elect:    $(BIN_DIR)/baseline_k_rls_elect
swkrls_eurusd:           $(BIN_DIR)/swkrls_eurusd
swkrls_elect:            $(BIN_DIR)/swkrls_elect


.PHONY: clean-baselines
clean-baselines:
	$(RM) $(BASELINE_PROGS) $(BASELINE_EURUSD_QRDRLS_OBJS) $(BASELINE_ELECT_QRDRLS_OBJS) \
	      $(BASELINE_EURUSD_KRLS_OBJS)  $(BASELINE_ELECT_KRLS_OBJS) \
	      $(BASELINE_EURUSD_SWKRLS_OBJS) $(BASELINE_ELECT_SWKRLS_OBJS)
