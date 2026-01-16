# Core library (shared code)

CORE_SRCS := \
  src/abo/ABO.cpp \
  src/abo/QR_decomposition.cpp \
  src/pseudo_inverse.cpp \
  src/abo/last_row_givens.cpp \
  src/abo/gau_rff.cpp \
  src/add_row_col.cpp \
  src/read_csv_func.cpp

CORE_OBJS := $(call make-objs,$(CORE_SRCS))

CORE_SRCS_2 := \
  src/baselines/QRD_RLS/qrd_rls.cpp \
  src/baselines/KRLS_RBF/krls_rbf.cpp

CORE_OBJS_2 := $(call make-objs,$(CORE_SRCS_2))

libcore_baseline.a: $(CORE_OBJS_2)
	ar rcs $@ $^

libcore.a: $(CORE_OBJS)
	ar rcs $@ $^

.PHONY: clean-libs
clean-libs:
	$(RM) libcore.a $(CORE_OBJS) libcore_baseline.a $(CORE_OBJS)


