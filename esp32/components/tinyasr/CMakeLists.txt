idf_component_register(SRCS "tinyasr.c" "kernels.c" "tinyasr_lm.c" "tasr_nemo.c" "tasr_seg.c"
                       INCLUDE_DIRS "include" "."
                       PRIV_INCLUDE_DIRS ".")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -ffp-contract=off -Wno-unused-function)

if(DEFINED ENV{TASR_DEBUG_SUMS})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TASR_DEBUG_SUMS)
endif()
if(DEFINED ENV{TASR_DEBUG_DUMP})
    target_compile_definitions(${COMPONENT_LIB} PRIVATE TASR_DEBUG_DUMP)
endif()
