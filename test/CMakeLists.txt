configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_SOURCE_DIR}/lit.cfg.py
)

set(TUTORIAL_TEST_DEPENDS
  tutorial-opt
  FileCheck
)

# All tests - run lit on build directory
add_lit_testsuite(check-tutorial "Running all tutorial tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
)

# Per-chapter test targets using --filter
add_custom_target(check-tutorial-ch1
  COMMAND lit ${CMAKE_CURRENT_BINARY_DIR} --filter=ch1-cpu-vector-dsl -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
  COMMENT "Running ch1 tests"
)

add_custom_target(check-tutorial-ch2
  COMMAND lit ${CMAKE_CURRENT_BINARY_DIR} --filter=ch2-cpu-vector-dsl-loops -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
  COMMENT "Running ch2 tests"
)

add_custom_target(check-tutorial-ch3
  COMMAND lit ${CMAKE_CURRENT_BINARY_DIR} --filter=ch3-gpu-tile-dsl -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
  COMMENT "Running ch3 tests"
)

add_custom_target(check-tutorial-ch4
  COMMAND lit ${CMAKE_CURRENT_BINARY_DIR} --filter=ch4-linalg-transforms -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
  COMMENT "Running ch4 tests"
)
