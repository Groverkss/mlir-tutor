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

# Per-tutorial test targets using --filter
add_custom_target(check-tutorial-list
  COMMAND lit ${CMAKE_CURRENT_BINARY_DIR} --filter=list -v
  DEPENDS ${TUTORIAL_TEST_DEPENDS}
  COMMENT "Running list tutorial tests"
)
