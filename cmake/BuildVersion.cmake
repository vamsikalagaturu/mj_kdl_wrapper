# Writes OUTPUT, the definition of mj_kdl::version(), at every build: VERSION for a build of the
# vVERSION tag or outside a git checkout (an sdist), else VERSION-dev+<short hash>, with .dirty
# when tracked files differ from it. The file is rewritten only when the string changes.
#   cmake -DSOURCE_DIR=<repo> -DVERSION=<X.Y.Z> -DOUTPUT=<file.cpp> -P BuildVersion.cmake

set(_version "${VERSION}")
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${SOURCE_DIR}/.git")
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --exact-match --match "v${VERSION}" HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _not_release OUTPUT_QUIET ERROR_QUIET
  )
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse --short HEAD
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE _no_hash OUTPUT_VARIABLE _hash
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
  )
  if(_not_release AND NOT _no_hash)
    set(_version "${VERSION}-dev+${_hash}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
      WORKING_DIRECTORY "${SOURCE_DIR}"
      OUTPUT_VARIABLE _changes ERROR_QUIET
    )
    if(NOT _changes STREQUAL "")
      string(APPEND _version ".dirty")
    endif()
  endif()
endif()

set(_source "#include \"mj_kdl_wrapper/mj_kdl_wrapper.hpp\"

const char *mj_kdl::version() { return \"${_version}\"; }
")
set(_old "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" _old)
endif()
if(NOT _old STREQUAL _source)
  file(WRITE "${OUTPUT}" "${_source}")
endif()
