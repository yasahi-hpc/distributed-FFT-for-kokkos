## FindNVSHMEM.cmake - Locate NVSHMEM library and headers
# This module provides the following variables:
#  NVSHMEM_FOUND        - True if NVSHMEM was found
#  NVSHMEM_LIBRARIES    - Libraries to link against for NVSHMEM
#  NVSHMEM_INCLUDE_DIRS - Include directories for NVSHMEM
#  NVSHMEM_VERSION      - NVSHMEM version (if detected)
#
# Usage:
#   find_package(NVSHMEM REQUIRED)
#   target_include_directories(MyTarget PUBLIC ${NVSHMEM_INCLUDE_DIRS})
#   target_link_libraries(MyTarget PUBLIC ${NVSHMEM_LIBRARIES})

if(NOT NVSHMEM_FIND_VERSION)
  set(NVSHMEM_FIND_VERSION "")
endif()

# Allow override through environment or CMake variable
set(_nvshmem_root "$ENV{NVSHMEM_ROOT}")
if(NOT _nvshmem_root AND DEFINED NVSHMEM_ROOT)
  set(_nvshmem_root "${NVSHMEM_ROOT}")
endif()

set(_nvshmem_include_hints
  ${_nvshmem_root}/include
  /usr/local/include
  /usr/include
)

set(_nvshmem_lib_hints
  ${_nvshmem_root}/lib
  ${_nvshmem_root}/lib64
  /usr/local/lib
  /usr/local/lib64
  /usr/lib
  /usr/lib64
)

set(_nvshmem_config_hints)
if(_nvshmem_root)
  list(APPEND _nvshmem_config_hints
    ${_nvshmem_root}
    ${_nvshmem_root}/lib/cmake/nvshmem
    ${_nvshmem_root}/lib64/cmake/nvshmem
  )
endif()

function(_nvshmem_detect_version include_dir)
  unset(NVSHMEM_VERSION PARENT_SCOPE)

  set(_nvshmem_version_header "${include_dir}/non_abi/nvshmem_version.h")
  if(EXISTS "${_nvshmem_version_header}")
    file(READ "${_nvshmem_version_header}" _nvshmem_header_content)
    string(REGEX MATCH "#define[ \t]+NVSHMEM_VENDOR_MAJOR_VERSION[ \t]+([0-9]+)" _major_match "${_nvshmem_header_content}")
    string(REGEX MATCH "#define[ \t]+NVSHMEM_VENDOR_MINOR_VERSION[ \t]+([0-9]+)" _minor_match "${_nvshmem_header_content}")
  elseif(EXISTS "${include_dir}/nvshmem.h")
    file(READ "${include_dir}/nvshmem.h" _nvshmem_header_content)
    string(REGEX MATCH "#define[ \t]+NVSHMEM_VERSION_MAJOR[ \t]+([0-9]+)" _major_match "${_nvshmem_header_content}")
    string(REGEX MATCH "#define[ \t]+NVSHMEM_VERSION_MINOR[ \t]+([0-9]+)" _minor_match "${_nvshmem_header_content}")
  endif()

  if(_major_match AND _minor_match)
    string(REGEX REPLACE ".*#define[ \t]+(NVSHMEM_VENDOR_)?VERSION_MAJOR[ \t]+([0-9]+).*" "\\2" _nvshmem_major "${_major_match}")
    string(REGEX REPLACE ".*#define[ \t]+(NVSHMEM_VENDOR_)?VERSION_MINOR[ \t]+([0-9]+).*" "\\2" _nvshmem_minor "${_minor_match}")
    set(NVSHMEM_VERSION "${_nvshmem_major}.${_nvshmem_minor}" PARENT_SCOPE)
  endif()
endfunction()

set(NVSHMEM_FOUND FALSE)

# Prefer the vendor-supplied package when available. NVHPC 26.3 installs split
# host/device targets here instead of a single libnvshmem.* artifact.
find_package(NVSHMEM CONFIG QUIET NO_MODULE
  HINTS ${_nvshmem_config_hints}
  PATH_SUFFIXES lib/cmake/nvshmem lib64/cmake/nvshmem
)

if(NVSHMEM_FOUND AND TARGET nvshmem::nvshmem_host)
  get_target_property(_nvshmem_target_includes nvshmem::nvshmem_host INTERFACE_INCLUDE_DIRECTORIES)
  if(_nvshmem_target_includes)
    list(GET _nvshmem_target_includes 0 NVSHMEM_INCLUDE_DIR)
  elseif(_nvshmem_root)
    set(NVSHMEM_INCLUDE_DIR "${_nvshmem_root}/include")
  endif()

  set(NVSHMEM_INCLUDE_DIRS "${NVSHMEM_INCLUDE_DIR}")
  set(NVSHMEM_LIBRARIES nvshmem::nvshmem_host)
  if(TARGET nvshmem::nvshmem_device)
    list(APPEND NVSHMEM_LIBRARIES nvshmem::nvshmem_device)
  endif()
else()
  set(NVSHMEM_FOUND FALSE)

  find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS ${_nvshmem_include_hints}
  )

  find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem nvshmem_host
    HINTS ${_nvshmem_lib_hints}
  )

  find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES nvshmem_device
    HINTS ${_nvshmem_lib_hints}
  )

  if(NOT NVSHMEM_HOST_LIBRARY OR NOT NVSHMEM_INCLUDE_DIR)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(PC_NVSHMEM QUIET nvshmem)
      if(PC_NVSHMEM_FOUND)
        set(NVSHMEM_INCLUDE_DIR ${PC_NVSHMEM_INCLUDE_DIRS})
        set(NVSHMEM_HOST_LIBRARY ${PC_NVSHMEM_LIBRARIES})
      endif()
    endif()
  endif()

  if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_HOST_LIBRARY)
    set(NVSHMEM_FOUND TRUE)
    set(NVSHMEM_INCLUDE_DIRS "${NVSHMEM_INCLUDE_DIR}")
    set(NVSHMEM_LIBRARIES "${NVSHMEM_HOST_LIBRARY}")
    if(NVSHMEM_DEVICE_LIBRARY)
      list(APPEND NVSHMEM_LIBRARIES "${NVSHMEM_DEVICE_LIBRARY}")
      find_package(CUDAToolkit QUIET)
      if(TARGET CUDA::cudart_static)
        list(APPEND NVSHMEM_LIBRARIES CUDA::cudart_static)
      endif()
    endif()
  endif()
endif()

if(NVSHMEM_FOUND AND NVSHMEM_INCLUDE_DIR)
  _nvshmem_detect_version("${NVSHMEM_INCLUDE_DIR}")
endif()

if(NVSHMEM_FOUND AND NOT TARGET NVSHMEM::NVSHMEM)
  add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
  set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIRS}"
    INTERFACE_LINK_LIBRARIES "${NVSHMEM_LIBRARIES}"
  )
endif()

mark_as_advanced(NVSHMEM_INCLUDE_DIR NVSHMEM_HOST_LIBRARY NVSHMEM_DEVICE_LIBRARY)

if(NVSHMEM_FOUND)
  message(STATUS "Found NVSHMEM: ${NVSHMEM_LIBRARIES} (include: ${NVSHMEM_INCLUDE_DIRS})")
else()
  message(FATAL_ERROR "Could not find NVSHMEM library or headers. Please set NVSHMEM_ROOT to the installation prefix.")
endif()
