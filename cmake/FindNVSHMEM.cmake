## FindNVSHMEM.cmake - Locate NVSHMEM library and headers
#
# Compatible with:
#   - NVHPC 25.x (classic libnvshmem.so layout)
#   - NVHPC 26.x (official CMake package with split host/device targets)
#
# Provides:
#   NVSHMEM_FOUND
#   NVSHMEM_LIBRARIES
#   NVSHMEM_INCLUDE_DIRS
#   NVSHMEM_VERSION
#
# Imported targets:
#   NVSHMEM::NVSHMEM          - compatibility target
#   nvshmem::nvshmem_host     - vendor target when available
#   nvshmem::nvshmem_device   - vendor target when available
#
# Usage:
#   find_package(NVSHMEM REQUIRED)
#   target_link_libraries(my_target PRIVATE NVSHMEM::NVSHMEM)
#

include(CMakeFindDependencyMacro)

if(NOT NVSHMEM_FIND_VERSION)
  set(NVSHMEM_FIND_VERSION "")
endif()

# -----------------------------------------------------------------------------
# Discover installation root
# -----------------------------------------------------------------------------

set(_nvshmem_root "")

if(DEFINED NVSHMEM_ROOT)
  set(_nvshmem_root "${NVSHMEM_ROOT}")
elseif(DEFINED ENV{NVSHMEM_ROOT})
  set(_nvshmem_root "$ENV{NVSHMEM_ROOT}")
endif()

# NVHPC default installation layouts
set(_nvshmem_config_hints
  ${_nvshmem_root}
  /opt/nvidia/hpc_sdk/Linux_x86_64
)

# -----------------------------------------------------------------------------
# NVHPC 26.x path:
# Use vendor-provided CMake package if available.
# -----------------------------------------------------------------------------

set(NVSHMEM_FOUND FALSE)

# Required because nvshmem::nvshmem_device links against:
#   CUDA::cudart_static
# in NVHPC 26.x.
find_package(CUDAToolkit QUIET)

find_package(NVSHMEM CONFIG QUIET NO_MODULE
  HINTS ${_nvshmem_config_hints}
  PATH_SUFFIXES
    lib/cmake/nvshmem
    lib64/cmake/nvshmem
)

if(TARGET nvshmem::nvshmem_host)
  get_target_property(_nvshmem_inc
    nvshmem::nvshmem_host
    INTERFACE_INCLUDE_DIRECTORIES
  )

  set(NVSHMEM_FOUND TRUE)
  set(NVSHMEM_INCLUDE_DIRS "${_nvshmem_inc}")

  # NVHPC 26.x separates host/device libraries. Both are required
  # for proper linkage of device-side NVSHMEM symbols.
  set(NVSHMEM_LIBRARIES
    nvshmem::nvshmem_host
    nvshmem::nvshmem_device
  )

  # Create compatibility target expected by older projects.
  if(NOT TARGET NVSHMEM::NVSHMEM)
    add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
    set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
      INTERFACE_LINK_LIBRARIES
        "nvshmem::nvshmem_host;nvshmem::nvshmem_device"
      INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIRS}"
    )
  endif()

  message(STATUS
    "Found NVSHMEM: "
    "nvshmem::nvshmem_host;nvshmem::nvshmem_device "
    "(include: ${NVSHMEM_INCLUDE_DIRS})"
  )
endif()

# -----------------------------------------------------------------------------
# Fallback for NVHPC 25.x and older installations.
# -----------------------------------------------------------------------------

if(NOT NVSHMEM_FOUND)

  find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS
      ${_nvshmem_root}/include
      ${_nvshmem_root}/nvshmem/include
      /usr/local/include
      /usr/include
    PATH_SUFFIXES include
  )

  find_library(NVSHMEM_LIBRARY
    NAMES nvshmem
    HINTS
      ${_nvshmem_root}/lib
      ${_nvshmem_root}/lib64
      ${_nvshmem_root}/nvshmem/lib
      ${_nvshmem_root}/nvshmem/lib64
      /usr/local/lib
      /usr/local/lib64
      /usr/lib
      /usr/lib64
    PATH_SUFFIXES lib lib64
  )

  # pkg-config fallback
  if(NOT NVSHMEM_LIBRARY OR NOT NVSHMEM_INCLUDE_DIR)
    find_package(PkgConfig QUIET)

    if(PkgConfig_FOUND)
      pkg_check_modules(PC_NVSHMEM QUIET nvshmem)

      if(PC_NVSHMEM_FOUND)
        set(NVSHMEM_INCLUDE_DIR ${PC_NVSHMEM_INCLUDE_DIRS})
        set(NVSHMEM_LIBRARY ${PC_NVSHMEM_LIBRARIES})
      endif()
    endif()
  endif()

  if(NVSHMEM_INCLUDE_DIR AND NVSHMEM_LIBRARY)
    set(NVSHMEM_FOUND TRUE)
    set(NVSHMEM_INCLUDE_DIRS ${NVSHMEM_INCLUDE_DIR})
    set(NVSHMEM_LIBRARIES ${NVSHMEM_LIBRARY})

    # Extract version if available
    if(EXISTS "${NVSHMEM_INCLUDE_DIR}/nvshmem.h")
      file(READ "${NVSHMEM_INCLUDE_DIR}/nvshmem.h"
        _nvshmem_header_content)

      string(REGEX MATCH
        "#define[ \t]+NVSHMEM_VERSION_MAJOR[ \t]+([0-9]+)"
        _major_match
        "${_nvshmem_header_content}"
      )

      string(REGEX MATCH
        "#define[ \t]+NVSHMEM_VERSION_MINOR[ \t]+([0-9]+)"
        _minor_match
        "${_nvshmem_header_content}"
      )

      if(_major_match AND _minor_match)
        string(REGEX REPLACE
          ".*#define[ \t]+NVSHMEM_VERSION_MAJOR[ \t]+([0-9]+).*"
          "\\1"
          NVSHMEM_VERSION_MAJOR
          "${_major_match}"
        )

        string(REGEX REPLACE
          ".*#define[ \t]+NVSHMEM_VERSION_MINOR[ \t]+([0-9]+).*"
          "\\1"
          NVSHMEM_VERSION_MINOR
          "${_minor_match}"
        )

        set(NVSHMEM_VERSION
          "${NVSHMEM_VERSION_MAJOR}.${NVSHMEM_VERSION_MINOR}"
        )
      endif()
    endif()

    # Compatibility imported target
    if(NOT TARGET NVSHMEM::NVSHMEM)
      add_library(NVSHMEM::NVSHMEM UNKNOWN IMPORTED)
      set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
        IMPORTED_LOCATION "${NVSHMEM_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NVSHMEM_INCLUDE_DIRS}"
      )
    endif()

    message(STATUS
      "Found NVSHMEM: ${NVSHMEM_LIBRARIES} "
      "(include: ${NVSHMEM_INCLUDE_DIRS})"
    )
  endif()
endif()

# -----------------------------------------------------------------------------
# Final result
# -----------------------------------------------------------------------------

mark_as_advanced(
  NVSHMEM_INCLUDE_DIR
  NVSHMEM_LIBRARY
)

if(NOT NVSHMEM_FOUND)
  message(FATAL_ERROR
    "Could not find NVSHMEM library or headers. "
    "Please set NVSHMEM_ROOT to the installation prefix."
  )
endif()
