# - Try to find the NVIDIA NCCL library
# Once done, this will define
#
#  NCCL_FOUND         – Set to TRUE if NCCL was found
#  NCCL_INCLUDE_DIRS  – where to find nccl.h
#  NCCL_LIBRARIES     – List of libraries to link against
#  NCCL_VERSION       – NCCL version string if detectable
#
# And creates the imported target:
#  NCCL::NCCL         - Imported target for NCCL
#
# Usage:
#   find_package(NCCL REQUIRED)
#   target_link_libraries(<tgt> PUBLIC NCCL::NCCL)

# Avoid repeated processing
if (TARGET NCCL::NCCL)
  return()
endif()

include(FindPackageHandleStandardArgs)

# Allow override via NCCL_ROOT
set(NCCL_ROOT
    $ENV{NCCL_ROOT}
    CACHE PATH "Root directory of NCCL installation")

# Detect system architecture for better path handling
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_NCCL_ARCH "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_NCCL_ARCH "x86_64")
else()
    set(_NCCL_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()

message(STATUS "NCCL_ROOT: ${NCCL_ROOT}")
message(STATUS "Detected architecture: ${_NCCL_ARCH}")

# 1) Locate header
find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
    ${NCCL_ROOT}/include
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    $ENV{CUDA_HOME}/include
    # Try to correct architecture mismatch in NCCL_ROOT
    ${NCCL_ROOT}/../../../${_NCCL_ARCH}/cores/nvidia/25.5/Linux_${_NCCL_ARCH}/25.5/comm_libs/nccl/include
  PATHS
    /usr/local/include
    /usr/include
    /opt/nvidia/hpc_sdk/Linux_x86_64/*/comm_libs/nccl/include
    /opt/nvidia/hpc_sdk/Linux_aarch64/*/comm_libs/nccl/include
    /work/opt/local/${_NCCL_ARCH}/cores/nvidia/*/Linux_${_NCCL_ARCH}/*/comm_libs/nccl/include
  PATH_SUFFIXES nccl
)

# 2) Locate library
find_library(NCCL_LIBRARY
  NAMES nccl libnccl
  HINTS
    ${NCCL_ROOT}/lib
    ${NCCL_ROOT}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/lib
    $ENV{CUDA_HOME}/lib64
    $ENV{CUDA_HOME}/lib
    # Try to correct architecture mismatch in NCCL_ROOT
    ${NCCL_ROOT}/../../../${_NCCL_ARCH}/cores/nvidia/25.5/Linux_${_NCCL_ARCH}/25.5/comm_libs/nccl/lib
  PATHS
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    /opt/nvidia/hpc_sdk/Linux_x86_64/*/comm_libs/nccl/lib
    /opt/nvidia/hpc_sdk/Linux_aarch64/*/comm_libs/nccl/lib
    /work/opt/local/${_NCCL_ARCH}/cores/nvidia/*/Linux_${_NCCL_ARCH}/*/comm_libs/nccl/lib
  PATH_SUFFIXES 
    x86_64-linux-gnu
    aarch64-linux-gnu
    nccl
)

# 3) Extract version from header if found
set(NCCL_VERSION "")
if (NCCL_INCLUDE_DIR AND EXISTS "${NCCL_INCLUDE_DIR}/nccl.h")
  # Try modern version format with separate MAJOR/MINOR/PATCH defines
  file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_lines
       REGEX "#define[ \t]+(NCCL_VERSION_MAJOR|NCCL_VERSION_MINOR|NCCL_VERSION_PATCH)[ \t]+[0-9]+")
  
  foreach(_line ${_nccl_version_lines})
    if(_line MATCHES "#define[ \t]+NCCL_VERSION_MAJOR[ \t]+([0-9]+)")
      set(NCCL_VERSION_MAJOR ${CMAKE_MATCH_1})
    elseif(_line MATCHES "#define[ \t]+NCCL_VERSION_MINOR[ \t]+([0-9]+)")
      set(NCCL_VERSION_MINOR ${CMAKE_MATCH_1})
    elseif(_line MATCHES "#define[ \t]+NCCL_VERSION_PATCH[ \t]+([0-9]+)")
      set(NCCL_VERSION_PATCH ${CMAKE_MATCH_1})
    endif()
  endforeach()
  
  if(DEFINED NCCL_VERSION_MAJOR AND DEFINED NCCL_VERSION_MINOR AND DEFINED NCCL_VERSION_PATCH)
    set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
  else()
    # Try legacy version format with NCCL_VERSION_CODE
    file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" _nccl_version_code_line
         REGEX "#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+")
    if(_nccl_version_code_line MATCHES "#define[ \t]+NCCL_VERSION_CODE[ \t]+([0-9]+)")
      set(_version_code ${CMAKE_MATCH_1})
      # Decode version from NCCL_VERSION_CODE (format: XXYYZZ or XXXYYZZZ)
      if(_version_code GREATER_EQUAL 10000)
        # New format: XXXYYZZZ
        math(EXPR NCCL_VERSION_MAJOR "${_version_code} / 10000")
        math(EXPR _temp "${_version_code} % 10000")
        math(EXPR NCCL_VERSION_MINOR "${_temp} / 100")
        math(EXPR NCCL_VERSION_PATCH "${_temp} % 100")
      else()
        # Old format: XXYYZZ  
        math(EXPR NCCL_VERSION_MAJOR "${_version_code} / 1000")
        math(EXPR _temp "${_version_code} % 1000")
        math(EXPR NCCL_VERSION_MINOR "${_temp} / 100")
        math(EXPR NCCL_VERSION_PATCH "${_temp} % 100")
      endif()
      set(NCCL_VERSION "${NCCL_VERSION_MAJOR}.${NCCL_VERSION_MINOR}.${NCCL_VERSION_PATCH}")
    endif()
  endif()
endif()

# 4) Handle result
find_package_handle_standard_args(NCCL
  REQUIRED_VARS NCCL_LIBRARY NCCL_INCLUDE_DIR
  VERSION_VAR NCCL_VERSION
)

# Set legacy variables for backwards compatibility
if(NCCL_FOUND)
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  
  # Create imported target
  if(NOT TARGET NCCL::NCCL)
    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
      IMPORTED_LOCATION "${NCCL_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )
    
    # NCCL typically requires CUDA runtime
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set_property(TARGET NCCL::NCCL APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES CUDA::cudart)
    endif()
  endif()
  
  if(NOT NCCL_FIND_QUIETLY)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY} (include: ${NCCL_INCLUDE_DIR})")
    if(NCCL_VERSION)
      message(STATUS "NCCL version: ${NCCL_VERSION}")
    endif()
  endif()
endif()

mark_as_advanced(NCCL_LIBRARY NCCL_INCLUDE_DIR)
