# Try to find oneCCL (Intel oneAPI Collective Communications Library)
# Provides imported target: oneCCL::oneCCL

# Search for the main header file
find_path(oneCCL_INCLUDE_DIR
    NAMES ccl.hpp
    PATHS
        $ENV{CCL_ROOT}/include
        $ENV{CCL_ROOT}/include/oneapi
        /opt/intel/oneapi/ccl/latest/include
        /opt/intel/oneapi/ccl/latest/include/oneapi
        /usr/include/oneapi
)

# Search for the library
find_library(oneCCL_LIBRARY
    NAMES ccl
    PATHS
        $ENV{CCL_ROOT}/lib
        $ENV{CCL_ROOT}/lib/intel64
        /opt/intel/oneapi/ccl/latest/lib
        /opt/intel/oneapi/ccl/latest/lib/intel64
        /usr/lib
        /usr/lib/intel64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(oneCCL
    REQUIRED_VARS oneCCL_LIBRARY oneCCL_INCLUDE_DIR
)

if(oneCCL_FOUND)
    set(oneCCL_INCLUDE_DIRS ${oneCCL_INCLUDE_DIR})
    set(oneCCL_LIBRARIES ${oneCCL_LIBRARY})

    add_library(oneCCL::oneCCL UNKNOWN IMPORTED)
    set_target_properties(oneCCL::oneCCL PROPERTIES
        IMPORTED_LOCATION "${oneCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${oneCCL_INCLUDE_DIR}"
    )
endif()
