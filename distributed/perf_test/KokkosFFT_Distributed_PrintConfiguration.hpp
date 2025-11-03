#ifndef KOKKOSFFT_DISTRIBUTED_PRINT_CONFIGURATION_HPP
#define KOKKOSFFT_DISTRIBUTED_PRINT_CONFIGURATION_HPP

#include <iostream>
#include <KokkosFFT.hpp>
#include "KokkosFFT_Distributed_config.hpp"
#include "KokkosFFT_Distributed_TplsVersion.hpp"

namespace KokkosFFT {
namespace Distributed {
namespace Impl {

inline void print_cufft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_CUFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_CUFFT: " << cufft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_CUFFT: no\n";
#endif
}

inline void print_rocfft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_ROCFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ROCFFT: " << rocfft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ROCFFT: no\n";
#endif
}

inline void print_hipfft_version_if_enabled(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: " << hipfft_version_string() << "\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: no\n";
#endif
}

inline void print_enabled_tpls(std::ostream& os) {
#if defined(KOKKOSFFT_ENABLE_TPL_FFTW)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_FFTW: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_FFTW: no\n";
#endif

  print_cufft_version_if_enabled(os);

#if defined(KOKKOSFFT_ENABLE_TPL_HIPFFT)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_HIPFFT: no\n";
#endif

#if defined(KOKKOSFFT_ENABLE_TPL_ONEMKL)
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ONEMKL: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_ENABLE_TPL_ONEMKL: no\n";
#endif

#if defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_NCCL)
  os << "  "
     << "KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_NCCL: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_NCCL: no\n";
#endif

#if defined(KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL)
  os << "  "
     << "KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL: yes\n";
#else
  os << "  "
     << "KOKKOSFFT_DISTRIBUTED_ENABLE_TPL_RCCL: no\n";
#endif
}

inline void print_version(std::ostream& os) {
  // KOKKOSFFT_VERSION is used because MAJOR, MINOR and PATCH macros
  // are not available in FFT
  os << "  "
     << "KokkosFFT::Distributed Version: " << DISTRIBUTED_FFT_VERSION_MAJOR
     << "." << DISTRIBUTED_FFT_VERSION_MINOR << "."
     << DISTRIBUTED_FFT_VERSION_PATCH << '\n';
}
}  // namespace Impl

inline void print_configuration(std::ostream& os) {
  Impl::print_version(os);

  os << "TPLs: \n";
  Impl::print_enabled_tpls(os);
}
}  // namespace Distributed
}  // namespace KokkosFFT

#endif
