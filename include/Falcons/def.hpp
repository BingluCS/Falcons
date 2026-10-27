#ifndef Falcons_DEF_HPP
#define Falcons_DEF_HPP

namespace Falcons {

typedef unsigned int uint;
typedef unsigned char uchar;
#define Falcons_ERROR_COMP_BUFFER_NOT_LARGE_ENOUGH \
    "The buffer for compressed data is not large enough."
}  // namespace Falcons

#ifdef _MSC_VER
#define ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define ALWAYS_INLINE inline
#endif

#endif
