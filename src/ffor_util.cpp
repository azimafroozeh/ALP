#include "fls/ffor_util.hpp"
#include <assert.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace fastlanes {

template <typename UT>
static uint8_t count_bits(UT x) noexcept {
	static_assert(std::is_same_v<UT, uint64_t>        //
	                  || std::is_same_v<UT, uint32_t> //
	                  || std::is_same_v<UT, uint16_t> //
	                  || std::is_same_v<UT, uint8_t>,
	              "UT must be uint64_t, uint32_t, uint16_t, or uint8_t");

	if (x == 0) { return 0; }

	if constexpr (std::is_same_v<UT, uint64_t>) {
		return static_cast<uint8_t>(64 - __builtin_clzll(x));
	} else if constexpr (std::is_same_v<UT, uint32_t>) {
		return static_cast<uint8_t>(32 - __builtin_clz(x));
	} else if constexpr (std::is_same_v<UT, uint16_t>) {
		return static_cast<uint8_t>(16 - (__builtin_clz(static_cast<uint32_t>(x)) - 16));
	} else if constexpr (std::is_same_v<UT, uint8_t>) {
		return static_cast<uint8_t>(8 - (__builtin_clz(static_cast<uint32_t>(x)) - 24));
	}

	return 0;
}

// Concept to enforce PT is an integral type
template <typename T>
concept integral = std::is_integral_v<T>;

template <integral PT>
uint8_t count_bits(PT max, PT min) {
	using UT = std::conditional_t<std::is_signed_v<PT>, std::make_unsigned_t<PT>, PT>;

	const UT delta = static_cast<UT>(max) - static_cast<UT>(min);

	const auto res = count_bits<UT>(delta);
	return res;
}

template uint8_t count_bits(int64_t max, int64_t min);
template uint8_t count_bits(int32_t max, int32_t min);
template uint8_t count_bits(int16_t max, int16_t min);
template uint8_t count_bits(int8_t max, int8_t min);

} // namespace fastlanes
