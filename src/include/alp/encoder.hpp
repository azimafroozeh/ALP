#ifndef ALP_ENCODER_HPP
#define ALP_ENCODER_HPP

#include "alp/config.hpp"
#include "alp/constants.hpp"
#include "alp/decoder.hpp"
#include "alp/sampler.hpp"
#include "common.hpp"
#include "constants.hpp"
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <list>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __AVX2__

#include <immintrin.h>

#endif

/*
 * ALP Encoding
 */
namespace alp {

template <typename PT>
struct state {
	using UT = typename inner_t<PT>::ut;
	using ST = typename inner_t<PT>::st;

	Scheme   scheme {Scheme::INVALID};
	uint16_t vector_size {config::VECTOR_SIZE};
	uint16_t exceptions_count {0};
	size_t   sampled_values_n {0};

	// ALP
	uint16_t                         k_combinations {5};
	std::vector<std::pair<int, int>> best_k_combinations;
	uint8_t                          exp {};
	uint8_t                          fac {};
	bw_t                             bit_width {};
	ST                               for_base {};

	// ALP RD
	bw_t                                   right_bit_width {0};
	bw_t                                   left_bit_width {0};
	UT                                     right_for_base {0}; // Always 0
	uint16_t                               left_for_base {0};  // Always 0
	uint16_t                               left_parts_dict[config::MAX_RD_DICTIONARY_SIZE] {};
	uint8_t                                actual_dictionary_size {};
	uint32_t                               actual_dictionary_size_bytes {};
	std::unordered_map<uint16_t, uint16_t> left_parts_dict_map;
};

template <typename PT>
struct encoder {
	using UT = typename inner_t<PT>::ut;
	using ST = typename inner_t<PT>::st;
	/*
	 * Check for special values which are impossible for ALP to encode
	 * because they cannot be cast to int64 without an undefined behaviour
	 */
	//! Analyze FFOR to obtain bitwidth and frame-of-reference value
	static void analyze_ffor(const ST* input_vector, bw_t& bit_width, ST* base_for);
	/*
	 * Function to sort the best combinations from each vector sampled from the rowgroup
	 * First criteria is number of times it appears
	 * Second criteria is bigger exponent
	 * Third criteria is bigger factor
	 */
	static bool compare_best_combinations(const std::pair<std::pair<int, int>, int>& t1,
	                                      const std::pair<std::pair<int, int>, int>& t2);
	/*
	 * Find the best combinations of factor-exponent from each vector sampled from a rowgroup
	 * This function is called once per rowgroup
	 * This operates over ALP first level samples
	 */
	static void find_top_k_combinations(const PT* smp_arr, state<PT>& stt);
	/*
	 * Find the best combination of factor-exponent for a vector from within the best k combinations
	 * This is ALP second level sampling
	 */
	static void find_best_exponent_factor_from_combinations(const std::vector<std::pair<int, int>>& top_combinations,
	                                                        const uint8_t                           top_k,
	                                                        const PT*                               input_vector,
	                                                        const uint16_t                          input_vector_size,
	                                                        uint8_t&                                factor,
	                                                        uint8_t&                                exponent);

	static void encode_simdized(const PT*            input_vector,
	                            PT*                  exceptions,
	                            exp_p_t*             exceptions_positions,
	                            exp_c_t*             exceptions_count,
	                            ST*                  encoded_integers,
	                            const factor_idx_t   factor_idx,
	                            const exponent_idx_t exponent_idx);

	static void encode(const PT*  input_vector,
	                   PT*        exceptions,
	                   uint16_t*  exceptions_positions,
	                   uint16_t*  exceptions_count,
	                   ST*        encoded_integers,
	                   state<PT>& stt);

	static void
	init(const PT* data_column, const size_t column_offset, const size_t tuples_count, PT* sample_arr, state<PT>& stt);
};

} // namespace alp

#endif
