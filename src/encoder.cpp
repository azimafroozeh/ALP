#include "alp/encoder.hpp"

namespace alp {

template <typename UT>
static uint8_t count_bits(UT x) {
	if (x == 0) { return 0; }
	if constexpr (std::is_same_v<UT, uint64_t>) {
		return static_cast<uint8_t>(64 - __builtin_clzll(x));
	} else {
		return static_cast<uint8_t>(32 - __builtin_clz(x));
	}
}

template <typename PT>
bool is_impossible_to_encode(const PT n) {
	return !std::isfinite(n)                          //
	       || std::isnan(n)                           //
	       || n > Constants<PT>::ENCODING_UPPER_LIMIT //
	       || n < Constants<PT>::ENCODING_LOWER_LIMIT //
	       || (n == 0.0 && std::signbit(n));          //! Verification for -0.0
}

//! Scalar encoding a single value with ALP
template <typename PT, typename ST>
ST encode_value(const PT value, const factor_idx_t factor_idx, const exponent_idx_t exponent_idx) {
	PT tmp_encoded_value = value * Constants<PT>::EXP_ARR[exponent_idx] * Constants<PT>::FRAC_ARR[factor_idx];
	if (is_impossible_to_encode<PT>(tmp_encoded_value)) { return Constants<PT>::ENCODING_UPPER_LIMIT; }

	tmp_encoded_value = tmp_encoded_value + Constants<PT>::MAGIC_NUMBER - Constants<PT>::MAGIC_NUMBER;
	return static_cast<ST>(tmp_encoded_value);
}

template <typename ST, typename UT>
static uint8_t count_bits(ST max, ST min) {
	const auto delta = (static_cast<UT>(max) - static_cast<UT>(min));
	auto       res   = count_bits<UT>(delta);
	return res;
}

template <typename PT>
void encoder<PT>::encode(const PT*  input_vector,
                         PT*        exceptions,
                         uint16_t*  exceptions_positions,
                         uint16_t*  exceptions_count,
                         ST*        encoded_integers,
                         state<PT>& stt) {

	if (stt.k_combinations > 1) { // Only if more than 1 found top combinations we sample and search
		find_best_exponent_factor_from_combinations(
		    stt.best_k_combinations, stt.k_combinations, input_vector, stt.vector_size, stt.fac, stt.exp);
	} else {
		stt.exp = stt.best_k_combinations[0].first;
		stt.fac = stt.best_k_combinations[0].second;
	}
	encode_simdized(
	    input_vector, exceptions, exceptions_positions, exceptions_count, encoded_integers, stt.fac, stt.exp);
}

template <typename PT>
void encoder<PT>::encode_simdized(const PT*            input_vector,
                                  PT*                  exceptions,
                                  exp_p_t*             exceptions_positions,
                                  exp_c_t*             exceptions_count,
                                  ST*                  encoded_integers,
                                  const factor_idx_t   factor_idx,
                                  const exponent_idx_t exponent_idx) {
	alignas(64) static PT ENCODED_VALUE_ARR[1024];
	alignas(64) static PT VALUE_ARR_WITHOUT_SPECIALS[1024];
	alignas(64) static UT TMP_INDEX_ARR[1024];

	exp_p_t  current_exceptions_count {0};
	uint64_t exceptions_idx {0};

	// make copy of input with all special values replaced by  ENCODING_UPPER_LIMIT
	const auto* tmp_input = reinterpret_cast<const UT*>(input_vector);
	for (size_t i {0}; i < config::VECTOR_SIZE; i++) {
		const auto is_special =
		    ((tmp_input[i] & Constants<PT>::SIGN_BIT_MASK) >=
		     Constants<PT>::EXPONENTIAL_BITS_MASK) // any NaN, +inf and -inf
		                                           // (https://stackoverflow.com/questions/29730530/)
		    || tmp_input[i] == Constants<PT>::NEGATIVE_ZERO;

		if (is_special) {
			VALUE_ARR_WITHOUT_SPECIALS[i] = Constants<PT>::ENCODING_UPPER_LIMIT;
		} else {
			VALUE_ARR_WITHOUT_SPECIALS[i] = input_vector[i];
		}
	}

#pragma clang loop vectorize_width(64)
	for (size_t i {0}; i < config::VECTOR_SIZE; i++) {
		auto const actual_value = VALUE_ARR_WITHOUT_SPECIALS[i];

		// Attempt conversion
		const ST encoded_value = encode_value<PT, ST>(actual_value, factor_idx, exponent_idx);
		encoded_integers[i]    = encoded_value;
		const PT decoded_value = decoder<PT>::decode_value(encoded_value, factor_idx, exponent_idx);
		ENCODED_VALUE_ARR[i]   = decoded_value;
	}

#ifdef __AVX512F__
	if constexpr (std::is_same_v<PT, double>) {
		for (size_t i {0}; i < config::VECTOR_SIZE; i = i + 8) {
			__m512d l            = _mm512_loadu_pd(ENCODED_VALUE_ARR + i);
			__m512d r            = _mm512_loadu_pd(VALUE_ARR_WITHOUT_SPECIALS + i);
			__m512i index        = _mm512_loadu_pd(DOUBLE_INDEX_ARR + i);
			auto    is_exception = _mm512_cmpneq_pd_mask(l, r);
			_mm512_mask_compressstoreu_pd(TMP_INDEX_ARR + exceptions_idx, is_exception, index);
			exceptions_idx += LOOKUP_TABLE[is_exception];
		}
	} else {
		for (size_t i {0}; i < config::VECTOR_SIZE; i = i + 16) {
			__m512   l            = _mm512_loadu_ps(ENCODED_VALUE_ARR + i);
			__m512   r            = _mm512_loadu_ps(VALUE_ARR_WITHOUT_SPECIALS + i);
			__m512i  index        = _mm512_loadu_si512(FLOAT_INDEX_ARR + i);
			uint16_t is_exception = _mm512_cmpneq_ps_mask(l, r);
			_mm512_mask_compressstoreu_ps(TMP_INDEX_ARR + exceptions_idx, is_exception, index);
			auto n_exceptions = LOOKUP_TABLE[is_exception & 0b0000000011111111] + LOOKUP_TABLE[is_exception >> 8];
			exceptions_idx += n_exceptions; // Update index
		}
	}
#else
	for (UT i {0}; i < config::VECTOR_SIZE; i++) {
		auto l                        = ENCODED_VALUE_ARR[i];
		auto r                        = VALUE_ARR_WITHOUT_SPECIALS[i];
		auto is_exception             = (l != r);
		TMP_INDEX_ARR[exceptions_idx] = i;
		exceptions_idx += is_exception;
	}
#endif

	ST a_non_exception_value = 0;
	for (size_t i {0}; i < config::VECTOR_SIZE; i++) {
		if (i != TMP_INDEX_ARR[i]) {
			a_non_exception_value = encoded_integers[i];
			break;
		}
	}

	for (exp_p_t j {0}; j < exceptions_idx; j++) {
		auto       i                                   = static_cast<exp_p_t>(TMP_INDEX_ARR[j]);
		const auto actual_value                        = input_vector[i];
		encoded_integers[i]                            = a_non_exception_value;
		exceptions[current_exceptions_count]           = actual_value;
		exceptions_positions[current_exceptions_count] = i;
		current_exceptions_count                       = current_exceptions_count + 1;
	}

	*exceptions_count = current_exceptions_count;
}

template <typename PT>
void encoder<PT>::init(
    const PT* data_column, const size_t column_offset, const size_t tuples_count, PT* sample_arr, state<PT>& stt) {
	stt.scheme           = Scheme::ALP;
	stt.sampled_values_n = sampler::first_level_sample<PT>(data_column, column_offset, tuples_count, sample_arr);
	stt.k_combinations   = config::MAX_K_COMBINATIONS;
	stt.best_k_combinations.clear();
	find_top_k_combinations(sample_arr, stt);
}

template <typename PT>
void encoder<PT>::find_best_exponent_factor_from_combinations(
    const std::vector<std::pair<uint8_t, uint8_t>>& top_combinations,
    const uint8_t                                   top_k,
    const PT*                                       input_vector,
    const uint16_t                                  input_vector_size,
    uint8_t&                                        factor,
    uint8_t&                                        exponent) {
	uint8_t  found_exponent {0};
	uint8_t  found_factor {0};
	uint64_t best_estimated_compression_size {0};
	uint8_t  worse_threshold_count {0};

	const size_t sample_increments = std::max(
	    static_cast<size_t>(1), static_cast<size_t>(std::ceil(input_vector_size / config::SAMPLES_PER_VECTOR)));

	// We try each K combination in search for the one which minimize the compression size in the vector
	for (size_t k {0}; k < top_k; k++) {
		const auto exp_idx    = top_combinations[k].first;
		const auto factor_idx = top_combinations[k].second;
		uint32_t   exception_count {0};
		uint32_t   estimated_bits_per_value {0};
		uint64_t   estimated_compression_size {0};
		ST         max_encoded_value {std::numeric_limits<ST>::min()};
		ST         min_encoded_value {std::numeric_limits<ST>::max()};

		for (size_t sample_idx = 0; sample_idx < input_vector_size; sample_idx += sample_increments) {
			const PT actual_value  = input_vector[sample_idx];
			const ST encoded_value = encode_value<PT, ST>(actual_value, factor_idx, exp_idx);
			const PT decoded_value = decoder<PT>::decode_value(encoded_value, factor_idx, exp_idx);
			if (decoded_value == actual_value) {
				if (encoded_value > max_encoded_value) { max_encoded_value = encoded_value; }
				if (encoded_value < min_encoded_value) { min_encoded_value = encoded_value; }
			} else {
				exception_count++;
			}
		}

		// Evaluate factor/exponent performance (we optimize for FOR)
		estimated_bits_per_value = count_bits<ST, UT>(max_encoded_value, min_encoded_value);
		estimated_compression_size += config::SAMPLES_PER_VECTOR * estimated_bits_per_value;
		estimated_compression_size += exception_count * (Constants<PT>::EXCEPTION_SIZE + EXCEPTION_POSITION_SIZE);

		if (k == 0) { // First try with first combination
			best_estimated_compression_size = estimated_compression_size;
			found_factor                    = factor_idx;
			found_exponent                  = exp_idx;
			continue; // Go to second
		}
		if (estimated_compression_size >=
		    best_estimated_compression_size) { // If current is worse or equal than previous
			worse_threshold_count += 1;
			if (worse_threshold_count == SAMPLING_EARLY_EXIT_THRESHOLD) {
				break; // We stop only if two are worse
			}
			continue;
		}
		// Otherwise we replace best and continue with next
		best_estimated_compression_size = estimated_compression_size;
		found_factor                    = factor_idx;
		found_exponent                  = exp_idx;
		worse_threshold_count           = 0;
	}
	exponent = found_exponent;
	factor   = found_factor;
}

template <typename PT>
void encoder<PT>::find_top_k_combinations(const PT* smp_arr, state<PT>& stt) {
	const auto n_vectors_to_sample =
	    static_cast<uint64_t>(std::ceil(static_cast<PT>(stt.sampled_values_n) / config::SAMPLES_PER_VECTOR));
	const uint64_t                     samples_size = std::min(stt.sampled_values_n, config::SAMPLES_PER_VECTOR);
	std::map<std::pair<int, int>, int> global_combinations;
	uint64_t                           smp_offset {0};

	// For each vector in the rg sample
	size_t best_estimated_compression_size {(samples_size * (Constants<PT>::EXCEPTION_SIZE + EXCEPTION_POSITION_SIZE)) +
	                                        (samples_size * (Constants<PT>::EXCEPTION_SIZE))};
	for (size_t smp_n = 0; smp_n < n_vectors_to_sample; smp_n++) {
		uint8_t found_factor {0};
		uint8_t found_exponent {0};
		// We start our optimization with the worst possible total bits obtained from compression
		uint64_t sample_estimated_compression_size {
		    (samples_size * (Constants<PT>::EXCEPTION_SIZE + EXCEPTION_POSITION_SIZE)) +
		    (samples_size * (Constants<PT>::EXCEPTION_SIZE))}; // worst scenario

		// We try all combinations in search for the one which minimize the compression size
		for (uint8_t exp_ref = Constants<PT>::MAX_EXPONENT; exp_ref >= 0; exp_ref--) {
			for (uint8_t factor_idx = exp_ref; factor_idx >= 0; factor_idx--) {
				uint16_t exceptions_count           = {0};
				uint16_t non_exceptions_count       = {0};
				uint32_t estimated_bits_per_value   = {0};
				uint64_t estimated_compression_size = {0};
				ST       max_encoded_value          = {std::numeric_limits<ST>::min()};
				ST       min_encoded_value          = {std::numeric_limits<ST>::max()};

				for (size_t i = 0; i < samples_size; i++) {
					const PT actual_value  = smp_arr[smp_offset + i];
					const ST encoded_value = encode_value<PT, ST>(actual_value, factor_idx, exp_ref);
					const PT decoded_value = decoder<PT>::decode_value(encoded_value, factor_idx, exp_ref);
					if (decoded_value == actual_value) {
						non_exceptions_count++;
						if (encoded_value > max_encoded_value) { max_encoded_value = encoded_value; }
						if (encoded_value < min_encoded_value) { min_encoded_value = encoded_value; }
					} else {
						exceptions_count++;
					}
				}

				// We do not take into account combinations which yield to almsot all exceptions
				if (non_exceptions_count < 2) { continue; }

				// Evaluate factor/exponent compression size (we optimize for FOR)
				estimated_bits_per_value = count_bits<ST, UT>(max_encoded_value, min_encoded_value);
				estimated_compression_size += samples_size * estimated_bits_per_value;
				estimated_compression_size +=
				    exceptions_count * (Constants<PT>::EXCEPTION_SIZE + EXCEPTION_POSITION_SIZE);

				if ((estimated_compression_size < sample_estimated_compression_size) ||
				    (estimated_compression_size == sample_estimated_compression_size && (found_exponent < exp_ref)) ||
				    // We prefer bigger exponents
				    ((estimated_compression_size == sample_estimated_compression_size && found_exponent == exp_ref) &&
				     (found_factor < factor_idx)) // We prefer bigger factors
				) {
					sample_estimated_compression_size = estimated_compression_size;
					found_exponent                    = exp_ref;
					found_factor                      = factor_idx;
					if (sample_estimated_compression_size < best_estimated_compression_size) {
						best_estimated_compression_size = sample_estimated_compression_size;
					}
				}
			}
		}
		std::pair<int, int> cmb = std::make_pair(found_exponent, found_factor);
		global_combinations[cmb]++;
		smp_offset += samples_size;
	}

	// We adapt scheme if we were not able to achieve compression in the current rg
	if (best_estimated_compression_size >= Constants<PT>::RD_SIZE_THRESHOLD_LIMIT) {
		stt.scheme = Scheme::ALP_RD;
		return;
	}

	// Convert our hash to a Combination vector to be able to sort
	// Note that this vector is always small (< 10 combinations)
	std::vector<std::pair<std::pair<int, int>, int>> best_k_combinations;
	best_k_combinations.reserve(global_combinations.size());
	for (auto const& itr : global_combinations) {
		best_k_combinations.emplace_back(itr.first, // Pair exp, fac
		                                 itr.second // N of times it appeared
		);
	}
	// We sort combinations based on times they appeared
	std::sort(best_k_combinations.begin(), best_k_combinations.end(), compare_best_combinations);
	if (best_k_combinations.size() < stt.k_combinations) {
		stt.k_combinations = static_cast<uint8_t>(best_k_combinations.size());
	}

	// Save k' best exp, fac combination pairs
	for (size_t i {0}; i < stt.k_combinations; i++) {
		stt.best_k_combinations.push_back(best_k_combinations[i].first);
	}
}

template <typename PT>
bool encoder<PT>::compare_best_combinations(const std::pair<std::pair<int, int>, int>& t1,
                                            const std::pair<std::pair<int, int>, int>& t2) {
	return (t1.second > t2.second) || (t1.second == t2.second && (t2.first.first < t1.first.first)) ||
	       ((t1.second == t2.second && t2.first.first == t1.first.first) && (t2.first.second < t1.first.second));
}

template <typename PT>
void encoder<PT>::analyze_ffor(const ST* input_vector, bw_t& bit_width, ST* base_for) {
	auto min = std::numeric_limits<ST>::max();
	auto max = std::numeric_limits<ST>::min();

	for (size_t i {0}; i < config::VECTOR_SIZE; i++) {
		if (input_vector[i] < min) { min = input_vector[i]; }
		if (input_vector[i] > max) { max = input_vector[i]; }
	}

	bit_width   = count_bits<ST, UT>(max, min);
	base_for[0] = min;
}

template struct encoder<double>;
template struct encoder<float>;
} // namespace alp