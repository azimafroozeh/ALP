#ifndef ALP_SAMPLER_HPP
#define ALP_SAMPLER_HPP

#include "alp/config.hpp"
#include <algorithm>
#include <cmath>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"

namespace alp::sampler {

template <class PT>
uint64_t first_level_sample(const PT* data_p, const uint64_t n_values, PT* sample_arr_p) {
	const uint64_t portion_to_sample     = std::min(config::ROWGROUP_SIZE, n_values);
	const uint64_t available_alp_vectors = std::ceil(static_cast<double>(portion_to_sample) / config::VECTOR_SIZE);
	uint64_t       sample_idx            = 0;
	uint64_t       data_idx              = 0;

	for (uint64_t vector_idx = 0; vector_idx < available_alp_vectors; vector_idx++) {
		const uint64_t n_values_in_cur_vector = std::min(n_values - data_idx, config::VECTOR_SIZE);

		//! We sample equidistant vectors; to do this we skip a fixed values of vectors
		//! If we are not in the correct jump, we do not take sample from this vector
		if (const bool must_select_rowgroup_sample = (vector_idx % config::ROWGROUP_SAMPLES_JUMP) == 0;
		    !must_select_rowgroup_sample) {
			data_idx += n_values_in_cur_vector;
			continue;
		}

		const uint64_t n_sampled_increments = std::max(
		    1,
		    static_cast<int32_t>(std::ceil(static_cast<double>(n_values_in_cur_vector) / config::SAMPLES_PER_VECTOR)));

		//! We do not take samples of non-complete duckdb vectors (usually the last one)
		//! Except in the case of too little data
		if (n_values_in_cur_vector < config::SAMPLES_PER_VECTOR && sample_idx != 0) {
			data_idx += n_values_in_cur_vector;
			continue;
		}

		// Storing the sample of that vector
		for (uint64_t i = 0; i < n_values_in_cur_vector; i += n_sampled_increments) {
			sample_arr_p[sample_idx] = data_p[data_idx + i];
			sample_idx++;
		}
		data_idx += n_values_in_cur_vector;
	}
	return sample_idx;
}

} // namespace alp::sampler

#pragma GCC diagnostic pop

#endif
