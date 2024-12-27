#include "alp.hpp"
#include "data.hpp"
#include "gtest/gtest.h"
#include <fstream>

/// ALP encoded size per vector  = bit_width + factor-idx + exponent-idx + ffor base;
double overhead_per_vector {static_cast<double>(8 + 8 + 8 + 64) / alp::config::VECTOR_SIZE};

///  ALP_RD Overhead encoded size
double alprd_overhead_per_vector {static_cast<double>(alp::config::MAX_RD_DICTIONARY_SIZE * 16) /
                                  alp::config::ROWGROUP_SIZE};

namespace test {
template <typename T>
void ALP_ASSERT(T original_val, T decoded_val) {
	if (original_val == 0.0 && std::signbit(original_val)) {
		ASSERT_EQ(decoded_val, 0.0);
		ASSERT_TRUE(std::signbit(decoded_val));
	} else if (std::isnan(original_val)) {
		ASSERT_TRUE(std::isnan(decoded_val));
	} else {
		ASSERT_EQ(original_val, decoded_val);
	}
}

} // namespace test
class alp_test : public ::testing::Test {
public:
	double* intput_buf {};
	double* exception_buf {};
	double* decoded_buf {};
	double* sample_buf {};
	double* glue_buf {};

	uint16_t* rd_exc_arr {};
	uint16_t* pos_arr {};
	int64_t*  ffor_buf {};
	int64_t*  unffor_arr {};
	int64_t*  base_buf {};
	int64_t*  encoded_buf {};

	uint64_t* ffor_right_buf {};
	uint16_t* ffor_left_arr {};
	uint64_t* right_buf {};
	uint16_t* left_arr {};
	uint64_t* unffor_right_buf {};
	uint16_t* unffor_left_arr {};

	alp::bw_t bit_width {};

	template <typename PT>
	void read_file_into_vector(const std::string& file_path, std::vector<PT>& input_arr) {
		std::ifstream file(file_path);
		if (!file.is_open()) { throw std::runtime_error("Failed to open file"); }

		std::string val_str;
		size_t      row_idx = 0;

		while (std::getline(file, val_str)) { // Use std::getline to ensure reading line by line
			if (!val_str.empty()) {           // Skip empty lines if any
				PT value_to_encode;
				if constexpr (std::is_same_v<PT, double>) {
					value_to_encode = std::stod(val_str);
				} else if constexpr (std::is_same_v<PT, float>) {
					value_to_encode = std::stof(val_str);
				} else {
					throw std::invalid_argument("Unsupported type");
				}
				if (row_idx >= input_arr.size()) {
					input_arr.resize(row_idx + 1); // Expand the array dynamically if needed
				}
				input_arr[row_idx] = value_to_encode;
				row_idx += 1;
			}
		}
	}

	void SetUp() override {
		intput_buf    = new double[alp::config::VECTOR_SIZE];
		sample_buf    = new double[alp::config::VECTOR_SIZE];
		exception_buf = new double[alp::config::VECTOR_SIZE];
		decoded_buf   = new double[alp::config::VECTOR_SIZE];
		glue_buf      = new double[alp::config::VECTOR_SIZE];
		//

		right_buf        = new uint64_t[alp::config::VECTOR_SIZE];
		ffor_right_buf   = new uint64_t[alp::config::VECTOR_SIZE];
		unffor_right_buf = new uint64_t[alp::config::VECTOR_SIZE];

		//
		encoded_buf = new int64_t[alp::config::VECTOR_SIZE];
		base_buf    = new int64_t[alp::config::VECTOR_SIZE];
		ffor_buf    = new int64_t[alp::config::VECTOR_SIZE];

		//
		rd_exc_arr      = new uint16_t[alp::config::VECTOR_SIZE];
		pos_arr         = new uint16_t[alp::config::VECTOR_SIZE];
		unffor_arr      = new int64_t[alp::config::VECTOR_SIZE];
		left_arr        = new uint16_t[alp::config::VECTOR_SIZE];
		ffor_left_arr   = new uint16_t[alp::config::VECTOR_SIZE];
		unffor_left_arr = new uint16_t[alp::config::VECTOR_SIZE];
	}

	~alp_test() override {
		delete[] intput_buf;
		delete[] sample_buf;
		delete[] exception_buf;
		delete[] rd_exc_arr;
		delete[] pos_arr;
		delete[] encoded_buf;
		delete[] decoded_buf;
		delete[] ffor_buf;
		delete[] unffor_arr;
		delete[] base_buf;
		delete[] right_buf;
		delete[] left_arr;
		delete[] unffor_right_buf;
		delete[] unffor_left_arr;
	}

	template <typename PT>
	void test_column(const alp_bench::ALPColumnDescriptor& column) {
		using UT = typename alp::inner_t<PT>::ut;
		using ST = typename alp::inner_t<PT>::st;

		auto* sample_arr       = reinterpret_cast<PT*>(sample_buf);
		auto* right_arr        = reinterpret_cast<UT*>(right_buf);
		auto* ffor_right_arr   = reinterpret_cast<UT*>(ffor_right_buf);
		auto* unffor_right_arr = reinterpret_cast<UT*>(unffor_right_buf);
		auto* glue_arr         = reinterpret_cast<PT*>(glue_buf);
		auto* exc_arr          = reinterpret_cast<PT*>(exception_buf);
		auto* dec_dbl_arr      = reinterpret_cast<PT*>(decoded_buf);
		auto* encoded_arr      = reinterpret_cast<ST*>(encoded_buf);
		auto* base_arr         = reinterpret_cast<ST*>(base_buf);
		auto* ffor_arr         = reinterpret_cast<ST*>(ffor_buf);

		alp::state<PT> stt;
		size_t         tuples_count {alp::config::VECTOR_SIZE};

		std::vector<PT> data;
		read_file_into_vector<PT>(column.csv_file_path, data);
		const auto*     input_arr = data.data();

		// Init
		alp::encoder<PT>::init(input_arr, tuples_count, sample_arr, stt);

		switch (stt.scheme) {
		case alp::Scheme::ALP_RD: {
			alp::rd_encoder<PT>::init(input_arr, tuples_count, sample_arr, stt);

			alp::rd_encoder<PT>::encode(input_arr, rd_exc_arr, pos_arr, right_arr, left_arr, stt);
			ffor::ffor(right_arr, ffor_right_arr, stt.right_bit_width, &stt.right_for_base);
			ffor::ffor(left_arr, ffor_left_arr, stt.left_bit_width, &stt.left_for_base);

			// Decode
			unffor::unffor(ffor_right_arr, unffor_right_arr, stt.right_bit_width, &stt.right_for_base);
			unffor::unffor(ffor_left_arr, unffor_left_arr, stt.left_bit_width, &stt.left_for_base);
			alp::rd_encoder<PT>::decode(glue_arr, unffor_right_arr, unffor_left_arr, rd_exc_arr, pos_arr, stt);

			for (size_t i = 0; i < alp::config::VECTOR_SIZE; ++i) {
				auto l = input_arr[i];
				auto r = glue_arr[i];
				if (l != r) { std::cout << i << " | " << i << " r : " << r << " l : " << l << '\n'; }
				test::ALP_ASSERT(r, l);
			}

			break;
		}
		case alp::Scheme::ALP: {
			// Encode
			alp::encoder<PT>::encode(input_arr, exc_arr, pos_arr, encoded_arr, stt);
			alp::encoder<PT>::analyze_ffor(encoded_arr, bit_width, base_arr);
			ffor::ffor(encoded_arr, ffor_arr, bit_width, base_arr);

			// Decode
			generated::falp::fallback::scalar::falp(ffor_arr, dec_dbl_arr, bit_width, base_arr, stt.fac, stt.exp);
			alp::decoder<PT>::patch_exceptions(dec_dbl_arr, exc_arr, pos_arr, stt);

			// validation
			for (size_t i = 0; i < alp::config::VECTOR_SIZE; ++i) {
				test::ALP_ASSERT(input_arr[i], dec_dbl_arr[i]);
			}

			ASSERT_EQ(column.exceptions_count, stt.n_exceptions);
			ASSERT_EQ(column.bit_width, bit_width);
		}
		default:;
		}

		std::cout << "\033[32m-- " << column.name << '\n';
	}
};

/// Test used for correctness of bitwidth and exceptions on the first vector of each dataset
TEST_F(alp_test, test_alp_double) {
	for (const auto& col : alp_bench::get_alp_dataset()) {
		test_column<double>(col);
	}
}

/// Test used for correctness of bitwidth and exceptions on the first vector of generated data
TEST_F(alp_test, test_alp_on_generated) {
	for (const auto& col : alp_bench::get_generated_cols()) {
		test_column<double>(col);
	}
}

// Test used for correctness of bitwidth and exceptions on the first vector of edge_case data
TEST_F(alp_test, test_alp_on_edge_case) {
	for (const auto& col : alp_bench::get_edge_case()) {
		test_column<double>(col);
	}
}

TEST_F(alp_test, alp_float_test_dataset) {
	for (const auto& col : alp_bench::get_float_test_dataset()) {
		test_column<float>(col);
	}
}

TEST_F(alp_test, alp_double_test_dataset) {
	for (const auto& col : alp_bench::get_double_test_dataset()) {
		test_column<double>(col);
	}
}

TEST_F(alp_test, test_alp_float_on_edge_cases) {
	for (const auto& col : alp_bench::get_float_edge_case()) {
		test_column<float>(col);
	}
}
