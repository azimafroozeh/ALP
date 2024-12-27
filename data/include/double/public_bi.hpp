#ifndef DOUBLE_PUBLIC_BI_HPP
#define DOUBLE_PUBLIC_BI_HPP

#include "column.hpp"

namespace alp_bench {

inline auto get_public_bi_dataset() {
	static std::array<ALPColumnDescriptor, 1> PUBLIC_BI = {{
		{0, "public_bi_CityMaxCapita_table_1_column_4.csv", ALP_CMAKE_SOURCE_DIR "/data/public_bi/CityMaxCapita_table_1_column_4.csv", "", 0, 0, 0, 0},
	}};

	return PUBLIC_BI;
}

} // namespace alp_bench
#endif