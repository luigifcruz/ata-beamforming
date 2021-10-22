#ifndef BLADE_CHANNELIZER_TEST_GENERIC_H
#define BLADE_CHANNELIZER_TEST_GENERIC_H

#include "blade/base.hh"
#include "blade/python.hh"
#include "blade/channelizer/base.hh"

namespace Blade::Channelizer::Test {

class BLADE_API Generic : protected Python {
public:
    Generic(const Channelizer::Generic::Config & config);
    ~Generic() = default;

    Result process();

    std::span<const std::complex<int8_t>> getInputData();
    std::span<const std::complex<int8_t>> getOutputData();
};

} // namespace Blade::Channelizer::Test

#endif