#include "blade/modules/beamformer/ata.hh"
#include "blade/utils/checker.hh"

using namespace Blade;

template<typename IT, typename OT>
class Test : public Pipeline {
 public:
    typedef typename Modules::Beamformer::ATA<IT, OT>::Config Config;

    explicit Test(const Config& config) {
        this->connect(beamformer, config, {input, phasors});
    }

    constexpr const std::size_t getInputSize() const {
        return beamformer->getInputSize();
    }

    constexpr const std::size_t getPhasorsSize() const {
        return beamformer->getPhasorsSize();
    }

    constexpr const std::size_t getOutputSize() const {
        return beamformer->getOutputSize();
    }

    Result run(const Vector<Device::CPU, IT>& input,
               const Vector<Device::CPU, IT>& phasors,
                     Vector<Device::CPU, OT>& output,
               const bool synchronize = false) {
        BL_CHECK(this->copy(beamformer->getInput(), input));
        BL_CHECK(this->compute());
        BL_CHECK(this->copy(output, beamformer->getOutput()));

        if (synchronize) {
            BL_CHECK(this->synchronize());
        }

        return Result::SUCCESS;
    }

 private:
    Vector<Device::CUDA, IT> input;
    Vector<Device::CUDA, IT> phasors;
    std::shared_ptr<Modules::Beamformer::ATA<IT, OT>> beamformer;
};
