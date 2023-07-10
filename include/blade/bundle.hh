#ifndef BLADE_BUNDLE_HH
#define BLADE_BUNDLE_HH

#include <span>
#include <string>
#include <memory>
#include <vector>

#include "blade/logger.hh"
#include "blade/module.hh"

namespace Blade {

class BLADE_API Bundle {
 public:
    Bundle(const cudaStream_t& stream) : stream(stream) {};

 protected:
    constexpr std::vector<std::shared_ptr<Module>>& getModules() {
        return modules;
    }

    template<typename Block>
    void connect(std::shared_ptr<Block>& module,
                 const typename Block::Config& config,
                 const typename Block::Input& input) {
        module = std::make_unique<Block>(config, input, stream);
        this->modules.push_back(module);
    }

    friend class Pipeline;

 private:
    const cudaStream_t stream;
    std::vector<std::shared_ptr<Module>> modules;
};

}  // namespace Blade

#endif
