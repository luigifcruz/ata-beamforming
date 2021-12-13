#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/memory/types.hh"

namespace Blade {

template<typename I, typename T> class Vector;

template<typename T>
class VectorImpl {
 public:
    VectorImpl()
             : container(),
               managed(false) {}
    explicit VectorImpl(const std::span<T>& other)
             : container(other),
               managed(false) {}
    explicit VectorImpl(T* ptr, const std::size_t& size)
             : container(ptr, size),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() {}

    constexpr T* data() const noexcept {
        return container.data();
    }

    constexpr std::size_t size() const noexcept {
        return container.size();
    }

    constexpr std::size_t size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return container.empty();
    }

    constexpr T& operator[](std::size_t idx) const {
        return container[idx];
    }

    virtual Result resize(const std::size_t& size) = 0;

 protected:
    std::span<T> container;
    bool managed;
};

}  // namespace Blade

#endif