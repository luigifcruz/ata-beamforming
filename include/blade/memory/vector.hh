#ifndef BLADE_MEMORY_VECTOR_HH
#define BLADE_MEMORY_VECTOR_HH

#include "blade/memory/types.hh"

namespace Blade {

template <typename T>
concept IsDimensions = 
requires(T t) {
    { t.size() } -> std::same_as<U64>;
    { t == t } -> std::same_as<BOOL>;
};

template<typename Type, typename Dims>
requires IsDimensions<Dims>
class VectorImpl {
 public:
    VectorImpl()
             : dimensions(),
               container(),
               managed(true) {}
    explicit VectorImpl(const std::span<Type>& other, const Dims& dims)
             : dimensions(dims),
               container(other),
               managed(false) {
        if (dims.size() != other.size()) {
            BL_FATAL("Container size ({}) doesn't match dimensions size ({})",
                     other.size(), dims.size());
            BL_CHECK_THROW(Result::ERROR);
        }
    }
    explicit VectorImpl(const std::span<Type>& other)
             : dimensions({other.size()}),
               container(other),
               managed(false) {}
    explicit VectorImpl(Type* ptr, const Dims& dims)
             : dimensions(dims),
               container(ptr, dims.size()),
               managed(false) {}
    explicit VectorImpl(void* ptr, const Dims& dims)
             : dimensions(dims),
               container(static_cast<Type*>(ptr), dims.size()),
               managed(false) {}

    VectorImpl(const VectorImpl&) = delete;
    VectorImpl(const VectorImpl&&) = delete;
    bool operator==(const VectorImpl&) = delete;
    VectorImpl& operator=(const VectorImpl&) = delete;

    virtual ~VectorImpl() {}

    constexpr Type* data() const noexcept {
        return container.data();
    }

    constexpr const U64 size() const noexcept {
        return container.size();
    }

    constexpr const U64 size_bytes() const noexcept {
        return container.size_bytes();
    }

    [[nodiscard]] constexpr const bool empty() const noexcept {
        return container.empty();
    }

    constexpr Type& operator[](U64 idx) {
        return container[idx];
    }

    constexpr const Type& operator[](U64 idx) const {
        return container[idx];
    }

    constexpr auto begin() {
        return container.begin();
    }

    constexpr auto end() {
        return container.end();
    }

    constexpr const auto begin() const {
        return container.begin();
    }

    constexpr const auto end() const {
        return container.end();
    }

    const Result link(const VectorImpl<Type, Dims>& src, Dims dstDims) {
        if (src.empty()) {
            BL_FATAL("Source can't be empty while linking.");
            return Result::ERROR;
        }

        if (!this->empty()) {
            BL_FATAL("Destination has to be empty while linking.");
            return Result::ERROR;
        }

        if (src.dims().size() != dstDims.size()) { 
            BL_FATAL("Dimensions mismatch. The number of elements of "
                     "the source and destination has to be the same.");
            return Result::ERROR;
        }

        this->managed = false;
        this->container = src.span();
        this->dimensions = dstDims;

        return Result::SUCCESS;
    }

    const Result link(const VectorImpl<Type, Dims>& src) {
        return link(src, src.dims());
    }

    virtual const Result resize(const Dims& dims) = 0;

    constexpr const Dims& dims() const {
        return dimensions;
    }

 protected:
    Dims dimensions;
    std::span<Type> container;
    bool managed;

    explicit VectorImpl(const Dims& dims)
             : dimensions(dims),
               container(),
               managed(true) {
        if (dims.size() <= 0) {
            BL_FATAL("Dimensions ({}) equals to invalid size ({}).", dims, dims.size());
            BL_CHECK_THROW(Result::ERROR);
        }
    }

    constexpr const std::span<Type>& span() const {
        return container;
    }
};

}  // namespace Blade

#endif
