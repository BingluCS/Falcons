/**
 * @file LinearQuantizer.hpp
 * @author Bing Lu
 * @brief Falcons signed-code quantization and reconstruction.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_LINEAR_QUANTIZER_HPP
#define Falcons_LINEAR_QUANTIZER_HPP

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include "Falcons/def.hpp"
#include "Falcons/quantizer/Quantizer.hpp"
#include "Falcons/utils/MemoryUtil.hpp"

namespace Falcons {

template <class T>
class LinearQuantizer : public concepts::QuantizerInterface<T, int> {
   public:
    LinearQuantizer() : error_bound(1), double_error_bound(2), double_error_bound_reciprocal(0.5), radius(32768) {}

    LinearQuantizer(double eb, int r = 32768) : error_bound(eb),double_error_bound(2*eb), double_error_bound_reciprocal(0.5 / eb), radius(r) {
        assert(eb != 0);
    }

    double get_eb() const { return error_bound; }

    ALWAYS_INLINE std::tuple<double, double, double> get_all_eb() const { return {error_bound, double_error_bound, double_error_bound_reciprocal}; }

    void set_eb(double eb) {
        error_bound = eb;
        double_error_bound = 2 * eb;
        double_error_bound_reciprocal = 1.0 / double_error_bound;
    }

    std::pair<int, int> get_out_range() const override { return std::make_pair(0, radius * 2); }

    // quantize the data with a prediction value, and returns the quantization index and the decompressed data
    // int quantize(T data, T pred, T& dec_data);
     ALWAYS_INLINE int quantize_and_overwrite(T &data, T pred) override {
        if constexpr (std::is_integral_v<T>) {
            int quant_index;
            T decompressed_data;
            if (quantize_integral(data, pred, quant_index, decompressed_data)) {
                data = decompressed_data;
                return quant_index;
            }
            unpred.emplace_back(data);
            return -this->radius;
        }

        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            //if (diff < 0)
            //    quant_index = -quant_index;
            //auto quant_index_shifted = this->radius + quant_index;
            // fused, f64-intermediate dequant to match the SIMD (svmla / fmadd) paths bit-for-bit
            T decompressed_data = static_cast<T>(std::fma(static_cast<double>(quant_index), this->double_error_bound, static_cast<double>(pred)));
            // if data is NaN, the error is NaN, and NaN <= error_bound is false
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                data = decompressed_data;
                return quant_index;                 // int16: store signed quant_index directly (no +radius)
            } else {
                unpred.emplace_back(data);
                return -this->radius;               // escape sentinel = -radius (tracks radius; default -32768)
            }
        } else {
            unpred.emplace_back(data);
            return -this->radius;
        }
    }

    ALWAYS_INLINE int quantize_without_overwrite(T data, T pred) {
        if constexpr (std::is_integral_v<T>) {
            int quant_index;
            T decompressed_data;
            if (quantize_integral(data, pred, quant_index, decompressed_data))
                return quant_index;
            unpred.emplace_back(data);
            return -this->radius;
        }

        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            T decompressed_data = static_cast<T>(std::fma(static_cast<double>(quant_index), this->double_error_bound, static_cast<double>(pred)));
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                return quant_index;
            } else {
                unpred.emplace_back(data);
                return -this->radius;
            }
        } else {
            unpred.emplace_back(data);
            return -this->radius;
        }
    }

    // recover the data using the quantization index
    ALWAYS_INLINE T recover(T pred, int quant_index) override {
        if (quant_index != -this->radius) {                // escape = -radius; other codes are signed quant errors
            return recover_pred(pred, quant_index);
        } else {
            return recover_unpred();
        }
    }

    ALWAYS_INLINE T recover_pred(T pred, int quant_index) {
        if constexpr (std::is_integral_v<T>) {
            if (!legacy_integral_stream) {
                T recovered;
                if (!recover_integral(pred, quant_index, recovered))
                    throw std::runtime_error("LinearQuantizer integral reconstruction overflow");
                return recovered;
            }
        }
        // must match quantize_and_overwrite's dequant exactly (fused, f64)
        return static_cast<T>(std::fma(static_cast<double>(quant_index), this->double_error_bound, static_cast<double>(pred)));
    }

    ALWAYS_INLINE T recover_unpred() { return unpred[index++]; }

    ALWAYS_INLINE int force_save_unpred(T ori) override {
        unpred.emplace_back(ori);
        return -this->radius;
    }

    size_t size_est() { return unpred.size() * sizeof(T); }

    void save(unsigned char *&c) const override {
        const uchar stream_uid = std::is_integral_v<T> ? integral_uid : uid;
        write(stream_uid, c);
        write(this->error_bound, c);
        write(this->radius, c);
        int unpred_size = unpred.size();
        write(unpred_size, c);
        if (unpred_size > 0) {
            write(unpred.data(), unpred.size(), c);
        }
    }

    void load(const unsigned char *&c, size_t &remaining_length) override {
        uchar uid_read;
        read(uid_read, c, remaining_length);
        if constexpr (std::is_integral_v<T>) {
            if (uid_read == integral_uid) {
                legacy_integral_stream = false;
            } else if (uid_read == uid) {
                legacy_integral_stream = true;
            } else {
                throw std::invalid_argument("LinearQuantizer uid mismatch");
            }
        } else {
            if (uid_read != uid)
                throw std::invalid_argument("LinearQuantizer uid mismatch");
        }
        double eb;
        read(eb, c, remaining_length);
        set_eb(eb);
        read(this->radius, c, remaining_length);
        int unpred_size = 0;
        read(unpred_size, c, remaining_length);
        if (unpred_size > 0) {
            unpred.resize(unpred_size);
            read(unpred.data(), unpred_size, c, remaining_length);
        }
        index = 0;
    }
    std::vector<T>& test_unpred() {
        return unpred;
    }

    void print() override {
        printf("[LinearQuantizer] error_bound = %.8G, radius = %d, unpred = %zu\n", error_bound, radius, unpred.size());
    }

   private:
    template <typename U>
    ALWAYS_INLINE static std::make_unsigned_t<U> integral_distance(U lhs, U rhs) {
        using Unsigned = std::make_unsigned_t<U>;
        const Unsigned ulhs = static_cast<Unsigned>(lhs);
        const Unsigned urhs = static_cast<Unsigned>(rhs);
        return lhs < rhs ? urhs - ulhs : ulhs - urhs;
    }

    template <typename U = T, typename std::enable_if_t<std::is_integral_v<U>, int> = 0>
    ALWAYS_INLINE bool recover_integral(U pred, int quant_index, U &recovered) const {
        if (quant_index == 0) {
            recovered = pred;
            return true;
        }
        if constexpr (std::is_unsigned_v<U>) {
            const unsigned magnitude_index = static_cast<unsigned>(
                quant_index < 0 ? -quant_index : quant_index);
            const double correction = static_cast<double>(magnitude_index) * double_error_bound;
            if (!std::isfinite(correction) || correction < 0)
                return false;
            const double rounded_correction = quant_index >= 0 ? std::floor(correction) : std::ceil(correction);
            const double unsigned_upper_bound = std::ldexp(1.0, std::numeric_limits<U>::digits);
            if (rounded_correction >= unsigned_upper_bound)
                return false;
            const U integer_correction = static_cast<U>(rounded_correction);
            if (quant_index >= 0) {
                if (integer_correction > std::numeric_limits<U>::max() - pred)
                    return false;
                recovered = static_cast<U>(pred + integer_correction);
            } else {
                if (integer_correction > pred)
                    return false;
                recovered = static_cast<U>(pred - integer_correction);
            }
            return true;
        }

        const double value = std::fma(static_cast<double>(quant_index), double_error_bound,
                                      static_cast<double>(pred));
        const double signed_upper_bound = std::ldexp(1.0, std::numeric_limits<U>::digits);
        if (!std::isfinite(value) || value < -signed_upper_bound || value >= signed_upper_bound)
            return false;
        recovered = static_cast<U>(value);
        return true;
    }

    template <typename U = T, typename std::enable_if_t<std::is_integral_v<U>, int> = 0>
    ALWAYS_INLINE bool quantize_integral(U data, U pred, int &quant_index, U &decompressed_data) const {
        const bool negative = data < pred;
        const double rounded = std::nearbyint(
            static_cast<double>(integral_distance(data, pred)) * double_error_bound_reciprocal);
        if (!std::isfinite(rounded) || rounded < 0 || rounded >= radius)
            return false;
        const int magnitude_index = static_cast<int>(rounded);
        quant_index = negative ? -magnitude_index : magnitude_index;
        if (!recover_integral(pred, quant_index, decompressed_data))
            return false;
        return static_cast<double>(integral_distance(data, decompressed_data)) <= error_bound;
    }

    std::vector<T> unpred;
    size_t index = 0;  // used in decompression only
    static constexpr uchar uid = 0b10;
    static constexpr uchar integral_uid = 0b100;
    bool legacy_integral_stream = false;

    double error_bound;
    double double_error_bound;
    double double_error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace Falcons
#endif
