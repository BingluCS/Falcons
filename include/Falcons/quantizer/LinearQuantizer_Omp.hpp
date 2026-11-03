/**
 * @file LinearQuantizer_Omp.hpp
 * @author Bing Lu
 * @brief Falcons parallel quantization and per-thread outlier handling.
 *
 * Copyright (c) 2026 Bing Lu.
 */

#ifndef Falcons_LINEAR_QUANTIZER_OMP_HPP
#define Falcons_LINEAR_QUANTIZER_OMP_HPP
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include "Falcons/def.hpp"
#include "Falcons/utils/Config.hpp"
#include "Falcons/quantizer/Quantizer_Omp.hpp"
#include "Falcons/utils/BufferBounds.hpp"
#include "Falcons/utils/MemoryUtil.hpp"

namespace Falcons {

template <class T>
class LinearQuantizerOMP : public concepts::QuantizerOMPInterface<T, int> {
   public:
    LinearQuantizerOMP() : error_bound(1), double_error_bound(2), double_error_bound_reciprocal(0.5), radius(32768) {}

    LinearQuantizerOMP(double eb, int r = 32768) : error_bound(eb),double_error_bound(2*eb), double_error_bound_reciprocal(0.5 / eb), radius(r) {
        assert(eb != 0);
    }

    LinearQuantizerOMP(const LinearQuantizerOMP &) = delete;
    LinearQuantizerOMP &operator=(const LinearQuantizerOMP &) = delete;
    LinearQuantizerOMP(LinearQuantizerOMP &&) = default;
    LinearQuantizerOMP &operator=(LinearQuantizerOMP &&) = default;

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
    ALWAYS_INLINE int quantize_and_overwrite(T &data, T pred, size_t data_idx) override {

        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            //if (diff < 0)
            //    quant_index = -quant_index;
            //auto quant_index_shifted = this->radius + quant_index;
            T decompressed_data = pred + quant_index * this->double_error_bound;
            // if data is NaN, the error is NaN, and NaN <= error_bound is false
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                data = decompressed_data;
                return quant_index;                 // int16: no +radius
            } else {
                save_unpred(data, data_idx);
                return -this->radius;                      // int16 escape (INT16_MIN)
            }
        } else {
            save_unpred(data, data_idx);
            return -this->radius;
        }
    }

    ALWAYS_INLINE int quantize_and_overwrite2(T &data, T pred, size_t tid) {

        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            //if (diff < 0)
            //    quant_index = -quant_index;
            //auto quant_index_shifted = this->radius + quant_index;
            T decompressed_data = pred + quant_index * this->double_error_bound;
            // if data is NaN, the error is NaN, and NaN <= error_bound is false
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                data = decompressed_data;
                return quant_index;                 // int16: no +radius
            } else {
                save_unpred2(data, tid);
                return -this->radius;
            }
        } else {
            save_unpred2(data, tid);
            return -this->radius;
        }
    }

    ALWAYS_INLINE int quantize_without_overwrite2(T data, T pred, size_t tid) {
        T diff = data - pred;
        int quant_index = static_cast<int>(std::nearbyint(diff * this->double_error_bound_reciprocal));
        if (quant_index > -this->radius && quant_index < this->radius ) {
            T decompressed_data = pred + quant_index * this->double_error_bound;
            T err = decompressed_data - data;
            if (err >= -this->error_bound && err <= this->error_bound) {
                return quant_index;
            } else {
                save_unpred2(data, tid);
                return -this->radius;
            }
        } else {
            save_unpred2(data, tid);
            return -this->radius;
        }
    }

    // recover the data using the quantization index
    ALWAYS_INLINE T recover(T pred, int quant_index) override {
        if (quant_index != -this->radius) {                // int16: predictable unless escape (-this->radius)
            return recover_pred(pred, quant_index);
        } else {
            // return recover_unpred();
            return T(0);
        }
    }


    ALWAYS_INLINE T recover_pred(T pred, int quant_index) {
        return pred + quant_index * this->double_error_bound;   // int16: no -radius
    }

    ALWAYS_INLINE T recover_unpred() { return unpred[index++]; }
    ALWAYS_INLINE T recover_unpred2(int tid) { return local_unpred[tid][local_unpred_idx[tid].value++]; }
    size_t local_unpred_count(int tid) const { return static_cast<size_t>(local_unpred_idx[tid].value); }
    void release_local_unpred() noexcept {
        decoded_unpred_owner.reset();
        local_unpred_idx_owner.reset();
        local_unpred_owner.reset();
        unpred_vec = nullptr;
        local_unpred = nullptr;
        local_unpred_idx = nullptr;
        local_thread_count = 0;
    }
    ALWAYS_INLINE void init_local_unpred(int thread_count, int each_num) {
        #ifdef _OPENMP
            static_assert(std::is_trivially_destructible_v<T>, "OMP values must be trivially destructible");
            if (thread_count <= 0 || each_num < 0)
                throw std::invalid_argument("Invalid local outlier capacity");
            const size_t alignment = 64;
            const size_t elems_per_cacheline = alignment / sizeof(T);
            size_t padded_each_num =
                buffer_bounds::mul(buffer_bounds::add(static_cast<size_t>(each_num), elems_per_cacheline - 1) /
                                   elems_per_cacheline, elems_per_cacheline);
            const size_t total = buffer_bounds::mul(static_cast<size_t>(thread_count), padded_each_num);
            release_local_unpred();
            if (each_num > 0) {
                unpred_vec = new (std::align_val_t(64)) T[total];
                local_unpred = new T*[thread_count];
                local_unpred_idx = new CacheLineInt[thread_count];
            } else {
                local_unpred_owner.reset(new T*[thread_count]());
                local_unpred_idx_owner.reset(new CacheLineInt[thread_count]);
                decoded_unpred_owner.reset(new std::unique_ptr<T[]>[thread_count]);
                local_unpred = local_unpred_owner.get();
                local_unpred_idx = local_unpred_idx_owner.get();
            }
            local_thread_count = thread_count;
            #pragma omp parallel for
            for (int tid = 0; tid < thread_count; ++tid) {
                local_unpred[tid] = each_num > 0 ? unpred_vec + tid * padded_each_num : nullptr;
                local_unpred_idx[tid].value = 0;
            }
        #endif
    }
    // Reset per-thread unpred READ cursors to 0 before decompression. Needed because
    // save_unpred2 (compress) and recover_unpred2 (decompress) share local_unpred_idx as
    // their cursor; after compress it points past the last written outlier. (The non-OMP
    // quantizer avoids this by using a separate decompress-only `index`.)
    ALWAYS_INLINE void reset_local_unpred_idx(int thread_count) {
        #ifdef _OPENMP
            for (int tid = 0; tid < thread_count; ++tid) local_unpred_idx[tid].value = 0;
        #endif
    }
    ALWAYS_INLINE T recover2(T pred, int quant_index, int tid) {
        if (quant_index != -this->radius) {                // int16: predictable unless escape (-this->radius); code 0 = zero error
            return recover_pred(pred, quant_index);
        } else {
            return recover_unpred2(tid);
            // return T(0);
        }
    }
    ALWAYS_INLINE void aggregate_local_unpred() {
        // #ifdef _OPENMP
        //     for(size_t i = 0; i < 1; i++){
        //         unpred.insert(unpred.end(), local_unpred[i], local_unpred[i] + local_unpred_idx[i].value);
        //     }
        //     // local_unpred.clear();
        // #endif
    }
    ALWAYS_INLINE int save_unpred(T ori, size_t data_idx){
        #ifdef _OPENMP
        #pragma omp critical
        {
            unpred.push_back(ori);
            unpred_idx.push_back(data_idx);
        }
        #else
            unpred.push_back(ori);
            unpred_idx.push_back(data_idx);
        #endif
        return 0;
    }

    ALWAYS_INLINE int save_unpred2(T ori, int tid){
        //local_unpred[tid].emplace_back(ori);
        local_unpred[tid][local_unpred_idx[tid].value++] = ori;
        return 0;
    }

    void unpack_unpred(T *data) const override {

        #ifdef _OPENMP
           #pragma omp parallel for
        #endif
        for(size_t i = 0; i < unpred.size(); i++)
            data[unpred_idx[i]] = unpred[i];

    }



    size_t size_est() { return unpred.size() * sizeof(T); }

    void save(unsigned char *&c) const override {
        write(uid, c);
        write(this->error_bound, c);
        write(this->radius, c);
        size_t unpred_size = unpred.size();
        write(unpred_size, c);
        if (unpred_size > 0) {
            assert (unpred_size == unpred_idx.size());
            write(unpred.data(), unpred_size, c);
            write(unpred_idx.data(), unpred_size, c);
        }
    }

    void save2(unsigned char *&c) const {
        write(uid, c);
        write(this->error_bound, c);
        write(this->radius, c);
        // write( local_unpred_idx[0].value, c);
        // if ( local_unpred_idx[0].value > 0) {
        //     write(local_unpred[0], local_unpred_idx[0].value, c);
        // }
    }

    void save3(unsigned char *&c, int tid) const {
        // write(uid, c);
        // write(this->error_bound, c);
        // write(this->radius, c);
        // std::cout << "local_unpred_idx[tid].value: " << local_unpred_idx[tid].value << std::endl;
        write( local_unpred_idx[tid].value, c);
        if ( local_unpred_idx[tid].value > 0) {
            write(local_unpred[tid], local_unpred_idx[tid].value, c);
        }
    //     std::cout << "loaded local unpred size: " << local_unpred_idx[tid].value << " for tid " << tid << std::endl;
    //     std::string filename = "unpred_decmp-" + std::to_string(tid);
    //     std::ofstream fout(filename, std::ios::binary);
    //     fout.write(reinterpret_cast<const char*>(local_unpred[tid]), sizeof(T) * local_unpred_idx[tid].value);
    //     fout.close();
    }

    void load(const unsigned char *&c, size_t &remaining_length) override {
        uchar uid_read;
        read(uid_read, c, remaining_length);
        if (uid_read != uid) {
            throw std::invalid_argument("LinearQuantizer uid mismatch");
        }
        double eb;
        read(eb, c, remaining_length);
        set_eb(eb);
        read(this->radius, c, remaining_length);
        size_t unpred_size = 0;
        read(unpred_size, c, remaining_length);
        if (unpred_size > 0) {
            unpred.resize(unpred_size);
            read(unpred.data(), unpred_size, c, remaining_length);
            unpred_idx.resize(unpred_size);
            read(unpred_idx.data(), unpred_size, c, remaining_length);
        }
        index = 0;
    }

    void load2(const unsigned char *&c, size_t &remaining_length) {
        if (remaining_length < sizeof(uchar) + sizeof(double) + sizeof(int))
            throw std::invalid_argument("Truncated LinearQuantizer header");
        uchar uid_read;
        read(uid_read, c, remaining_length);
        if (uid_read != uid) {
            throw std::invalid_argument("LinearQuantizer uid mismatch");
        }
        double eb;
        read(eb, c, remaining_length);
        set_eb(eb);
        read(this->radius, c, remaining_length);
        // local_unpred = new T*[1];
        // local_unpred_idx = new CacheLineInt[1];
        // read(local_unpred_idx[0].value, c, remaining_length);
        // if (local_unpred_idx[0].value > 0) {
        //     local_unpred[0] = new T[local_unpred_idx[0].value];
        //     read(local_unpred[0], local_unpred_idx[0].value, c, remaining_length);
        //     local_unpred_idx[0].value = 0;
        // }
        // index = 0;
    }

    void load3(const unsigned char *&c, size_t &remaining_length, int tid) {
        if (tid < 0 || tid >= local_thread_count || !decoded_unpred_owner)
            throw std::invalid_argument("Invalid LinearQuantizer outlier stream");
        const unsigned char *pos = c;
        size_t remaining = remaining_length;
        int count = 0;
        if (remaining < sizeof(count))
            throw std::invalid_argument("Truncated LinearQuantizer outlier count");
        read(count, pos, remaining);
        if (count < 0 || static_cast<size_t>(count) > remaining / sizeof(T))
            throw std::invalid_argument("Invalid LinearQuantizer outlier count");
        std::unique_ptr<T[]> decoded;
        if (count > 0) {
            decoded.reset(new T[count]);
            read(decoded.get(), static_cast<size_t>(count), pos, remaining);
        }
        decoded_unpred_owner[tid] = std::move(decoded);
        local_unpred[tid] = decoded_unpred_owner[tid].get();
        local_unpred_idx[tid].value = 0;
        c = pos;
        remaining_length = remaining;
    }

    void print() override {
        printf("[LinearQuantizer] error_bound = %.8G, radius = %d, unpred = %zu\n", error_bound, radius, unpred.size());
    }
   private:
    std::vector<T> unpred;
    T* unpred_vec = nullptr;
    T** local_unpred = nullptr;
    CacheLineInt* local_unpred_idx = nullptr;
    std::unique_ptr<T*[]> local_unpred_owner;
    std::unique_ptr<CacheLineInt[]> local_unpred_idx_owner;
    std::unique_ptr<std::unique_ptr<T[]>[]> decoded_unpred_owner;
    int local_thread_count = 0;
    std::vector<size_t> unpred_idx;
    size_t index = 0;  // used in decompression only
    uchar uid = 0b11;

    double error_bound;
    double double_error_bound;
    double double_error_bound_reciprocal;
    int radius;  // quantization interval radius
};

}  // namespace Falcons
#endif
