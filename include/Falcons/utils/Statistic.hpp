#ifndef Falcons_STATISTIC_HPP
#define Falcons_STATISTIC_HPP

#include "Config.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_FEATURE_SVE2
#include <arm_sve.h>
#endif

namespace Falcons {

template <class T>
ALWAYS_INLINE double statistic_difference(T lhs, T rhs) {
    if constexpr (std::is_integral_v<T>) {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned ulhs = static_cast<Unsigned>(lhs);
        const Unsigned urhs = static_cast<Unsigned>(rhs);
        const Unsigned magnitude = lhs < rhs ? urhs - ulhs : ulhs - urhs;
        const double difference = static_cast<double>(magnitude);
        return lhs < rhs ? -difference : difference;
    }
    return static_cast<double>(lhs) - static_cast<double>(rhs);
}

template <class T>
inline void scalar_minmax(const T *data, size_t begin, size_t end, T &out_min, T &out_max) {
    out_min = out_max = data[begin];
    for (size_t i = begin + 1; i < end; ++i) {
        out_max = std::max(out_max, data[i]);
        out_min = std::min(out_min, data[i]);
    }
}

#ifdef __ARM_FEATURE_SVE2
template <class T>
inline void sve2_minmax(const T *data, size_t begin, size_t end, T &out_min, T &out_max) {
    if constexpr (std::is_same_v<T, float>) {
        const size_t vl = svcntw();
        const svbool_t pg = svptrue_b32();
        svfloat32_t vmax = svdup_f32(data[begin]);
        svfloat32_t vmin = svdup_f32(data[begin]);

        size_t i = begin;
        for (; i + vl <= end; i += vl) {
            const svfloat32_t v = svld1_f32(pg, data + i);
            vmax = svmax_f32_x(pg, vmax, v);
            vmin = svmin_f32_x(pg, vmin, v);
        }

        out_max = svmaxv_f32(pg, vmax);
        out_min = svminv_f32(pg, vmin);
        for (; i < end; ++i) {
            out_max = std::max(out_max, data[i]);
            out_min = std::min(out_min, data[i]);
        }
    }
    else if constexpr (std::is_same_v<T, double>) {
        const size_t vl = svcntd();
        const svbool_t pg = svptrue_b64();
        svfloat64_t vmax = svdup_f64(data[begin]);
        svfloat64_t vmin = svdup_f64(data[begin]);

        size_t i = begin;
        for (; i + vl <= end; i += vl) {
            const svfloat64_t v = svld1_f64(pg, data + i);
            vmax = svmax_f64_x(pg, vmax, v);
            vmin = svmin_f64_x(pg, vmin, v);
        }

        out_max = svmaxv_f64(pg, vmax);
        out_min = svminv_f64(pg, vmin);
        for (; i < end; ++i) {
            out_max = std::max(out_max, data[i]);
            out_min = std::min(out_min, data[i]);
        }
    }
    else {
        scalar_minmax(data, begin, end, out_min, out_max);
    }
}
#endif

#ifdef __AVX2__
template <class T>
inline void avx2_minmax(const T *data, size_t begin, size_t end, T &out_min, T &out_max) {
    if constexpr (std::is_same_v<T, float>) {
        __m256 vmax = _mm256_set1_ps(data[begin]);
        __m256 vmin = _mm256_set1_ps(data[begin]);

        size_t i = begin;
        for (; i + 8 <= end; i += 8) {
            const __m256 v = _mm256_loadu_ps(data + i);
            vmax = _mm256_max_ps(vmax, v);
            vmin = _mm256_min_ps(vmin, v);
        }

        float tmp_max[8], tmp_min[8];
        _mm256_storeu_ps(tmp_max, vmax);
        _mm256_storeu_ps(tmp_min, vmin);

        out_max = tmp_max[0];
        out_min = tmp_min[0];
        for (int k = 1; k < 8; ++k) {
            out_max = std::max(out_max, tmp_max[k]);
            out_min = std::min(out_min, tmp_min[k]);
        }

        for (; i < end; ++i) {
            out_max = std::max(out_max, data[i]);
            out_min = std::min(out_min, data[i]);
        }
    }
    else if constexpr (std::is_same_v<T, double>) {
        __m256d vmax = _mm256_set1_pd(data[begin]);
        __m256d vmin = _mm256_set1_pd(data[begin]);

        size_t i = begin;
        for (; i + 4 <= end; i += 4) {
            const __m256d v = _mm256_loadu_pd(data + i);
            vmax = _mm256_max_pd(vmax, v);
            vmin = _mm256_min_pd(vmin, v);
        }

        double tmp_max[4], tmp_min[4];
        _mm256_storeu_pd(tmp_max, vmax);
        _mm256_storeu_pd(tmp_min, vmin);

        out_max = tmp_max[0];
        out_min = tmp_min[0];
        for (int k = 1; k < 4; ++k) {
            out_max = std::max(out_max, tmp_max[k]);
            out_min = std::min(out_min, tmp_min[k]);
        }

        for (; i < end; ++i) {
            out_max = std::max(out_max, data[i]);
            out_min = std::min(out_min, data[i]);
        }
    }
    else {
        scalar_minmax(data, begin, end, out_min, out_max);
    }
}
#endif

template <class T>
T data_range(const T *data, size_t num, const bool enable_parallel = true) {
    if (num <= 16) {
        T min_val, max_val;
        scalar_minmax(data, 0, num, min_val, max_val);
        return max_val - min_val;
    }

    T min_val = data[0];
    T max_val = data[0];
    if (!enable_parallel) {
#ifdef __ARM_FEATURE_SVE2
        sve2_minmax(data, 0, num, min_val, max_val);
#elif defined(__AVX2__)
        avx2_minmax(data, 0, num, min_val, max_val);
#else
        scalar_minmax(data, 0, num, min_val, max_val);
#endif
        return max_val - min_val;
    }
#ifdef _OPENMP
    #pragma omp parallel reduction(min:min_val) reduction(max:max_val)
    {
        const size_t thread_count = static_cast<size_t>(omp_get_num_threads());
        const size_t thread_id = static_cast<size_t>(omp_get_thread_num());
        const size_t base = num / thread_count;
        const size_t remain = num % thread_count;
        const size_t begin = thread_id * base + std::min(thread_id, remain);
        const size_t end = begin + base + (thread_id < remain ? 1 : 0);

        if (begin < end) {
            T local_min, local_max;
#ifdef __ARM_FEATURE_SVE2
            sve2_minmax(data, begin, end, local_min, local_max);
#elif defined(__AVX2__)
            avx2_minmax(data, begin, end, local_min, local_max);
#else
            scalar_minmax(data, begin, end, local_min, local_max);
#endif
            min_val = std::min(min_val, local_min);
            max_val = std::max(max_val, local_max);
        }
    }
#endif
    return max_val - min_val;
}

inline int factorial(int n) { return (n == 0) || (n == 1) ? 1 : n * factorial(n - 1); }

inline double computeABSErrBoundFromPSNR(double psnr, double threshold, double value_range) {
    double v1 = psnr + 10 * log10(1 - 2.0 / 3.0 * threshold);
    double v2 = v1 / (-20);
    double v3 = pow(10, v2);
    return value_range * v3;
}

template <class T>
void calAbsErrorBound(Config &conf, const T *data, T range = 0) {
    if (conf.errorBoundMode != EB_ABS) {
        if (conf.errorBoundMode == EB_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num, conf.openmp));
        } else if (conf.errorBoundMode == EB_PSNR) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = computeABSErrBoundFromPSNR(conf.psnrErrorBound, 0.99,
                                                            ((range > 0) ? range : data_range(data, conf.num, conf.openmp)));
        } else if (conf.errorBoundMode == EB_L2NORM) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound = sqrt(3.0 / conf.num) * conf.l2normErrorBound;
        } else if (conf.errorBoundMode == EB_ABS_AND_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound =
                std::min(conf.absErrorBound, conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num, conf.openmp)));
        } else if (conf.errorBoundMode == EB_ABS_OR_REL) {
            conf.errorBoundMode = EB_ABS;
            conf.absErrorBound =
                std::max(conf.absErrorBound, conf.relErrorBound * ((range > 0) ? range : data_range(data, conf.num, conf.openmp)));
        } else {
            throw std::invalid_argument("Error bound mode not supported");
        }
    }
}

template <typename Type>
double autocorrelation1DLag1(const Type *data, size_t numOfElem, Type avg) {
    double cov = 0;
    for (size_t i = 0; i < numOfElem; i++) {
        cov += (data[i] - avg) * (data[i] - avg);
    }
    cov = cov / numOfElem;

    if (cov == 0) {
        return 0;
    } else {
        int delta = 1;
        double sum = 0;

        for (size_t i = 0; i < numOfElem - delta; i++) {
            sum += (data[i] - avg) * (data[i + delta] - avg);
        }
        return sum / (numOfElem - delta) / cov;
    }
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse, double &max_diff) {
    size_t i = 0;
    Type Max = ori_data[0];
    Type Min = ori_data[0];
    max_diff = fabs(statistic_difference(data[0], ori_data[0]));
    double diff_sum = 0;
    double maxpw_relerr = 0;
    double sum1 = 0, sum2 = 0, l2sum = 0;
    for (i = 0; i < num_elements; i++) {
        const double original = static_cast<double>(ori_data[i]);
        const double reconstructed = static_cast<double>(data[i]);
        sum1 += original;
        sum2 += reconstructed;
        l2sum += reconstructed * reconstructed;
    }
    double mean1 = sum1 / num_elements;
    double mean2 = sum2 / num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double *diff = static_cast<double *>(malloc(num_elements * sizeof(double)));

    for (i = 0; i < num_elements; i++) {
        const double original = static_cast<double>(ori_data[i]);
        const double reconstructed = static_cast<double>(data[i]);
        diff[i] = statistic_difference(data[i], ori_data[i]);
        diff_sum += diff[i];
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        double err = fabs(diff[i]);
        if (ori_data[i] != 0) {
            relerr = err / fabs(original);
            if (maxpw_relerr < relerr) maxpw_relerr = relerr;
        }

        if (max_diff < err) max_diff = err;
        prodSum += (original - mean1) * (reconstructed - mean2);
        sum3 += (original - mean1) * (original - mean1);
        sum4 += (reconstructed - mean2) * (reconstructed - mean2);
        sum += err * err;
    }
    double std1 = sqrt(sum3 / num_elements);
    double std2 = sqrt(sum4 / num_elements);
    double ee = prodSum / num_elements;
    double acEff = ee / std1 / std2;

    double mse = sum / num_elements;
    double range = fabs(statistic_difference(Max, Min));
    if (mse == 0) {
        psnr = std::numeric_limits<double>::infinity();
        nrmse = 0;
    } else {
        psnr = 20 * log10(range) - 10 * log10(mse);
        nrmse = sqrt(mse) / range;
    }

    double normErr = sqrt(sum);
    double normErr_norm = normErr / sqrt(l2sum);

    if constexpr (std::is_integral_v<Type>) {
        if constexpr (std::is_signed_v<Type>) {
            printf("Min=%lld, Max=%lld, range=%.20G\n", static_cast<long long>(Min),
                   static_cast<long long>(Max), range);
        } else {
            printf("Min=%llu, Max=%llu, range=%.20G\n", static_cast<unsigned long long>(Min),
                   static_cast<unsigned long long>(Max), range);
        }
    } else {
        printf("Min=%.20G, Max=%.20G, range=%.20G\n", static_cast<double>(Min),
               static_cast<double>(Max), range);
    }
    printf("Max absolute error = %.2G\n", max_diff);
    printf("Max relative error = %.2G\n", range > 0 ? max_diff / range : 0);
    printf("Max pw relative error = %.2G\n", maxpw_relerr);
    printf("PSNR = %f, NRMSE= %.10G\n", psnr, nrmse);
    printf("normError = %f, normErr_norm = %f\n", normErr, normErr_norm);
    printf("acEff=%f\n", acEff);
    //        printf("errAutoCorr=%.10f\n", autocorrelation1DLag1<double>(diff, num_elements, diff_sum / num_elements));
    free(diff);
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements) {
    double psnr, nrmse, max_diff;
    verify(ori_data, data, num_elements, psnr, nrmse, max_diff);
}

template <typename Type>
void verify(Type *ori_data, Type *data, size_t num_elements, double &psnr, double &nrmse) {
    double max_diff;
    verify(ori_data, data, num_elements, psnr, nrmse, max_diff);
}
}  // namespace Falcons

#endif
