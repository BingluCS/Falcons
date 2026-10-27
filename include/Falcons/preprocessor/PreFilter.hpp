//
// Created by Kai Zhao on 1/29/21.
//

#ifndef Falcons_PREFILTER_HPP
#define Falcons_PREFILTER_HPP

#include "Falcons/preprocessor/PreProcessor.hpp"

namespace Falcons {
    template<class T, uint N>

    class PreFilter : public concepts::PreprocessorInterface<T, N> {

        void preprocess(T *data, std::array<size_t, N> dims, std::pair<T, T> range, T defaultValue) {
            for (T &d : data) {
                if (d > range.second || d < range.first) {
                    d = defaultValue;
                }
            }
        }
    };
}
#endif //Falcons_PRETRANSPOSE_H
