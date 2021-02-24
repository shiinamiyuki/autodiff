#pragma once

template <class Scalar>
Scalar F(Scalar x, Scalar y) {
    // return x * x + y * y + 1.0;
    auto t  = (1.0 - x);
    auto t2 = (y - x * x);
    return t * t + 100.0 * t2 * t2;
}