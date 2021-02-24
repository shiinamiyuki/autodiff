#include <iostream>
#include "simple_test.h"
extern void grad_F(float x, float y, float dz, float &dx, float &dy);

int main() {
    float x = -0.6f;
    float y = 0.4f;

    for (int iter = 0; iter < 10000; iter++) {
        float dx, dy;
        grad_F(x, y, 1.0, dx, dy);
        printf("iter=%d, F(%f,%f)=%f\n", iter, x, y, F(x, y));
        float lr = 0.001f;
        x -= lr * dx;
        y -= lr * dy;
    }
}