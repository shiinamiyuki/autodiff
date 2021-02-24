#include <autodiff/autodiff.h>
#include <iostream>
int main() {
    using namespace autodiff;
    using ADFloat = ADVar<float>;
    start_recording();
    ADFloat x("x");
    ADFloat y("y");
    ADFloat z = x + y;
    stop_recording();
    set_gradient(z, "1");
    backward();
    std::cout << codegen() << std::endl;
}