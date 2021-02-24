#include <autodiff/autodiff.h>
#include <iostream>
#include <fstream>
#include "simple_test.h"

int main() {
    using namespace autodiff;
    using ADFloat = ADVar<float>;
    start_recording();
    ADFloat x("x");
    ADFloat y("y");
    ADFloat z = F(x, y);
    stop_recording();
    set_gradient(z, "dz");
    backward();
    // std::cout << codegen() << std::endl;
    
    std::ofstream out("grad.cpp");
    out << "void grad_F(float x, float y, float dz, float& dx, float &dy){\n";
    out << codegen();
    out << "dx = " << gradients(x) << ";\n";
    out << "dy = " << gradients(y) << ";\n";
    out << "}\n";
}