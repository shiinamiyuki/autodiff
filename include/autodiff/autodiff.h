#pragma once

#include <iostream>
#include <memory>
#include <cassert>
#include <list>
#include <string>
#include <vector>
#include <type_traits>
#include <sstream>
namespace autodiff {
    enum class Type {
        Void,
        Bool,
        Int8,
        Int32,
        UInt32,
        Float32,
        Float64,
    };
    inline size_t size_of(Type type) {
        switch (type) {
        case Type::Bool:
            return 1;
        case Type::Float32:
            return 4;
        case Type::Float64:
            return 8;
        case Type::Int32:
            return 4;
        case Type::UInt32:
            return 4;
        default:
            assert(false);
        }
    }
    inline std::string type_to_str(Type type) {
        switch (type) {
        case Type::Void:
            return "void";
        case Type::Bool:
            return "bool";
        case Type::Float32:
            return "float";
        case Type::Float64:
            return "double";
        case Type::Int32:
            return "int";
        case Type::UInt32:
            return "unsigned int";
        default:
            abort();
        }
    }
    inline bool is_float(Type t) { return t == Type::Float32 || t == Type::Float64; }
    inline bool is_int(Type t) { return t != Type::Void && !is_float(t); }
    template <typename T>
    inline Type from_cpp_type() = delete;
    template <>
    inline Type from_cpp_type<void>() {
        return Type::Void;
    }
    template <>
    inline Type from_cpp_type<bool>() {
        return Type::Bool;
    }
    template <>
    inline Type from_cpp_type<int8_t>() {
        return Type::Int8;
    }
    template <>
    inline Type from_cpp_type<int32_t>() {
        return Type::Int32;
    }
    template <>
    inline Type from_cpp_type<uint32_t>() {
        return Type::UInt32;
    }
    template <>
    inline Type from_cpp_type<float>() {
        return Type::Float32;
    }
    template <>
    inline Type from_cpp_type<double>() {
        return Type::Float64;
    }

    class ADRecorder {
      public:
        struct Var {
            int32_t id = (int32_t)-1;
            Type type;
            std::string forward;
            std::string backward;
            int32_t deps[4] = {-1, -1, -1, -1};
        };
        std::vector<Var> vars;

        struct CG {
            std::ostringstream var_decl;
            std::ostringstream forward;
            std::ostringstream backward;
        };
        CG cg;

      public:
        // std::list<std::string>
    };
    static ADRecorder recorder;
    inline int32_t append(Type type, const std::string &forward, const std::string &backward) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward};
        recorder.vars.push_back(var);
        return var.id;
    }
    inline int32_t append(Type type, const std::string &forward, const std::string &backward, int32_t dep0) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward, {dep0, -1, -1, -1}};
        recorder.vars.push_back(var);
        return var.id;
    }
    inline int32_t append(Type type, const std::string &forward, const std::string &backward, int32_t dep0,
                          int32_t dep1) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward, {dep0, dep1, -1, -1}};
        recorder.vars.push_back(var);
        return var.id;
    }
    inline int32_t append(Type type, const std::string &forward, const std::string &backward, int32_t dep0,
                          int32_t dep1, int32_t dep2) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward, {dep0, dep1, dep2, -1}};
        recorder.vars.push_back(var);
        return var.id;
    }
    template <class Scalar>
    class ADVar {
        ADVar(int32_t id, bool _) : id(id) { (void)_; }
        static std::string to_string(Scalar v) {
            if constexpr (std::is_floating_point_v<Scalar>) {
                std::ostringstream os;
                os.precision(std::numeric_limits<Scalar>::max_digits10);
                os << v;
                return os.str();
            } else {
                return std::to_string(v);
            }
        }

      public:
        int32_t id;
        explicit ADVar(Scalar v) : ADVar(to_string(v)) {}
        ADVar(const std::string &symbol) { id = append(from_cpp_type<Scalar>(), "$v=" + symbol + ";", ""); }
        ADVar() : ADVar(Scalar()) {}
        template <class T>
        explicit ADVar(const ADVar<T> &rhs) {
            if (std::is_floating_point_v<T>) {
                id = append(from_cpp_type<Scalar>(), std::string("$v = ") + from_cpp_type<Scalar>() + "($0)",
                            "d$0 += d$v;");
            } else {
                id = append(from_cpp_type<Scalar>(), std::string("$v = ") + from_cpp_type<Scalar>() + "($0)", "");
            }
        }
        static ADVar from_id(int32_t id) { return ADVar(id, true); }
        ADVar operator+(const ADVar &rhs) const {
            std::string forward  = "$v = $0 + $1;";
            std::string backward = "d$0 += d$v;d$1 += d$v;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, id, rhs.id));
        }
        ADVar operator-(const ADVar &rhs) const {
            std::string forward  = "$v = $0 - $1;";
            std::string backward = "d$0 += d$v;d$1 -= d$v;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, id, rhs.id));
        }
        ADVar operator*(const ADVar &rhs) const {
            std::string forward  = "$v = $0 * $1;";
            std::string backward = "d$0 += d$v * $1;d$1 += d$v * $0;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, id, rhs.id));
        }
        ADVar operator/(const ADVar &rhs) const {
            std::string forward  = "$v = $0 / $1;";
            std::string backward = "d$0 += d$v / $1;d$1 += d$v * $0 / ($1 * $1);";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, id, rhs.id));
        }
        ADVar operator-() const {
            std::string forward  = "$v = -$0;";
            std::string backward = "d$0 += -d$v;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, id));
        }
        friend ADVar select(const ADVar<bool> &cond, const ADVar &a, const ADVar &b) {
            std::string forward  = "$v = $0 ? $1 : $2;";
            std::string backward = "if($0){d$1 += d$v;}else{d$2 += d$v;}";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, cond.id, a.id, b.id));
        }
        friend ADVar sin(const ADVar &x) {
            std::string forward  = "$v = std::sin($0);";
            std::string backward = "d$0 += d$v * std::cos($0);";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
        friend ADVar cos(const ADVar &x) {
            std::string forward  = "$v = std::cos($0);";
            std::string backward = "d$0 += d$v * std::sin($0);";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
        friend ADVar log(const ADVar &x) {
            std::string forward  = "$v = std::log($0);";
            std::string backward = "d$0 += d$v / $0;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
        friend ADVar exp(const ADVar &x) {
            std::string forward  = "$v = std::exp($0);";
            std::string backward = "d$0 += d$v * $v;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
        friend ADVar sqrt(const ADVar &x) {
            std::string forward  = "$v = std::sqrt($0);";
            std::string backward = "d$0 += d$v * 0.5 / $v;";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
#define CMP_OP(op)                                                                                                     \
    ADVar<bool> operator op(const ADVar &rhs) const {                                                                  \
        return ADVar<bool>::from_id(append(Type::Bool, "$v = $0 " #op " $1;", "", id, rhs.id));                        \
    }                                                                                                                  \
    friend ADVar<bool> operator op(Scalar lhs, const ADVar &rhs) { return ADVar(lhs) op rhs; }                         \
    ADVar<bool> operator op(Scalar rhs) const { return *this op ADVar(rhs); }
        CMP_OP(==)
        CMP_OP(!=)
        CMP_OP(<=)
        CMP_OP(>=)
        CMP_OP(<)
        CMP_OP(>)
#define SCALAR_OP(op, op_assign)                                                                                       \
    friend ADVar operator op(Scalar lhs, const ADVar &rhs) { return ADVar(lhs) op rhs; }                               \
    ADVar operator op(Scalar rhs) const { return *this op ADVar(rhs); }                                                \
    ADVar &operator op_assign(const Scalar rhs) {                                                                      \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }                                                                                                                  \
    ADVar &operator op_assign(const ADVar &rhs) {                                                                      \
        *this = *this op rhs;                                                                                          \
        return *this;                                                                                                  \
    }
        SCALAR_OP(+, +=)
        SCALAR_OP(-, -=)
        SCALAR_OP(*, *=)
        SCALAR_OP(/, /=)
    };

    inline bool replace_one(std::string &str, const std::string &from, const std::string &to) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos)
            return false;
        str.replace(start_pos, from.length(), to);
        return true;
    }
    inline void replace(std::string &str, const std::string &from, const std::string &to) {
        while (replace_one(str, from, to))
            ;
    }
    inline void start_recording() {
        recorder.vars.clear();
        recorder.cg = ADRecorder::CG();
    }
    inline void stop_recording() {
        auto &cg = recorder.cg;
        for (auto &var : recorder.vars) {
            cg.var_decl << type_to_str(var.type) << " v" << var.id << " = 0;\n";
            cg.var_decl << type_to_str(var.type) << " dv" << var.id << " = 0;\n";
        }
        for (auto &var : recorder.vars) {
            auto forward = var.forward;
            replace(forward, "$v", "v" + std::to_string(var.id));
            for (int i = 0; i < 4; i++) {
                if (var.deps[i] == -1) {
                    break;
                }
                replace(forward, "$" + std::to_string(i), "v" + std::to_string(var.deps[i]));
            }
            if (!forward.empty())
                cg.forward << forward << "\n";
        }
    }
    template <class Scalar>
    inline void set_gradient(const ADVar<Scalar> &v, const std::string &symbol) {
        auto &cg = recorder.cg;
        cg.backward << "dv" << v.id << " =" << symbol << ";\n";
    }
    inline void backward() {
        auto &cg = recorder.cg;
        for (auto riter = recorder.vars.rbegin(); riter != recorder.vars.rend(); riter++) {
            auto &var     = *riter;
            auto backward = var.backward;
            replace(backward, "$v", "v" + std::to_string(var.id));
            for (int i = 0; i < 4; i++) {
                if (var.deps[i] == -1) {
                    break;
                }
                replace(backward, "$" + std::to_string(i), "v" + std::to_string(var.deps[i]));
            }
            if (!backward.empty() && is_float(var.type))
                cg.backward << backward << "\n";
        }
    }
    template <class Scalar>
    std::string gradients(const ADVar<Scalar> &v) {
        return "dv" + std::to_string(v.id);
    }
    inline std::string codegen() {
        auto &cg = recorder.cg;
        return cg.var_decl.str() + cg.forward.str() + cg.backward.str();
    }
} // namespace autodiff