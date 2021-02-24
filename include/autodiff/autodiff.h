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
#ifdef NODE_APPROACH
    class ADNode {
      public:
        Type type;
        std::string grad;
        bool requires_grad = false;
        ADNode(Type type) : type(type) {
            if (is_float(type)) {
                requires_grad = true;
            }
        }
    };
    class CloneNode : public ADNode {
      public:
        std::shared_ptr<ADNode> value;
        CloneNode(std::shared_ptr<ADNode> value) : ADNode(value->type), value(value) { requires_grad = false; }
    };
    class SymbolNode : public ADNode {
      public:
        std::string sym;
        SymbolNode(Type type, std::string sym) : ADNode(type), sym(sym) {}
    };
    class BinaryADNode : public ADNode {
      public:
        enum class Op { Add, Sub, Mul, Div };
        Op op;
        std::shared_ptr<ADNode> lhs, rhs;
        BinaryADNode(Type type, Op op, std::shared_ptr<ADNode> lhs, std::shared_ptr<ADNode> rhs)
            : ADNode(type), op(op), lhs(lhs), rhs(rhs) {}
    };
    class UnaryADNode : public ADNode {
      public:
        enum class Op { Neg, Sin, Cos, Tan, Log, Pow, Sqrt };
        Op op;
        std::shared_ptr<ADNode> expr;
        UnaryADNode(Type type, Op op, std::shared_ptr<ADNode> expr) : ADNode(type), op(op), expr(expr) {}
    };

    class ADVarBase {
      protected:
        std::shared_ptr<ADNode> node;

        std::string grad;

      public:
        ADVarBase(std::shared_ptr<ADNode> node) : node(node) {}
        ADVarBase(std::shared_ptr<ADNode> node, bool requires_grad) : node(node) {
            this->node->requires_grad = requires_grad;
        }
    };
    template <class Scalar>
    class ADVar : public ADVarBase {
      public:
        using ADVarBase::ADVarBase;
        bool requires_grad() const { return node->requires_grad; }
#    define SCALAR_OP(op, name)                                                                                        \
        friend ADVar operator op(Scalar lhs, const ADVar &rhs) {                                                       \
            auto s = std::to_string(lhs);                                                                              \
            return ADVar(                                                                                              \
                std::make_shared<BinaryADNode>(name, std::make_shared<SymbolNode>(from_cpp_type<Scalar>(), s), node),  \
                requires_grad());                                                                                      \
        }                                                                                                              \
        ADVar operator op(Scalar rhs) const {                                                                          \
            auto s = std::to_string(rhs);                                                                              \
            return ADVar(                                                                                              \
                std::make_shared<BinaryADNode>(name, node, std::make_shared<SymbolNode>(from_cpp_type<Scalar>(), s)),  \
                requires_grad());                                                                                      \
        }
        ADVar operator op(const ADVar &rhs) const {
            return ADVar(std::make_shared<BinaryADNode>(name, node, rhs.node), requires_grad());
        }
        SCALAR_OP(+, Add)
        SCALAR_OP(-, Sub)
        SCALAR_OP(*, Mul)
        SCALAR_OP(/, Div)
        void set_requires_grad(bool b) {
            static_assert(!std::is_integral_v<Scalar>);
            if (b != this->requires_grad) {
                node = std::make_shared<CloneNode>(node);
            }
            this->node->requires_grad = b;
        }
        void set_gradients(const std::string &symbol) { node->grad = symbol; }
    };
#else
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
    static int32_t append(Type type, const std::string &forward, const std::string &backward) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward};
        recorder.vars.push_back(var);
        return var.id;
    }
    static int32_t append(Type type, const std::string &forward, const std::string &backward, int32_t dep0) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward, {dep0, -1, -1, -1}};
        recorder.vars.push_back(var);
        return var.id;
    }
    static int32_t append(Type type, const std::string &forward, const std::string &backward, int32_t dep0,
                          int32_t dep1) {
        ADRecorder::Var var{(int32_t)recorder.vars.size(), type, forward, backward, {dep0, dep1, -1, -1}};
        recorder.vars.push_back(var);
        return var.id;
    }
    template <class Scalar>
    class ADVar {
        ADVar(int32_t id, bool _) : id(id) { (void)_; }

      public:
        int32_t id;
        explicit ADVar(Scalar v) : ADVar(std::to_string(v)) {}
        ADVar(const std::string &symbol) { id = append(from_cpp_type<Scalar>(), "$v=" + symbol + ";", ""); }
        ADVar() : ADVar(Scalar()) {}
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
            std::string backward = "d$0 += d$v / $0";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
        friend ADVar sqrt(const ADVar &x) {
            std::string forward  = "$v = std::sqrt($0);";
            std::string backward = "d$0 += d$v * 0.5 / $v";
            return from_id(append(from_cpp_type<Scalar>(), forward, backward, x.id));
        }
#    define SCALAR_OP(op, op_assign)                                                                                   \
        friend ADVar operator op(Scalar lhs, const ADVar &rhs) { return ADVar(lhs) op rhs; }                           \
        ADVar operator op(Scalar rhs) const { return *this op ADVar(rhs); }                                            \
        ADVar &operator op_assign(const Scalar rhs) {                                                                  \
            *this = *this op rhs;                                                                                      \
            return *this;                                                                                              \
        }                                                                                                              \
        ADVar &operator op_assign(const ADVar &rhs) {                                                                  \
            *this = *this op rhs;                                                                                      \
            return *this;                                                                                              \
        }
        SCALAR_OP(+, +=)
        SCALAR_OP(-, -=)
        SCALAR_OP(*, *=)
        SCALAR_OP(/, /=)
    };
#endif
    inline bool replace_one(std::string &str, const std::string &from, const std::string &to) {
        size_t start_pos = str.find(from);
        if (start_pos == std::string::npos)
            return false;
        str.replace(start_pos, from.length(), to);
        return true;
    }
    void replace(std::string &str, const std::string &from, const std::string &to) {
        while (replace_one(str, from, to))
            ;
    }
    inline void start_recording() { recorder.vars.clear(); }
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
            cg.forward << forward;
        }
    }
    template <class Scalar>
    inline void set_gradient(const ADVar<Scalar> &v, const std::string &symbol) {
        auto &cg = recorder.cg;
        cg.backward << " dv" << v.id << " =" << symbol << ";\n";
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
            cg.backward << backward;
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