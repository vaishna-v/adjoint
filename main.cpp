#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>
#include <memory>

class Value;
using SP = std::shared_ptr<Value>;

// Value- heap-allocated node in the compute graph, owns its data/grad/backward fn
class Value : public std::enable_shared_from_this<Value> {
public:
    double data, grad = 0.0;
    std::vector<SP> prev;
    std::function<void()> _backward = [] {};

    // factory- always construct on heap so shared_from_this works
    static SP make(double data, std::vector<SP> parents = {}) {
        return std::shared_ptr<Value>(new Value(data, std::move(parents)));
    }

    void zero_grad() { grad = 0.0; }

    // topological sort then reverse-accumulate gradients
    void backward() {
        std::vector<Value*> topo;
        std::unordered_set<Value*> visited;

        std::function<void(Value*)> build = [&](Value* v) {
            if (visited.insert(v).second)
                for (auto& p : v->prev) build(p.get());
            topo.push_back(v);
        };

        build(this);
        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
    }

    SP operator+(SP rhs) {
        auto out = make(data + rhs->data, {shared_from_this(), rhs});
        // d/dx (a+b) = 1 for both
        out->_backward = [self = shared_from_this(), rhs, out = out.get()] {
            self->grad += out->grad;
            rhs->grad  += out->grad;
        };
        return out;
    }

    SP operator-(SP rhs) {
        auto out = make(data - rhs->data, {shared_from_this(), rhs});
        // d/dx (a-b) = 1, d/dy = -1
        out->_backward = [self = shared_from_this(), rhs, out = out.get()] {
            self->grad += out->grad;
            rhs->grad  -= out->grad;
        };
        return out;
    }

    SP operator*(SP rhs) {
        auto out = make(data * rhs->data, {shared_from_this(), rhs});
        // product rule: da = b, db = a
        out->_backward = [self = shared_from_this(), rhs, out = out.get()] {
            self->grad += rhs->data  * out->grad;
            rhs->grad  += self->data * out->grad;
        };
        return out;
    }

    SP operator/(SP rhs) {
        auto out = make(data / rhs->data, {shared_from_this(), rhs});
        // quotient rule: da = 1/b, db = -a/b^2
        out->_backward = [self = shared_from_this(), rhs, out = out.get()] {
            self->grad +=  out->grad / rhs->data;
            rhs->grad  += -self->data * out->grad / (rhs->data * rhs->data);
        };
        return out;
    }

    // d/dx x^n = n * x^(n-1)
    SP pow(double exp) {
        auto out = make(std::pow(data, exp), {shared_from_this()});
        out->_backward = [self = shared_from_this(), exp, out = out.get()] {
            self->grad += exp * std::pow(self->data, exp - 1) * out->grad;
        };
        return out;
    }

    // tanh and its derivative (1 - tanh^2)
    SP tanh() {
        double t = std::tanh(data);
        auto out = make(t, {shared_from_this()});
        out->_backward = [self = shared_from_this(), t, out = out.get()] {
            self->grad += (1.0 - t * t) * out->grad;
        };
        return out;
    }

    // relu- passes gradient only where input was positive
    SP relu() {
        auto out = make(data > 0 ? data : 0.0, {shared_from_this()});
        out->_backward = [self = shared_from_this(), out = out.get()] {
            self->grad += (out->data > 0 ? 1.0 : 0.0) * out->grad;
        };
        return out;
    }

private:
    Value(double data, std::vector<SP> parents = {})
        : data(data), prev(parents) {}
};

// helpers so you can write  a + b  instead of  a->operator+(b)
inline SP operator+(SP a, SP b) { return (*a) + b; }
inline SP operator-(SP a, SP b) { return (*a) - b; }
inline SP operator*(SP a, SP b) { return (*a) * b; }
inline SP operator/(SP a, SP b) { return (*a) / b; }

inline SP V(double d) { return Value::make(d); }

int main() {
    SP a = V(3.0);
    SP b = V(4.0);
    SP c = V(2.0);

    // f = (a + b) * c
    SP ab = a + b;
    SP f  = ab * c;

    f->backward();

    std::cout << "f    = " << f->data  << "\n"; // 14
    std::cout << "df/da = " << a->grad << "\n"; // 2
    std::cout << "df/db = " << b->grad << "\n"; // 2
    std::cout << "df/dc = " << c->grad << "\n"; // 7

    // verify new math operations
    SP x = V(0.5);
    SP y = x->tanh();
    y->backward();
    std::cout << "tanh(0.5)  = " << y->data << "\n";
    std::cout << "d(tanh)/dx = " << x->grad << "\n"; // 1 - tanh(0.5)^2
}
