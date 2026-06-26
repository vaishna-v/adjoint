#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>

// Value- data, grad, parent nodes, and a local backward fn for chain rule
class Value {
public:
    double data, grad = 0.0;
    std::vector<Value*> prev;
    std::function<void()> _backward = [] {};

    Value(double data, std::vector<Value*> parents = {})
        : data(data), prev(parents) {}

    // topological sort then reverse-accumulate gradients
    void backward() {
        std::vector<Value*> topo;
        std::unordered_set<Value*> visited;

        std::function<void(Value*)> build = [&](Value* v) {
            if (visited.insert(v).second)
                for (auto p : v->prev) build(p);
            topo.push_back(v);
        };

        build(this);
        grad = 1.0;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            (*it)->_backward();
    }

    Value operator+(Value& rhs) {
        Value out(data + rhs.data, {this, &rhs});
        // d/dx (a+b) = 1 for both
        out._backward = [&, &out=out] {
            grad     += out.grad;
            rhs.grad += out.grad;
        };
        return out;
    }

    Value operator-(Value& rhs) {
        Value out(data - rhs.data, {this, &rhs});
        // d/dx (a-b) = 1, d/dy = -1
        out._backward = [&, &out=out] {
            grad     += out.grad;
            rhs.grad -= out.grad;
        };
        return out;
    }

    Value operator*(Value& rhs) {
        Value out(data * rhs.data, {this, &rhs});
        // product rule: da = b, db = a
        out._backward = [&, &out=out] {
            grad     += rhs.data * out.grad;
            rhs.grad += data     * out.grad;
        };
        return out;
    }

    Value operator/(Value& rhs) {
        Value out(data / rhs.data, {this, &rhs});
        // quotient rule: da = 1/b, db = -a/b^2
        out._backward = [&, &out=out] {
            grad     +=  out.grad / rhs.data;
            rhs.grad += -data * out.grad / (rhs.data * rhs.data);
        };
        return out;
    }
};

int main() {
    Value a(3.0), b(4.0), c(2.0);

    // f = (a + b) * c
    Value ab = a + b;
    Value f  = ab * c;

    f.backward();

    std::cout << "f    = " << f.data  << "\n"; // 14
    std::cout << "df/da = " << a.grad << "\n"; // 2
    std::cout << "df/db = " << b.grad << "\n"; // 2
    std::cout << "df/dc = " << c.grad << "\n"; // 7
}