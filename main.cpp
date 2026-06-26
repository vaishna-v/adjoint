#include <iostream>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>
#include <memory>
#include <random>

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

// random weight init in [-1, 1]
static std::mt19937 rng(42);
static std::uniform_real_distribution<double> dist(-1.0, 1.0);
inline SP randW() { return V(dist(rng)); }

// Neuron- one dot product + bias, optional nonlinearity
class Neuron {
public:
    std::vector<SP> w;
    SP b;
    bool nonlin;

    Neuron(int nin, bool nonlin = true) : b(V(0.0)), nonlin(nonlin) {
        for (int i = 0; i < nin; i++) w.push_back(randW());
    }

    SP operator()(const std::vector<SP>& x) {
        auto act = b;
        for (int i = 0; i < (int)w.size(); i++) act = act + w[i] * x[i];
        return nonlin ? act->tanh() : act;
    }

    std::vector<SP> params() {
        auto p = w;
        p.push_back(b);
        return p;
    }
};

// Layer- a row of neurons all receiving the same input
class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; i++) neurons.emplace_back(nin, nonlin);
    }

    std::vector<SP> operator()(const std::vector<SP>& x) {
        std::vector<SP> out;
        for (auto& n : neurons) out.push_back(n(x));
        return out;
    }

    std::vector<SP> params() {
        std::vector<SP> p;
        for (auto& n : neurons)
            for (auto& v : n.params()) p.push_back(v);
        return p;
    }
};

// MLP- stack of layers, last one has no nonlinearity (raw logit/regression output)
class MLP {
public:
    std::vector<Layer> layers;

    // sizes = [nin, h1, h2, ..., nout]
    MLP(std::vector<int> sizes) {
        for (int i = 0; i < (int)sizes.size() - 1; i++) {
            bool last = (i == (int)sizes.size() - 2);
            layers.emplace_back(sizes[i], sizes[i+1], !last);
        }
    }

    std::vector<SP> operator()(std::vector<SP> x) {
        for (auto& l : layers) x = l(x);
        return x;
    }

    std::vector<SP> params() {
        std::vector<SP> p;
        for (auto& l : layers)
            for (auto& v : l.params()) p.push_back(v);
        return p;
    }

    void zero_grad() { for (auto& p : params()) p->zero_grad(); }
};

// SGD- nudge each param opposite to its gradient
class SGD {
public:
    std::vector<SP> params;
    double lr;

    SGD(std::vector<SP> params, double lr) : params(params), lr(lr) {}

    void step() {
        for (auto& p : params) p->data -= lr * p->grad;
    }

    void zero_grad() { for (auto& p : params) p->zero_grad(); }
};

int main() {
    // tiny dataset- XOR-like, 2 inputs -> 1 output
    std::vector<std::vector<double>> X = {{0,0},{0,1},{1,0},{1,1}};
    std::vector<double>              Y = { -1,   1,   1,  -1 };

    MLP model({2, 4, 4, 1});
    SGD optim(model.params(), 0.05);

    for (int epoch = 0; epoch < 100; epoch++) {
        // forward + MSE loss
        SP loss = V(0.0);
        for (int i = 0; i < 4; i++) {
            std::vector<SP> x = {V(X[i][0]), V(X[i][1])};
            SP pred = model(x)[0];
            SP diff = pred - V(Y[i]);
            loss = loss + diff->pow(2);
        }

        optim.zero_grad();
        loss->backward();
        optim.step();

        if (epoch % 10 == 0)
            std::cout << "epoch " << epoch << "  loss = " << loss->data << "\n";
    }
}