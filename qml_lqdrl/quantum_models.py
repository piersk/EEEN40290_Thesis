import pennylane as qml
from pennylane import numpy as np

class QuantumActor:
    def __init__(self, n_qubits, m_layers):
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.theta = np.random.randn(m_layers, n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, theta):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for l in range(m_layers):
                for i in range(n_qubits):
                    qml.RX(x[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i+1])
                for i in range(n_qubits):
                    qml.RY(theta[l][i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def __call__(self, x):
        return self.qnode(x, self.theta)

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)

class QuantumCritic:
    def __init__(self, n_qubits, m_layers):
        self.n_qubits = n_qubits
        self.m_layers = m_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.theta = np.random.randn(m_layers, n_qubits, requires_grad=True)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, theta):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for l in range(m_layers):
                for i in range(n_qubits):
                    qml.RX(x[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CZ(wires=[i, i+1])
                for i in range(n_qubits):
                    qml.RY(theta[l][i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = circuit

    def __call__(self, x):
        return self.qnode(x, self.theta)

    def update_params(self, new_theta):
        self.theta = new_theta

    def draw(self, x):
        return qml.draw(self.qnode)(x, self.theta)

    def latex(self, x):
        return qml.draw_mpl(self.qnode)(x, self.theta)

    def decode_op(self, q_values, scale=30, method="mean"):
        """Decode multi-qubit outputs."""
        q_array = qml.numpy.stack(q_values) if isinstance(q_values, (list, tuple)) else q_values
        if method == "mean":
            return scale * qml.numpy.mean(q_array)
        elif method == "sum":
            return qml.numpy.sum(q_array)
        else:
            raise ValueError("Unknown decoding method")

    def evaluate(self, x, k_shots=10):
        total = np.zeros(self.n_qubits)
        for k in range(k_shots):
            total += self.qnode(x, self.theta)
        avg_q_per_qubit = total / k_shots
        return avg_q_per_qubit 
