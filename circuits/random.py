from pennylane.templates import RandomLayers
import pennylane as qml
import numpy as np


def ry_random(qubits, kernel_size, filters, n_layers):

    # The number of filters can not exceed the number of qubits
    if filters > qubits:
        # If so, force the number of filters to the maximum possible
        filters = qubits
    
    # The sqrt(kernel size) can not exceed the number of qubits
    if kernel_size**2 > qubits:
         # If so, force the kernel size to the maximum possible
        kernel_size = int(np.sqrt(qubits))


    dev = qml.device("default.qubit", wires=qubits)
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, qubits))

    @qml.qnode(dev)
    def circuit(phi):
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)))

        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit

