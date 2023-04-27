from pennylane.templates import RandomLayers
import pennylane as qml
import numpy as np


def ry_random(qubits, kernel_size, filters, n_layers, seed=1):
    '''
        Creates a Quantum Circuit with a first layer of Ry gates and a sequence of Random Circuits

        Inputs:
            - qubits: number of qubits for the quantum circuit
            - kernel_size: kernel size for quantum convolution, is used to regulates the number of Ry gates.
                           !!! kernel_size**2 must be less than #qubits !!! 
            - filters: number of desired output filters.
            - n_layers: filters must be less than qubits
        Output:
            - circuit: return the quantum circuit
    '''

    # The number of filters can not exceed the number of qubits
    # If so, force the number of filters to the maximum possible
    if filters > qubits: filters = qubits
    
    # The sqrt(kernel size) can not exceed the number of qubits
    # If so, force the kernel size to the maximum possible
    if kernel_size**2 > qubits: kernel_size = int(np.sqrt(qubits))

    dev = qml.device("default.qubit", wires=qubits)
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, qubits))

    @qml.qnode(dev)
    def circuit(input, weights):
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RY(np.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit
