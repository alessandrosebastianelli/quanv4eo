from pennylane.templates import RandomLayers
import pennylane as qml
import numpy as np


import jax
from jax.config import config
config.update('jax_enable_x64', True)

@jax.jit
def ry_random(qubits, kernel_size, filters, n_layers, key=jax.random.PRNGKey(0), seed=1):
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

    # Notice how the device construction now happens within the jitted method.
    # Also note the added '.jax' to the device path.

    dev = qml.device("default.qubit.jax", wires=2, shots=10, prng_key=key)
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, qubits))

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax", diff_method=None)
    def circuit(phi, seed=seed):
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RY(np.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit

@jax.jit
def rx_random(qubits, kernel_size, filters, n_layers, key=jax.random.PRNGKey(0), seed=1):
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

    # Notice how the device construction now happens within the jitted method.
    # Also note the added '.jax' to the device path.
    dev = qml.device("default.qubit.jax", wires=2, shots=10, prng_key=key)

    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, qubits))

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax", diff_method=None)
    def circuit(phi, seed=seed):
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RX(np.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit

def rz_random(qubits, kernel_size, filters, n_layers, key = jax.random.PRNGKey(0), seed=1):
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

    # Notice how the device construction now happens within the jitted method.
    # Also note the added '.jax' to the device path.
    dev = qml.device("default.qubit.jax", wires=2, shots=10, prng_key=key)

    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, qubits))

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax", diff_method=None)
    def circuit(phi, seed=seed):
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RZ(np.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit