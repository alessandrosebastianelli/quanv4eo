from pennylane.templates import RandomLayers
import pennylane as qml
import numpy as np

##### Jax & Pennylane
# https://pennylane.ai/qml/demos/tutorial_jax_transformations.html
#####
import jax
import jax.numpy as jnp
from functools import partial

#@jax.jit
@partial(jax.jit, static_argnames=['qubits', 'kernel_size', 'filters', 'n_layers', 'seed'])
def ry_random(phi, qubits, kernel_size, filters, n_layers, seed):
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

    #print(n_layers, qubits)

    dev = qml.device("default.qubit.jax", wires=qubits, shots=1, prng_key=jax.random.PRNGKey(758493))
    # Random circuit parameters
    rand_params = jax.random.uniform(jax.random.PRNGKey(758493), shape=(n_layers, qubits))#, maxval=2*jnp.pi)

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax")
    def circuit():
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RY(jnp.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit()

@partial(jax.jit, static_argnames=['qubits', 'kernel_size', 'filters', 'n_layers', 'seed'])
def rx_random(phi, qubits, kernel_size, filters, n_layers, seed):
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

    #print(n_layers, qubits)

    dev = qml.device("default.qubit.jax", wires=qubits, shots=1, prng_key=jax.random.PRNGKey(758493))
    # Random circuit parameters
    rand_params = jax.random.uniform(jax.random.PRNGKey(758493), shape=(n_layers, qubits))#, maxval=2*jnp.pi)

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax")
    def circuit():
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RX(jnp.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit()

@partial(jax.jit, static_argnames=['qubits', 'kernel_size', 'filters', 'n_layers', 'seed'])
def rz_random(phi, qubits, kernel_size, filters, n_layers, seed):
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

    #print(n_layers, qubits)

    dev = qml.device("default.qubit.jax", wires=qubits, shots=1, prng_key=jax.random.PRNGKey(758493))
    # Random circuit parameters
    rand_params = jax.random.uniform(jax.random.PRNGKey(758493), shape=(n_layers, qubits))#, maxval=2*jnp.pi)

    # Now we can create our qnode within the circuit function.
    @qml.qnode(dev, interface="jax")
    def circuit():
        # Encoding of kernel_size x kernel_size classical input values
        for j in range(kernel_size**2):
            qml.RZ(jnp.pi * phi[j], wires=j)
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(qubits)), seed=seed)
        # Measurement producing #filters classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(filters)]

    return circuit()
