import numpy as np
from numba import cuda
import time

# Definir las dimensiones de las matrices
n = 1000
m = 1000
p = 1000

# Definir las matrices que se van a multiplicar
A = np.random.randn(n, m) + 1j * np.random.randn(n, m)
B = np.random.randn(m, p) + 1j * np.random.randn(m, p)
C = np.random.randn(p, m) + 1j * np.random.randn(p, m)

# Multiplicar las matrices utilizando NumPy
start_time = time.time()
D_numpy = A.dot(B).dot(C)
numpy_time = time.time() - start_time

# Multiplicar las matrices utilizando Numba y CUDA
@cuda.jit
def matrix_mult(a, b, c, d):
    i, j = cuda.grid(2)

    if i < a.shape[0] and j < b.shape[1]:
        sum_real = 0.0
        sum_imag = 0.0
        for k in range(a.shape[1]):
            sum_real += a[i, k].real * b[k, j].real - a[i, k].imag * b[k, j].imag
            sum_imag += a[i, k].real * b[k, j].imag + a[i, k].imag * b[k, j].real
        d[i, j] = complex(sum_real, sum_imag)

# Crear los objetos de memoria en la GPU
a_gpu = cuda.to_device(A)
b_gpu = cuda.to_device(B)
c_gpu = cuda.to_device(C)
d_gpu = cuda.device_array((n, p), dtype=np.complex128)

# Definir el tamaño del bloque y de la grilla
block = (16, 16)
grid = ((n + block[0] - 1) // block[0], (p + block[1] - 1) // block[1])

# Ejecutar el kernel de CUDA en la GPU
start_time = time.time()
matrix_mult[grid, block](a_gpu, b_gpu, c_gpu, d_gpu)
cuda_time = time.time() - start_time

# Copiar el resultado desde la GPU a la CPU
D_gpu = d_gpu.copy_to_host()

# Calcular la diferencia entre los resultados
diff = np.abs(D_numpy - D_gpu)

# Verificar si la diferencia es menor que una tolerancia dada
tolerance = 1e-3
is_close = np.all(diff < tolerance)

# Imprimir los resultados
print("Resultado de NumPy:")
print(D_numpy)

print("Resultado de Numba y CUDA:")
print(D_gpu)

# Imprimir el resultado de la verificación
if is_close:
    print("El resultado es cercano entre NumPy y Numba+CUDA.")
else:
    print("El resultado difiere más allá de la tolerancia entre NumPy y Numba+CUDA.")

# Imprimir los tiempos de ejecución
print("Tiempo de ejecución con NumPy: %f segundos" % numpy_time)
print("Tiempo de ejecución con Numba y CUDA: %f segundos" % cuda_time)
