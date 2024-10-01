import matplotlib.pyplot as plt

# Matrix sizes (assuming M=N=K for all cases)
matrix_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# TFlops data for different implementations
tflops_naive = [0.42048, 1.94245, 4.42909, 4.63712, 4.78555, 4.76005, 3.25899, 3.28565]
tflops_float4 = [0.0969711, 0.406179, 1.69068, 6.86251, 13.8581, 12.2531, 11.9171, 11.4097]
tflops_8x8_float4 = [0.0562432, 0.0738832, 0.296355, 1.1839, 4.74396, 8.79142, 7.9129, 8.32315]
tflops_8x8_float4_2 = [0.117967, 0.38354, 1.58972, 6.45044, 25.9499, 21.6301, 19.8703, 19.1099]
tflops_8x8_float4_3 = [0.0903994, 0.206104, 0.823361, 3.23149, 3.65489, 3.85061, 3.82752, 3.8733]
tflops_smem_float4 = [0.247278, 1.11406, 4.69703, 19.1078, 33.2531, 32.5154, 31.988, 31.6517]
tflops_double_smem_float4 = [0.216512, 0.955379, 3.99391, 18.6094, 33.8312, 32.5972, 31.4139, 31.5265]
tflops_double_smem_2x1_float4 = [0.317381, 1.44273, 6.22994, 29.6216, 71.6691, 67.2297, 62.9372, 57.9009]
tflops_double_smem_4x1_float4 = [0.224404, 1.28326, 5.98985, 26.6505, 68.8064, 65.1629, 63.14, 58.8808]
tflops_sgemm_128x128x8 = [0.212438, 0.993139, 4.20332, 17.3069, 54.0235, 48.0013, 45.5683, 43.7465]
tflops_cublas = [0.650095, 3.62378, 17.78, 35.5562, 60.2139, 54.0386, 53.2204, 52.8366]

# Plot
plt.figure(figsize=(10, 6))

# Plot each implementation
plt.plot(matrix_sizes, tflops_cublas, label='cublas', marker='o')
plt.plot(matrix_sizes, tflops_naive, label='sgemm_naive', marker='o')
plt.plot(matrix_sizes, tflops_float4, label='sgemm_float4', marker='o')
plt.plot(matrix_sizes, tflops_8x8_float4, label='sgemm_8x8_float4_1reg', marker='o')
plt.plot(matrix_sizes, tflops_8x8_float4_2, label='sgemm_8x8_float4_4reg', marker='o')
plt.plot(matrix_sizes, tflops_smem_float4, label='sgemm_smem_4x4_float4_2reg', marker='o')
plt.plot(matrix_sizes, tflops_double_smem_float4, label='sgemm_double_smem_4x4_float4_2reg', marker='o')
plt.plot(matrix_sizes, tflops_sgemm_128x128x8, label='sgemm_double_smem_8x8_float4_4reg', marker='o')
plt.plot(matrix_sizes, tflops_double_smem_2x1_float4, label='sgemm_double_smem_8x4_float4_2reg', marker='o')
plt.plot(matrix_sizes, tflops_double_smem_4x1_float4, label='sgemm_double_smem_16x4_float4_4reg', marker='o')

# Add labels and title
plt.xlabel('Matrix Size (M=N=K)')
plt.ylabel('TFLOPS')
plt.title('TFLOPS vs Matrix Size for GEMM Implementations')
plt.xscale('log')  # Use a logarithmic scale for the x-axis
plt.xticks(matrix_sizes, labels=matrix_sizes)
plt.legend(loc='best')
plt.grid(True, which='both', linestyle='--')

# Show the plot
plt.show()
