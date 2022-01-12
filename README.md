# KNN OCR
Распознавание текста методом K-ближайших соседей, распараллеленное с помощью CUDA и OpenMP.

# Производительность
## Производительность загрузки тренировочного набора данных в память (50 прогонов)
Синхронно: 4058 мс.
OpenMP + Blocking CUDA memcpy: 1703 мс.
OpenMP + Async CUDA memcpy: 728 мс.

# Литература
CUDA 7: стандартный Stream в каждый Thread!
https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/