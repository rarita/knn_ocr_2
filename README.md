# KNN OCR
������������� ������ ������� K-��������� �������, ���������������� � ������� CUDA � OpenMP.

# ������������������
## ������������������ �������� �������������� ������ ������ � ������ (50 ��������)
���������: 4058 ��.
OpenMP + Blocking CUDA memcpy: 1703 ��.
OpenMP + Async CUDA memcpy: 728 ��.

# ����������
CUDA 7: ����������� Stream � ������ Thread!
https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/

https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/