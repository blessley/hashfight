NVCC  = nvcc
ARCH  = sm_35
STD   = c++11
FLAGS = -O3 -std=$(STD) -arch=$(ARCH)

SortSearch: SortSearch.cu
	$(NVCC) $(FLAGS) SortSearch.cu -o SortSearch

