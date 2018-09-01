NVCC  = nvcc
ARCH  = sm_35
STD   = c++11
FLAGS = -O3 -std=$(STD) -arch=$(ARCH) -Xcompiler -fopenmp --expt-extended-lambda 

plan:
	$(NVCC) $(FLAGS) SortSearch.cu -o SortSearch

