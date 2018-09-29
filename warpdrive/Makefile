NVCC  = nvcc
ARCH  = sm_70
STD   = c++11
FLAGS = -O3 -std=$(STD) -arch=$(ARCH) -Xcompiler -fopenmp --expt-extended-lambda 

plan:
	$(NVCC) $(FLAGS) test_plan.cu -o test_plan

