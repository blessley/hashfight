COMPILER  = g++
STD   = c++11
FLAGS = -O3 -std=$(STD)

GenerateData:
	$(COMPILER) $(FLAGS) GenerateData.cxx mt19937ar.cpp random_numbers.cpp  -o GenerateData -I.

clean:
	rm -f *.o
