COMPILER  = g++
STD       = c++11
FLAGS     = -O3 -std=$(STD) -mavx
INCLUDE   = /opt/intel/tbb/include 
LIB       = /opt/intel/tbb/lib

HashMap: HashMap.cxx
	${COMPILER} ${FLAGS} -o HashMap HashMap.cxx -I${INCLUDE} -L${LIB} -ltbb -lrt

