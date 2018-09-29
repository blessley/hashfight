COMPILER  = g++
STD       = c++11
FLAGS     = -O3 -std=$(STD) -mavx
INCLUDE   = /usr/include/tbb 
LIB       = /usr/lib/

UnorderedMap: UnorderedMap.cxx
	${COMPILER} ${FLAGS} -o UnorderedMap UnorderedMap.cxx -I${INCLUDE} -L${LIB} -ltbb -ltbbmalloc -ltbbmalloc_proxy -lrt

