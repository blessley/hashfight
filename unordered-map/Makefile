COMPILER  = g++
STD       = c++11
FLAGS     = -O3 -std=$(STD) -march=skylake-avx512
INCLUDE   = /glade/u/home/shaomeng/Install/tbb2018_20180618oss/include
LIB       = /glade/u/home/shaomeng/Install/tbb2018_20180618oss/lib/intel64/gcc4.7

UnorderedMap: UnorderedMap.cxx
	${COMPILER} ${FLAGS} -o UnorderedMap UnorderedMap.cxx -I${INCLUDE} -L${LIB} -ltbb -ltbbmalloc -ltbbmalloc_proxy -lrt -Wl,-rpath=${LIB} 

