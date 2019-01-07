COMPILER    = g++
STD         = c++11
FLAGS       = -O3 -std=$(STD) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -march=skylake-avx512
TBB_INCL    = /glade/u/home/shaomeng/Install/tbb2018_20180618oss/include
TBB_LIB     = /glade/u/home/shaomeng/Install/tbb2018_20180618oss/lib/intel64/gcc4.7
THRUST_INCL = /glade/work/shaomeng/thrust

SortSearch: SortSearch.cpp
	LD_PRELOAD=libtbbmalloc_proxy.so.2 ;\
	$(COMPILER) $(FLAGS) SortSearch.cpp -o SortSearch -I${THRUST_INCL} -I${TBB_INCL} -L${TBB_LIB} -ltbb -ltbbmalloc -ltbbmalloc_proxy -lrt -Wl,-rpath=${TBB_LIB} 

