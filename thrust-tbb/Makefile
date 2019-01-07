COMPILER    = g++
STD         = c++11
FLAGS       = -O3 -std=$(STD) -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB
TBB_INCL    = /usr/include/tbb
TBB_LIB     = /usr/lib/x86_64-linux-gnu
THRUST_INCL = /home/users/blessley/thrust

SortSearch: SortSearch.cpp
	LD_PRELOAD=libtbbmalloc_proxy.so.2 ;\
	$(COMPILER) $(FLAGS) SortSearch.cpp -o SortSearch -I${THRUST_INCL} -I${TBB_INCL} -L${TBB_LIB} -ltbb -lrt -ltbbmalloc -ltbbmalloc_proxy

