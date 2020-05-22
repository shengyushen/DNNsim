cmake --build cmake-build-release/ --target all
cd cmake-build-release/
#make VERBOSE=1
/usr/bin/c++    -fopenmp -O3    CMakeFiles/DNNsim.dir/main.cpp.o  -o ../DNNsim -rdynamic base/libbase.a core/libcore.a sys/libsys.a base/libbase.a sys/libsys.a proto/libproto.a /usr/local/lib/libprotobuf.a
cd ..

