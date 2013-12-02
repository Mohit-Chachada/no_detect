proj: num_extract.o main.o
	g++ `pkg-config opencv --cflags` -o proj num_extract.o main.o `pkg-config opencv --libs` -g 
calibrate.o: num_extract.hpp num_extract.cpp
	g++ `pkg-config opencv --cflags` -c num_extract.cpp `pkg-config opencv --libs` -g
main.o: num_extract.hpp main.cpp
	g++ `pkg-config opencv --cflags` -c main.cpp `pkg-config opencv --libs` -g
clean:
	rm -f *.o core proj
