CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

main: mkdir dk_series clean

mkdir:
	mkdir -p ../rand_network
	mkdir -p ../bin

dk_series: network.o rewiring.o dk_series.o
	$(CXX) $(CXXFLAGS) -o ../bin/dk_series network.o rewiring.o dk_series.o -O3
dk_series.o: ./dk_series.cpp ./network.h ./rewiring.h
	$(CXX) $(CXXFLAGS) -c ./dk_series.cpp -O3
network.o: ./network.cpp ./network.h
	$(CXX) $(CXXFLAGS) -c ./network.cpp -O3
rewiring.o: ./rewiring.cpp ./network.h ./rewiring.h
	$(CXX) $(CXXFLAGS) -c ./rewiring.cpp -O3

clean:
	rm *.o