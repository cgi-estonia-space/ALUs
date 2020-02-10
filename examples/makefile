all: inverter

inverter: inverter.cpp
	g++ -c inverter.cpp
	g++ -o inverter inverter.o -lgdal

clean:
	rm -f inverter.o
	rm -f inverter
