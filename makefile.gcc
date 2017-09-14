CXX=gcc
CXXFLAGS=-std=c++11 -O3 -g 
LFLAGS=-lfftw3 -lm

dnls: clean
	$(CXX) $(CXXFLAGS) $@.cpp  -o $@ $(LFLAGS)

%.o:	%.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(LFLAGS)

clean:
	rm -f *~ *.o dnls *.dat
