/**************************************************************************
 Name        : dnls.cu
 Author      : J.D. Bodyfelt
 Version     :
 Copyright   : (c) 2016, Massey University
 Description :
 **************************************************************************/

#include "dnls.cuh"

int main(void)
{
	cudaSetDevice(1);
	Time tobj(1e-2, 1e3, false);
	Lattice DNLS(0.0f, 4.0f);
	std::string fbase("Test");
	DNLS.Evolve(tobj, fbase);

	return 0;
}

