#ifndef DNLS_H_
#define DNLS_H_

/*********************** Import Libraries **************************/
#include <stdlib.h>
#include <stdio.h>
#include<random>
#include<math.h>
#include<time.h>
#include<complex.h>
#include<fftw3.h>

/************************ Macro Definitions ***********************/
#ifndef NX
#define NX 1024				// Number of Lattice Sites
#endif

#ifndef NT
#define NT 1024				// Number of Time Records
#endif

#ifndef BATCH
#define BATCH 1024			// Number of Realizations
#endif

#ifndef TAU
#define TAU 6.283185307179586476925286766559	// 2*PI
#endif
/*********************** Macro Functions ***********************/


/************************ De-CUDA'd Kernels ************************/
void Evolve_Potential(dcomplex *Y, double *V, double g, dcomplex tau)
{
	for(int k=0; k<NX*BATCH; k++)
		Y[k] *= cexp( tau * (V[k] + g*pow( cuCabs(Y[k]), 2 )) );
}
/*===================================*/
void Evolve_Kinetic_Rotate(dcomplex *Y, double *ROT, dcomplex tau)
{
	for(int k=0; k<NX; k++){
		for(int j=0; j<BATCH; j++){
			Y[k + j*NX] *= cexp( tau * ROT[k] );
			Y[k + j*NX] /= (double)NX;
		}
	}
}
/********************** Class Definitions *********************/
class Time
{
public:
	std::complex<double> SIC[5];
	double t=0.0f, dt = 1e-2, tmax = 1e4, alf, tss;
	bool logf=false;
	clock_t tic;
	int j=0;
	/*=======================================================*/
	Time(double dt_, double tmax_, bool logf_) :
		dt(dt_), tmax(tmax_), logf(logf_)
	{
		if(logf)
		{
			alf = exp( log(tmax/dt) / (double)(NT-1) );
			tss = dt * pow( alf, j );
		}
		else
		{
			alf = (tmax/dt) / (double)NT;
			tss = (j+1) * alf;
		}
		if(dt > tss) {
			printf("ERROR: step (dt) > snapshot (tss)\n");
			exit(1);
		}
		tic = clock();
		double C[3] = {
				1.0f/6.0f,
				1.0f/2.0f,
				2.0f/3.0f };
		for(int k=0; k<3; k++) SIC[k] = I*dt*C[k];
		SIC[3] = 2.0f*SIC[0];
		SIC[4] = -1.0f*SIC[0];
	}
	/*=======================================================*/
	void update()
	{
		j+=1;
		if(logf){ tss = dt * pow( alf, j ); }
		else{ tss = (j+1) * alf; }
	}
	/*=======================================================*/
	void reset()
	{
		j=0; t=0.0f;
		if(logf) { tss = dt * pow( alf, j ); }
		else { tss = (j+1) * alf; }
	}
	/*=======================================================*/
	double toc(void)
	{ return ((double)(clock() - tic))/CLOCKS_PER_SEC; }
	/*=======================================================*/
};
/****************************************************************/
class Lattice
{
public:
	size_t SZ[3] = {
			NX*BATCH*sizeof(dcomplex),
			NX*BATCH*sizeof(double),
			NX*sizeof(double) };
	std::complex<double> *Y;
	double *V, *ROT;
	double g;
	cufftHandle plan;
	/*=======================================================*/
	Lattice(double g_, double W) : g(g_)
	/* At some point, we'll need a constructor that has input
	 * codes for ICs and different potentials.
	 * Also, need to have device query and perhaps setting
	 * NX,NT, & BATCH by results. */
	{
		cudaMalloc( (void**)&Y, SZ[0] );
		cudaMalloc( (void**)&V, SZ[1] );
		cudaMalloc( (void**)&ROT, SZ[2] );
		cufftPlan1d(&plan, NX, CUFFT_Z2Z, BATCH);
		//
		Build_Full_Packet_Random_Phase(1.0);
		Build_Anderson_Potential(W);
		Build_Rotator();
	}
	/*=======================================================*/
	~Lattice(void)
	{
		cufftDestroy(plan);
		cudaFree(Y); cudaFree(V);
		cudaFree(ROT);
	}
	/*=======================================================*/
	void Build_Full_Packet_Random_Phase(double S0)
	{
		std::default_random_engine gen;
		std::uniform_real_distribution<double> ang(0.0f, TAU);
		//
		dcomplex *cpuY = (dcomplex*) malloc(SZ[0]);
		for(int k=0; k<NX*BATCH; k++)
			cpuY[k] = cuCmul(
					make_cuDoubleComplex(sqrt(S0),0.0f),
					cuExp( make_cuDoubleComplex(0.0f, ang(gen)) )
					);
		cudaMemcpy(Y, cpuY, SZ[0], cudaMemcpyHostToDevice);
		free(cpuY);
	}
	void Build_Single_Site_Random_Phase(double S0)
	{
		std::default_random_engine gen;
		std::uniform_real_distribution<double> ang(0.0f, TAU);
		//
		dcomplex *cpuY = (dcomplex*) malloc(SZ[0]);
		double arg;
		for(int k=0; k<NX*BATCH; k++)
			cpuY[k] = make_cuDoubleComplex(0.0f,0.0f);
		for(int k=0; k<BATCH; k++)
		{
			arg = ang(gen);
			cpuY[(NX/2) + k*NX].x = sqrt(S0)*cos(arg);
			cpuY[(NX/2) + k*NX].y = sqrt(S0)*sin(arg);
		}
		cudaMemcpy(Y, cpuY, SZ[0], cudaMemcpyHostToDevice);
		free(cpuY);

	}
	/*=======================================================*/
	void Build_Anderson_Potential(double W)
	{	/* This will eventually need a non-CUDA branch! */
		std::default_random_engine gen;
		std::uniform_real_distribution<double> distro(-W/2.0,W/2.0);
		double *cpuV = (double*) malloc(SZ[1]);
		for(int k=0; k<NX*BATCH; k++) cpuV[k] = distro(gen);
		cudaMemcpy(V, cpuV, SZ[1], cudaMemcpyHostToDevice);
		free(cpuV);
	}
	/*=======================================================*/
	void Build_Rotator()
	{	/* This will eventually need a non-CUDA branch! */
		double *cpuROT = (double*) malloc(SZ[2]);
		for(int k=0; k<NX; k++)
			cpuROT[k] = 2.0f*cos( (TAU*k)/(double)NX );
		cudaMemcpy(ROT, cpuROT, SZ[2], cudaMemcpyHostToDevice);
		free(cpuROT);
	}
	/*=======================================================*/
	void Evolve(Time tobj, std::string fbase)
	{
		Evolve_Potential<<<NX,BATCH>>>(Y,V,g,tobj.SIC[0]);
		while(tobj.t < tobj.tmax)
		{
			Evolve_Kinetic(tobj.SIC[1]);
			Evolve_Potential<<<NX,BATCH>>>(Y,V,g,tobj.SIC[2]);
			Evolve_Kinetic(tobj.SIC[1]);
			Evolve_Potential<<<NX,BATCH>>>(Y,V,g,tobj.SIC[3]);
			tobj.t += tobj.dt;
			//
			if(tobj.t >= tobj.tss)
			{
				Evolve_Potential<<<NX,BATCH>>>(Y,V,g,tobj.SIC[4]);
				Y2Txt(tobj.t, tobj.j, fbase);
				Evolve_Potential<<<NX,BATCH>>>(Y,V,g,tobj.SIC[0]);
				tobj.update();
			}
		}
	}
	/*=======================================================*/
	void Evolve_Kinetic(dcomplex tau)
	{
		cufftExecZ2Z(plan,Y,Y,CUFFT_FORWARD);
		Evolve_Kinetic_Rotate<<<NX,BATCH>>>(Y,ROT,tau);
		cufftExecZ2Z(plan,Y,Y,CUFFT_INVERSE);
	}
	/*=======================================================*/
	void Y2Txt(double t, int rec, std::string fbase, bool eikon=true)
	{
		dcomplex *hY = (dcomplex*) malloc(SZ[0]);
		cudaMemcpy(hY,Y,SZ[0],cudaMemcpyDeviceToHost);
		std::string ext("_Y.dat");
		FILE *fid = fopen( (fbase+ext).c_str(), "a");
		fprintf(fid, "%.6e\t", t);
		if(eikon==true)
		{
		for(int k=0; k<NX*BATCH; k++)
			fprintf( fid, "%.6e\t%.6e\t", cuCabs(hY[k]), cuCarg(hY[k]) );
		}
		else
		{
		for(int k=0; k<NX*BATCH; k++)
			fprintf(fid, "%.6e\t%.6e\t", hY[k].x, hY[k].y);
		}
		fprintf(fid, "\n");
		fclose(fid);
		free(hY);
	}
	/*=======================================================*/
	void Y2Bin(int rec, std::string fbase)
	{
		dcomplex *hY = (dcomplex*) malloc(SZ[0]);
		cudaMemcpy(hY,Y,SZ[0],cudaMemcpyDeviceToHost);
		std::string ext("_Y.bin");
		FILE *fid = fopen( (fbase+ext).c_str(), "ab");
		fwrite(&hY, SZ[0],1,fid);
		fclose(fid);
		free(hY);
	}
	/*=======================================================*/
};

#endif /* DNLS_CUH_ */
