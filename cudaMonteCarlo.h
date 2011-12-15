//
// Copyright (c) 2011 Ronaldo Carpio
//                                     
// Permission to use, copy, modify, distribute and sell this software
// and its documentation for any purpose is hereby granted without fee,
// provided that the above copyright notice appear in all copies and   
// that both that copyright notice and this permission notice appear
// in supporting documentation.  The authors make no representations
// about the suitability of this software for any purpose.          
// It is provided "as is" without express or implied warranty.
//                                                            
  


#ifndef _cudaMonteCarlo_h_
#define _cudaMonteCarlo_h_

#include <vector>

#define DLLEXPORT __declspec(dllexport)

typedef std::vector<double> DoubleVector;
	  
void DLLEXPORT cuda_setup(const double *pFGridBegin, const double *pFGridEnd, const double *pFValsBegin, const double *pFValsEnd, 
  const double *pRandomDrawsBegin, const double *pRandomDrawsEnd);
void DLLEXPORT interp1d_vector(const double *pArgBegin, const double *pArgEnd, double *pResultBegin); 
double DLLEXPORT mytest1(double x, double y);
void DLLEXPORT test_setup(DoubleVector &fGrid, DoubleVector &fVals, DoubleVector &fSlopes);
void DLLEXPORT deviceReset();
double DLLEXPORT test_interp2(double xi);
double DLLEXPORT cuda_calcEV(double s1, double s2, double W, double expMean1);

#endif //_cudaMonteCarlo_h