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
  

// data structures specific to the bank problem

#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <iostream>
#include <boost/python.hpp>
#include <boost/python/dict.hpp>

#include "bankProblem.h"
#include "maximizer.h"
#include "debugMsg.h"

namespace bpl = boost::python;
using namespace pyublas;	

BankParams3::BankParams3(double beta, double rFast, double rSlow,
           DoublePyArray const &SlowOutFrac,
		   DoublePyArray const &FastOutFrac, DoublePyArray const &FastInFrac,
		   DoublePyArray const &ProbSpace, DoublePyArray const &BankruptcyPenalty) {
  m_beta = beta;
  m_pPrevIter = NULL;
  m_rFast = rFast;
  m_rSlow = rSlow;
  
  // check random variable probabilities sum to 1.0
  if (std::accumulate(ProbSpace.begin(), ProbSpace.end(), 0.0) != 1.0) {
#ifdef FP_STRICT
    throw std::invalid_argument("ProbSpace doesn't sum to 1.0");
#else
    printf("warning: ProbSpace doesn't sum to 1.0\n");
#endif //FP_STRICT
  }
  // check array sizes match
  if (ProbSpace.size() != SlowOutFrac.size()) { throw std::invalid_argument("array sizes don't match"); }  
  if (ProbSpace.size() != FastOutFrac.size()) { throw std::invalid_argument("array sizes don't match"); }
  if (ProbSpace.size() != FastInFrac.size()) { throw std::invalid_argument("array sizes don't match"); }    
  m_SlowOutFrac = SlowOutFrac;  
  m_FastOutFrac = FastOutFrac;
  m_FastInFrac = FastInFrac;    
  m_ProbSpace = ProbSpace;
  if (BankruptcyPenalty.size() != 3) { throw std::invalid_argument("bankruptcy penalty must have size=3"); }
  for (int i=0; i<3; i++) {
    m_BankruptcyPenalty[i] = BankruptcyPenalty.sub(i);
  }
}

void BankParams3::setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray) {
  m_StateGrid1 = bpl::extract<DoublePyArray>(stateGridList[0]);
  m_StateGrid2 = bpl::extract<DoublePyArray>(stateGridList[1]);  
  m_pStateGrid1 = (PyArrayObject const*) m_StateGrid1.data().handle().get();
  m_pStateGrid2 = (PyArrayObject const*) m_StateGrid2.data().handle().get();  
  m_PrevIterArray = WArray;	
  m_pPrevIter = (PyArrayObject const*) m_PrevIterArray.data().handle().get();	  
  m_pPrevIterInterp.reset(new Interp2D(m_StateGrid1, m_StateGrid2, DoublePyMatrix(m_StateGrid1.size(), m_StateGrid2.size(), WArray)));
}

double BankParams3::objectiveFunction(DoubleVector const &controlVars) const {
  double d = controlVars[0];
  double slowInFrac = controlVars[1];	
  
  double EV = calc_EV(d, slowInFrac);
  double result = d + m_beta * EV;
  return result;
}

// M is actually M/F
// S is actually S/F
// M, S, F are all positive, but F is actually a short quantity, so an outflow of S is positive for M, while an outflow
//   of F is negative
double BankParams3::calc_EV(double d, double slowInFrac, DoubleVector *pNextM, DoubleVector *pNextS) const {  
  double M = m_M;
  double S = m_S;  
  assert(M >= 0.0);
  double rFast = m_rFast;
  double rSlow = m_rSlow;
  double V;
  double sum = 0.0;
  DoublePyArray::const_iterator i1, i2, i3, i5;
  DoubleVector::iterator i_nextM, i_nextS;
  // save the computed values for next period's state, if necessary
  if (pNextM != NULL) {
    pNextM->resize(m_ProbSpace.size());
    i_nextM = pNextM->begin();
  }
  if (pNextS != NULL) {
    pNextS->resize(m_ProbSpace.size());
    i_nextS = pNextS->begin();
  }

  for (i1=m_FastOutFrac.begin(), i2=m_FastInFrac.begin(), i3=m_SlowOutFrac.begin(), i5=m_ProbSpace.begin();
       i5 != m_ProbSpace.end(); i1++, i2++, i3++, i5++) {
    double fast_out_frac = *i1;
    double fast_in_frac = *i2;
    double slow_out_frac = *i3;
    double slow_in_frac = slowInFrac;
    double prob_space = *i5;
	double fast_growth = (1.0 + fast_in_frac - fast_out_frac) * (1.0 + rFast);
    double nextM = (M - d + slow_out_frac*S - slow_in_frac*S - fast_out_frac + fast_in_frac)/fast_growth;
	double nextS = (1.0 + slow_in_frac - slow_out_frac) * S * (1.0 + rSlow) / fast_growth;	  
    if (nextM <= 0.0) {
	  //V = 0.0;
	  V = (m_BankruptcyPenalty[0] * nextM) + (m_BankruptcyPenalty[1] * nextS) + m_BankruptcyPenalty[2];
	  nextS = -1.0;
	} else {	  
      //V = interp2d_grid(m_pStateGrid1, m_pStateGrid2, m_pPrevIter, nextM, nextS);
	  V = m_pPrevIterInterp->interp(nextM, nextS);
	}
	sum += prob_space * fast_growth * V;	// need to multiply by fast_growth since state variables are divided by F
	if (pNextM != NULL) {
	  (*i_nextM) = nextM;
	  i_nextM++;
	}
	if (pNextS != NULL) {
	  (*i_nextS) = nextS;
	  i_nextS++;
	}
  }
  return sum;
}	

bpl::tuple BankParams3::calc_EV_wrap(double d, double slowInFrac) {
  DoubleVector nextM, nextS;
  double EV = calc_EV(d, slowInFrac, &nextM, &nextS);
  return bpl::make_tuple(EV, nextM, nextS);
}

////////////////////////////////////////////////////////////////////////////////////////////////
// BankParams4 begin
////////////////////////////////////////////////////////////////////////////////////////////////

BankParams4::BankParams4(double beta, double rFast, double rSlow,
           DoublePyArray const &SlowOutFrac,
		   DoublePyArray const &FastOutFrac, DoublePyArray const &FastInFrac,
		   DoublePyArray const &ProbSpace, DoublePyArray const &BankruptcyPenalty, double PopGrowth) {
  m_beta = beta;
  m_pPrevIter = NULL;
  m_rFast = rFast;
  m_rSlow = rSlow;
  m_PopGrowth = PopGrowth;
  
  // check random variable probabilities sum to 1.0
  if (std::accumulate(ProbSpace.begin(), ProbSpace.end(), 0.0) != 1.0) {
#ifdef FP_STRICT
    throw std::invalid_argument("ProbSpace doesn't sum to 1.0");
#else
    printf("warning: ProbSpace doesn't sum to 1.0\n");
#endif //FP_STRICT
  }
  // check array sizes match
  if (ProbSpace.size() != SlowOutFrac.size()) { throw std::invalid_argument("array sizes don't match"); }  
  if (ProbSpace.size() != FastOutFrac.size()) { throw std::invalid_argument("array sizes don't match"); }
  if (ProbSpace.size() != FastInFrac.size()) { throw std::invalid_argument("array sizes don't match"); }    
  m_SlowOutFrac = SlowOutFrac;  
  m_FastOutFrac = FastOutFrac;
  m_FastInFrac = FastInFrac;    
  m_ProbSpace = ProbSpace;
  if (BankruptcyPenalty.size() != 3) { throw std::invalid_argument("bankruptcy penalty must have size=3"); }
  for (int i=0; i<3; i++) {
    m_BankruptcyPenalty[i] = BankruptcyPenalty.sub(i);
  }
}

void BankParams4::setPrevIteration(bpl::list const &stateGridList, DoublePyArray const &WArray) {
  m_StateGrid1 = bpl::extract<DoublePyArray>(stateGridList[0]);
  m_StateGrid2 = bpl::extract<DoublePyArray>(stateGridList[1]);  
  m_StateGrid3 = bpl::extract<DoublePyArray>(stateGridList[2]);  
  m_pStateGrid1 = (PyArrayObject const*) m_StateGrid1.data().handle().get();
  m_pStateGrid2 = (PyArrayObject const*) m_StateGrid2.data().handle().get();  
  m_pStateGrid3 = (PyArrayObject const*) m_StateGrid3.data().handle().get();  
  m_PrevIterArray = WArray;	
  m_pPrevIter = (PyArrayObject const*) m_PrevIterArray.data().handle().get();	  
  //m_pPrevIterInterp.reset(new Interp2D(m_StateGrid1, m_StateGrid2, DoublePyMatrix(m_StateGrid1.size(), m_StateGrid2.size(), WArray)));
}

double BankParams4::objectiveFunction(DoubleVector const &controlVars) const {
  double d = controlVars[0];
  double slowInFrac = controlVars[1];	
  
  double EV = calc_EV(d, slowInFrac);
  double result = d + m_beta * EV;
  return result;
}

// M is actually M/F
// S is actually S/F
// M, S, F are all positive, but F is actually a short quantity, so an outflow of S is positive for M, while an outflow
//   of F is negative
double BankParams4::calc_EV(double d, double slowInFrac, DoubleVector *pNextM, DoubleVector *pNextS, DoubleVector *pNextP) const {  
  double M = m_M;
  double S = m_S;  
  double P = m_P;
  assert(M >= 0.0);
  double rFast = m_rFast;
  double rSlow = m_rSlow;
  double pop_growth = m_PopGrowth;
  double V;
  double sum = 0.0;
  DoublePyArray::const_iterator i1, i2, i3, i5;
  DoubleVector::iterator i_nextM, i_nextS, i_nextP;
  // save the computed values for next period's state, if necessary
  if (pNextM != NULL) {
    pNextM->resize(m_ProbSpace.size());
    i_nextM = pNextM->begin();
  }
  if (pNextS != NULL) {
    pNextS->resize(m_ProbSpace.size());
    i_nextS = pNextS->begin();
  }
  if (pNextP != NULL) {
    pNextP->resize(m_ProbSpace.size());
    i_nextP = pNextP->begin();
  }

  for (i1=m_FastOutFrac.begin(), i2=m_FastInFrac.begin(), i3=m_SlowOutFrac.begin(), i5=m_ProbSpace.begin();
       i5 != m_ProbSpace.end(); i1++, i2++, i3++, i5++) {
    double fast_out_frac = *i1;
    double fast_in_frac = *i2;
    double slow_out_frac = *i3;
    double slow_in_frac = slowInFrac;
    double prob_space = *i5;
	double fast_growth = (1.0 + fast_in_frac*P - fast_out_frac) * (1.0 + rFast);
    double nextM = (M - d + slow_out_frac*S - slow_in_frac*S - fast_out_frac + fast_in_frac*P)/fast_growth;
	double nextS = (1.0 + slow_in_frac - slow_out_frac) * S * (1.0 + rSlow) / fast_growth;	  
	double nextP = (pop_growth * P) / fast_growth;
    if (nextM <= 0.0) {
	  //V = 0.0;
	  V = (m_BankruptcyPenalty[0] * nextM) + (m_BankruptcyPenalty[1] * nextS) + m_BankruptcyPenalty[2];
	  nextS = -1.0;
	} else {	  
      V = interp3d_grid(m_pStateGrid1, m_pStateGrid2, m_pStateGrid3, m_pPrevIter, nextM, nextS, nextP);
	  //V = m_pPrevIterInterp->interp(nextM, nextS);
	}
	sum += prob_space * fast_growth * V;	// need to multiply by fast_growth since state variables are divided by F
	if (pNextM != NULL) {
	  (*i_nextM) = nextM;
	  i_nextM++;
	}
	if (pNextS != NULL) {
	  (*i_nextS) = nextS;
	  i_nextS++;
	}
	if (pNextP != NULL) {
	  (*i_nextP) = nextP;
	  i_nextP++;
	}
	
  }
  return sum;
}	

bpl::tuple BankParams4::calc_EV_wrap(double d, double slowInFrac) {
  DoubleVector nextM, nextS, nextP;
  double EV = calc_EV(d, slowInFrac, &nextM, &nextS, &nextP);
  return bpl::make_tuple(EV, nextM, nextS, nextP);
}


BOOST_PYTHON_MODULE(_bankProblem)
{                              
  bpl::class_<BankParams3, bpl::bases<BellmanParams>>("BankParams3", bpl::init<double, double, double,
           DoublePyArray, DoublePyArray, DoublePyArray, 
		   DoublePyArray, DoublePyArray>())
        .def("calc_EV", &BankParams3::calc_EV_wrap)
    ;  
  bpl::class_<BankParams4, bpl::bases<BellmanParams>>("BankParams4", bpl::init<double, double, double,
           DoublePyArray, DoublePyArray, DoublePyArray, 
		   DoublePyArray, DoublePyArray, double>())
        .def("calc_EV", &BankParams4::calc_EV_wrap)
    ;  

}                                          
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  