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
  

#include <pyublas/numpy.hpp>
                            
pyublas::numpy_vector<double> doublify(pyublas::numpy_vector<double> x)
{                                                                      
  return 2*x;
}            
 
BOOST_PYTHON_MODULE(testBoost)
{                              
  boost::python::def("doublify", doublify);
}                                          

