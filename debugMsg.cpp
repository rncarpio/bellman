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
  


#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <float.h>
#include <iostream>
#include <boost/python.hpp>
#include <mds_utils/python/fileobj.hpp>
#include "debugMsg.h"

using namespace boost;
using namespace python;
using namespace std;
namespace mdspy = mds_utils::python;

//static boost::python::object g_OutputFileObj;
static mdspy::FileObj *g_pOutputFileObj = NULL;

// set output file for debug messages
void setOutputFile(object &pyfile) {
  if (g_pOutputFileObj != NULL) {
    delete g_pOutputFileObj;
  }
  g_pOutputFileObj = new mdspy::FileObj(pyfile);
}

// debug message
void DebugMsg(char *format, ...) {
  va_list args;
  char pcBuf[1024+1];
  if (g_pOutputFileObj != NULL) {
    ostream &ofs(*g_pOutputFileObj);

    va_start(args, format);
    int nBytesWritten = vsnprintf_s(pcBuf, 1024+1, 1024, format, args);
    va_end(args);	
    ofs.write(pcBuf, nBytesWritten);
  }

}

BOOST_PYTHON_MODULE(_debugMsg)
{                              
  boost::python::def("setOutputFile", setOutputFile);
}                                          

