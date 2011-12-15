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
  


#include <Python.h>

PyObject *MyCommand(PyObject *self, PyObject *args)
{ 
  PyObject *result = NULL;
  long a, b;

  if (PyArg_ParseTuple(args, "ii", &a, &b)) {
    result = Py_BuildValue("i", a + b);
  }
  return result;
}

PyMethodDef methods[] = {
  {"add", MyCommand, METH_VARARGS},
  {NULL, NULL},
};

#ifdef __cplusplus
extern "C" {
#endif

void initdemo() {
  (void)Py_InitModule("demo", methods);
}

#ifdef __cplusplus
}
#endif
