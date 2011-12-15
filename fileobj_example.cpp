
// fileobj_example.cpp
// 
// Copyright (c) 2009 - Michele De Stefano (micdestefano@gmail.com)
// 
// Distributed under the MIT License (See accompanying file COPYING)

/**
 * \example	fileobj_example.cpp
 * 
 * A simple Python extension module that shows the mds_utils::python::FileObj usage.
 */


#include <boost/python.hpp>
#include <mds_utils/python/fileobj.hpp>
#include <iostream>

using namespace boost;
using namespace python;
using namespace std;

namespace mdspy = mds_utils::python;


double read_double(object &py_file) {

	mdspy::FileObj		fobj(py_file);
	
	istream				&ifs(fobj);
	
	double				val;
	
	ifs.read(reinterpret_cast<char*>(&val),sizeof(double));

	return val;
}



void write_double(object &py_file,double val) {

	mdspy::FileObj		fobj(py_file);
	
	ostream				&ofs(fobj);
	
	ofs.write(reinterpret_cast<char*>(const_cast<double*>(&val)),sizeof(double));
}


BOOST_PYTHON_MODULE(fileobj_example) {

	def("read_double",read_double);
	def("write_double",write_double);
}

