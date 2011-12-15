
.SUFFIXES:

DEBUG = 0
# use strict floating point, results for some points are different with strict off
FP_STRICT = 1

# unix
ifeq ($(SHELL),/bin/sh)
	CC = gcc
	CXX = g++
# windows
else
	VC_VERSION = 10

	CC = cl.exe
	CXX = cl.exe
	NVCC = nvcc.exe -m32 -arch=sm_20
#--compiler-options /MD
	LINK = link.exe
	ifeq ($(DEBUG), 0)
		CXXFLAGS = /nologo /O2 /MD /W3 /GS- /EHsc /Zm1000 /DNDEBUG /WL
		LINKFLAGS = /DLL /nologo /INCREMENTAL:NO
		NVCCFLAGS = 
	else
		CXXFLAGS = /nologo /Od /MD /W3 /GS- /Z7 /EHsc /Zm1000 /WL
		LINKFLAGS = /DLL /nologo /INCREMENTAL:NO /DEBUG
		NVCCFLAGS = -g
	endif
	ifeq ($(FP_STRICT), 1)
		CXXFLAGS += /fp:strict /DFP_STRICT
	endif
	ARBB_INC_DIR = arbb/include
	ARBB_LIB_DIR = arbb/lib/ia32
	CUDA_INC_DIR = "C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA/v4.0/include"
	CUDA_LIB_DIR = "C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/CUDA/v4.0/lib/Win32"
	GSL_INC_DIR = "C:/temp/gsl-1.15/gsl-1.15"
	GSL_LIB_DIR = "C:/temp/gsl-1.15/gsl-1.15/build.vc10/dll/Win32/Release"
	BOOST_INC_DIR = "c:/boost/boost_1_44"
	PYTHON_DIR = c:/Python26
	PYUBLAS_INC_DIR = $(PYTHON_DIR)/lib/site-packages/PyUblas-2011.1-py2.6-win32.egg/include
	INCLUDES = -I$(BOOST_INC_DIR) -I$(PYUBLAS_INC_DIR) \
		-Ilocal/include -IC:/Python26/lib/site-packages/numpy/core/include -IC:/Python26/include \
		-IC:/Python26/PC -I$(ARBB_INC_DIR)
	TBB_LIB_DIR_VC9 = tbb30_018oss/lib/ia32/vc9
	TBB_LIB_DIR_VC10 = tbb30_018oss/lib/ia32/vc10
	BOOST_PYTHON_LIB_VC9 = boost_python-vc90-mt-1_44.lib
	BOOST_PYTHON_LIB_VC10 = boost_python-vc100-mt-1_44.lib
#	BOOST_PYTHON_LIB_VC9 = boost_python-vc90-mt-1_46_1.lib
#	BOOST_PYTHON_LIB_VC10 = boost_python-vc100-mt-1_46_1.lib
ifeq ($(VC_VERSION), 9)
	TBB_LIB_DIR = $(TBB_LIB_DIR_VC9)
	BOOST_PYTHON_LIB = $(BOOST_PYTHON_LIB_VC9)
else
	TBB_LIB_DIR = $(TBB_LIB_DIR_VC10)
	BOOST_PYTHON_LIB = $(BOOST_PYTHON_LIB_VC10)
endif
	BOOST_LIB_DIR = "local/boost_1_44/lib"
#	BOOST_LIB_DIR = "c:/boost/boost_1_46_1/lib"
	LIB_DIRS = /LIBPATH:$(BOOST_LIB_DIR) /LIBPATH:C:/Python26/libs \
		/LIBPATH:C:/Python26/PCbuild /LIBPATH:$(TBB_LIB_DIR) /LIBPATH:$(ARBB_LIB_DIR)
	LIBS = tbb.lib $(BOOST_PYTHON_LIB) arbb.lib
	BUILD_DIR = build
endif

_ponzi2_fns.pyd: ponzi2_fns.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) $< /OUT:$@ \
		/IMPLIB:ponzi2_fns.lib \
		/MANIFESTFILE:ponzi2_fns.pyd.manifest

_myfuncs.pyd: myfuncs.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) $< /OUT:$@ \
		/IMPLIB:myfuncs.lib \
		/MANIFESTFILE:myfuncs.pyd.manifest

_ponzi3_fns.pyd: ponzi3_fns.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) $< /OUT:$@ \
		/IMPLIB:ponzi3_fns.lib \
		/MANIFESTFILE:ponzi3_fns.pyd.manifest

_debugMsg.pyd: debugMsg.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) $< /OUT:$@ \
		/IMPLIB:debugMsg.lib \
		/MANIFESTFILE:debugMsg.pyd.manifest

debugMsgFiles = _debugMsg.pyd debugMsg.obj debugMsg.lib debugMsg.pyd.manifest debugMsg.pdb

_maximizer.pyd: maximizer.obj _debugMsg.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib $< /OUT:$@ \
		/IMPLIB:maximizer.lib \
		/MANIFESTFILE:maximizer.pyd.manifest

maximizerFiles = _maximizer.pyd maximizer.obj maximizer.lib maximizer.pyd.manifest maximizer.pdb

_ponziProblem.pyd: ponziProblem.obj _maximizer.pyd _debugMsg.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib maximizer.lib $< /OUT:$@ \
		/IMPLIB:ponziProblem.lib \
		/MANIFESTFILE:ponziProblem.pyd.manifest

_bankProblem.pyd: bankProblem.obj _maximizer.pyd _debugMsg.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib maximizer.lib $< /OUT:$@ \
		/IMPLIB:bankProblem.lib \
		/MANIFESTFILE:bankProblem.pyd.manifest

_optDividends.pyd: optDividends.obj _maximizer.pyd _debugMsg.pyd _myfuncs.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib maximizer.lib myfuncs.lib $< /OUT:$@ \
		/IMPLIB:optDividends.lib \
		/MANIFESTFILE:optDividends.pyd.manifest

_merton.pyd: merton.obj _maximizer.pyd _debugMsg.pyd _myfuncs.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib maximizer.lib myfuncs.lib $< /OUT:$@ \
		/IMPLIB:merton.lib \
		/MANIFESTFILE:merton.pyd.manifest

#  _testCuda.pyd
_consumptionSavings.pyd: consumptionSavings.obj _maximizer.pyd _debugMsg.pyd _myfuncs.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) debugMsg.lib maximizer.lib myfuncs.lib \
		$< /OUT:$@ \
		/IMPLIB:consumptionSavings.lib \
		/MANIFESTFILE:consumptionSavings.pyd.manifest

_testCuda.pyd: testCuda.obj cudaMonteCarlo.dll _myfuncs.pyd
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) \
	$(LIBS) myfuncs.lib cudaMonteCarlo.lib $< /OUT:$@ \
		/IMPLIB:testCuda.lib \
		/MANIFESTFILE:testCuda.pyd.manifest

cudaMonteCarlo.dll: cudaMonteCarlo.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) /LIBPATH:$(CUDA_LIB_DIR) cuda.lib cudart.lib \
	$(LIBS) $< /OUT:$@ \
		/IMPLIB:cudaMonteCarlo.lib \
		/MANIFESTFILE:cudaMonteCarlo.dll.manifest

_test_arbb.pyd: test_arbb.obj
	$(LINK) $(LINKFLAGS) $(LIB_DIRS) $(LIBS) $< /OUT:$@ \
		/IMPLIB:test_arbb.lib \
		/MANIFESTFILE:test_arbb.pyd.manifest

CUDA_FILES = cudaMonteCarlo.obj cudaMonteCarlo.cu.cpp cudaMonteCarlo.dll cudaMonteCarlo.lib

cudaMonteCarlo.obj: cudaMonteCarlo.cu
	$(NVCC) -cuda $(NVCCFLAGS) $<
	$(CXX) /c $(CXXFLAGS) $(INCLUDES) /TpcudaMonteCarlo.cu.cpp -Fo$@

testcu:	testcu.cu
	$(NVCC) $< -o $@

testDebugCuda:	testDebugCuda.cu
	$(NVCC) -g -G $< -o $@

cartesian: cartesian_product.cu
	$(NVCC) $< -o $@

%.obj: %.cpp %.h
	$(CXX) /c $(CXXFLAGS) $(INCLUDES) /Tp$< -Fo$@

%.obj: %.cpp
	$(CXX) /c $(CXXFLAGS) $(INCLUDES) /Tp$< -Fo$@

testgsl: testgsl.cpp
	$(CXX) /c $(CXXFLAGS) $(INCLUDES) -I$(GSL_INC_DIR) /Tp$< -Fotestgsl.obj
	$(LINK) $(LIB_DIRS) $(LIBS) /LIBPATH:$(GSL_LIB_DIR) gsl.lib testgsl.obj /OUT:$@.exe

test1: test1.cpp
	$(CXX) /c $(CXXFLAGS) $(INCLUDES) -I$(GSL_INC_DIR) /Tp$< -Fotestgsl.obj
	$(LINK) $(LIB_DIRS) $(LIBS) /LIBPATH:$(GSL_LIB_DIR) gsl.lib testgsl.obj /OUT:$@.exe

TARGETS = debugMsg maximizer ponziProblem bankProblem ponzi2_fns ponzi3_fns myfuncs \
	consumptionSavings test_arbb optDividends \
# testCuda merton

ALL_FILES = $(foreach lib, $(TARGETS), _$(lib).pyd $(lib).obj $(lib).lib $(lib).pyd.manifest $(lib).pdb)
all: $(foreach lib, $(TARGETS), _$(lib).pyd)

clean:
# unix
ifeq ($(SHELL),/bin/sh)
	rm -f $(ALL_FILES) $(CUDA_FILES)

# windows
else
	del $(ALL_FILES) $(CUDA_FILES)
endif


