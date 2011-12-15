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
  

#include <iostream>
#include <fstream>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/version.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

using namespace std;
using boost::shared_ptr;
using namespace boost::filesystem;
using namespace boost::archive;

template<class Archive> 
class Parent {

	friend class boost::serialization::access;	
	virtual void save(Archive & ar,const unsigned int version) const
	  {cout<<"Parent::save"<<endl;}
	virtual void load(Archive & ar,const unsigned int version)
	  {cout<<"Parent::load"<<endl;}
	BOOST_SERIALIZATION_SPLIT_MEMBER()
public:
Parent() {cout<<"Parent()"<<endl;}
virtual ~Parent() {cout<<"~Parent()"<<endl;}
};

template<class Archive>
class Child : public Parent<Archive> {
	friend class boost::serialization::access;
	void save(Archive & ar,const unsigned int version) const
	  {cout<<"Child::save"<<endl;}
	void load(Archive & ar,const unsigned int version)
	  {cout<<"Child::load"<<endl;}
	BOOST_SERIALIZATION_SPLIT_MEMBER()
public:
	Child() {cout<<"Child()"<<endl;}
	~Child() {cout<<"~Child()"<<endl;}
};

int main()
{
	boost::shared_ptr<Parent<text_oarchive>> p(new Child<text_oarchive>());
	ofstream out("test.txt");
	text_oarchive ofs(out);
	ofs << *p;
}