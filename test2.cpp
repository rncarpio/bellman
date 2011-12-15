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
  

#include <cstdlib>
#include <time.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

void test2() {
  // generate 32M random numbers on the host
  std::vector<int> h_vec(32 << 20); 
  std::generate(h_vec.begin(), h_vec.end(), rand);
  // sort data on the device (846M keys per sec on GeForce GTX 480)
  std::sort(h_vec.begin(), h_vec.end());                        
}

int main(void) {
  time_t t1, t2, t3;
  t2 = time(NULL);
  test2();        
  t3 = time(NULL);

  printf("CPU: %d\n", t3-t2);
  return 0;                                               
}            
