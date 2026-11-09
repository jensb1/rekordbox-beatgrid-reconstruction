#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int, int);
typedef void (*CBD)(void*, double, int);
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  double period=argc>1?atof(argv[1]):491803.27868852;
  int start=argc>2?atoi(argv[2]):536;
  char ua[0x400]; memset(ua,0,sizeof(ua));
  ((Ctor)(base+(0x100f02b3cULL-0x100000000ULL)))(ua);
  ((Create)(base+(0x100f02da0ULL-0x100000000ULL)))(ua,44100,44100*600);
  *(int*)(ua+0x20)=4000;   // count, createBeatData bails when < 1
  unsigned long long before[0x80]; memcpy(before,ua,sizeof(before));
  ((CBD)(base+(0x100f04cbcULL-0x100000000ULL)))(ua,period,start);
  printf("changed 8-byte words in ua[0..0x400):\n");
  for(int o=0;o<0x400;o+=8){
    unsigned long long a=before[o/8], b=*(unsigned long long*)(ua+o);
    if(a!=b) printf("  +0x%03x : %016llx -> %016llx\n",o,a,b);
  }
  int* b=*(int**)(ua+0x58); int* e=*(int**)(ua+0x60);
  long bytes=(char*)e-(char*)b;
  printf("vector +0x58: %ld bytes\n",bytes);
  for(long i=0;i<bytes/4 && i<12;i++) printf("  int[%ld] = %d\n", i, b[i]);
  return 0;
}
