#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
#include <initializer_list>
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
  int ns=44100*600;
  char ua[0x400]; memset(ua,0,sizeof(ua));
  ((Ctor)(base+(0x100f02b3cULL-0x100000000ULL)))(ua);
  int ok=((Create)(base+(0x100f02da0ULL-0x100000000ULL)))(ua,44100,ns);
  fprintf(stderr,"create=%d fields +8=%d +c=%d +1c=%d +20=%d\n",ok,
          *(int*)(ua+8),*(int*)(ua+0xc),*(int*)(ua+0x1c),*(int*)(ua+0x20));
  *(int*)(ua+0x20)=4000;                     // count, must be >= 1
  ((CBD)(base+(0x100f04cbcULL-0x100000000ULL)))(ua,period,start);
  for (int off : {0x58}) {
    for(int o=0x40;o<=0x80;o+=8) fprintf(stderr,"  ua+0x%02x = 0x%016llx\n",o,*(unsigned long long*)(ua+o));
  int* bb=*(int**)(ua+off); int* ee=*(int**)(ua+off+8);
    long nn=((char*)ee-(char*)bb);
    fprintf(stderr,"vector at +0x%x: bytes=%ld",off,nn);
    if (nn%24==0) {
      long cnt=nn/24;
      fprintf(stderr," (24-stride, count=%ld) first pos:",cnt);
      for(long i=0;i<cnt && i<3;i++){
        fprintf(stderr,"\n    [%ld] ints:", i);
        for(int k=0;k<6;k++) fprintf(stderr," %11d", ((int*)bb)[i*6+k]);
        fprintf(stderr," | doubles:");
        for(int k=0;k<3;k++) fprintf(stderr," %.6f", ((double*)bb)[i*3+k]);
      }
    }
    fprintf(stderr,"\n");
    if(off==0x60){ FILE* f=fopen("/tmp/cbd_out.bin","wb"); fwrite(bb,4,nn,f); fclose(f);}
  }
  int* b=*(int**)(ua+0x60);
  int* e=*(int**)(ua+0x68);
  long n=(char*)e-(char*)b; n/=4;
  fprintf(stderr,"vector at +0x60: count=%ld\n", n);
  for(long i=0;i<n && i<8;i++) fprintf(stderr,"  %ld: %d\n",i,b[i]);
  FILE* f=fopen("/tmp/cbd_out.bin","wb"); fwrite(b,4,n,f); fclose(f);
  return 0;
}
