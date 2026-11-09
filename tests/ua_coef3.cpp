#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int, int);
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  int ns=argc>1?atoi(argv[1]):44100*600;
  char ua[0x400]; memset(ua,0,sizeof(ua));
  ((Ctor)(base+(0x100f02b3cULL-0x100000000ULL)))(ua);
  int ok=((Create)(base+(0x100f02da0ULL-0x100000000ULL)))(ua,44100,ns);
  fprintf(stderr,"create=%d\n",ok);
  // dump floats and look for biquad-like 5-tuples (b0,b1,b2,a1,a2)
  int offs[5]={0x88,0x90,0x98,0xa0,0xa8};
  for(int k=0;k<5;k++){
    char* p=*(char**)(ua+offs[k]);
    printf("filter %d @ ua+0x%02x -> %p\n",k,offs[k],(void*)p);
    for(int o=0;o<0x60;o+=4){
      float f=*(float*)(p+o);
      if(f!=0.0f && (f>-4.0f&&f<4.0f)) printf("   +0x%02x = %.9g\n",o,f);
    }
  }
  return 0;
}
