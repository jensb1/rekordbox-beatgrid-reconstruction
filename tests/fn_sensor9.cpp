#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
typedef void (*Fn)(void*, float*, int, void*);
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int,int,int,int,int);
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  FILE* fa=fopen(argv[1],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  int chunk=argc>2?atoi(argv[2]):ns; if(chunk>ns) chunk=ns;
  char self[0x800]; memset(self,0,sizeof(self));
  ((Ctor)(base+(0x100f5c5e0ULL-0x100000000ULL)))(self);
  ((Create)(base+(0x100f5c78cULL-0x100000000ULL)))(self,44100,0,ns,8,6);
  ((Fn)(base+(0x100f49f7cULL-0x100000000ULL)))(self,a,chunk,NULL);
  ((void(*)(void*))(base+(0x100f4a620ULL-0x100000000ULL)))(self);   // selectBand()
  printf("chunk=%d numSamples=%d\n",chunk,ns);
  for(int off=0;off<0x600;off+=8){
    uintptr_t b=*(uintptr_t*)(self+off), e=*(uintptr_t*)(self+off+8);
    if(b>0x10000000ULL && e>b && (e-b)<20000000ULL){
      long d=(long)(e-b);
      int* p0=(int*)b;
      printf("  +0x%03x: bytes=%ld  %s  first ints:",off,d,(d%24==0)?"(24-stride)":"");
      for(int k=0;k<4 && (long)(k*4)<d;k++) printf(" %d",p0[k]);
      printf("\n");
    }
  }
  return 0;
}
