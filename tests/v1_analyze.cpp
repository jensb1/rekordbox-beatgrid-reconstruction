#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mach-o/dyld.h>
#include <initializer_list>
typedef void (*Ctor)(void*, int);
typedef int (*CreateFn)(void*, bool, unsigned long, int, int, bool, bool, bool);
typedef int (*SetSampleFn)(void*, float*, float*, int);
typedef void (*AnalyzeFn)(void*);
struct BA { int pos; int pad; float iv; int a; int b; unsigned char flag,c1,c2,c3; };
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  FILE* fa=fopen(argv[1],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  char w[0x100]; memset(w,0,sizeof(w));
  ((Ctor)(base+(0x101b766b4ULL-0x100000000ULL)))(w,1);
  int cok=((CreateFn)(base+(0x101b766f0ULL-0x100000000ULL)))(w,1,44100,ns,2,1,1,0);
  int sok=((SetSampleFn)(base+(0x101b76850ULL-0x100000000ULL)))(w,a,a,ns);
  void* an=*(void**)(w+8);
  fprintf(stderr,"create=%d setsample=%d analyzer=%p ver=%d\n",cok,sok,an,*(int*)(w+0x14));
  for(int off : {0x58,0xe8,0x2c0}){
    BA* b=*(BA**)((char*)an+off); BA* e=*(BA**)((char*)an+off+8);
    fprintf(stderr,"  vec +0x%x count=%ld\n",off,(long)(((char*)e-(char*)b)/24));
  }
  ((AnalyzeFn)(base+(0x100efeb74ULL-0x100000000ULL)))(an);
  fprintf(stderr,"after v1 analyze():\n");
  for(int off : {0x58,0xe8,0x2c0}){
    BA* b=*(BA**)((char*)an+off); BA* e=*(BA**)((char*)an+off+8);
    long n=(long)(((char*)e-(char*)b)/24);
    fprintf(stderr,"  vec +0x%x count=%ld",off,n);
    for(long i=0;i<n && i<5;i++) fprintf(stderr," %d",b[i].pos);
    fprintf(stderr,"\n");
  }
  return 0;
}
