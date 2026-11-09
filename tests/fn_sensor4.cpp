#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mach-o/dyld.h>
struct BA { int pos; int pad; float iv; int a; int b; unsigned char flag,c1,c2,c3; };
typedef void (*Fn)(void*, float*, int, void*);
typedef void (*Ctor)(void*);
typedef int (*Create)(void*, int,int,int,int,int);
struct V { BA* b; BA* e; BA* c; };
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Fn f=(Fn)(base+(0x100f49f7cULL-0x100000000ULL));
  FILE* fa=fopen(argv[1],"rb"); fseek(fa,0,SEEK_END); long asz=ftell(fa); fseek(fa,0,SEEK_SET);
  float* a=(float*)malloc(asz); fread(a,1,asz,fa); fclose(fa); int ns=asz/4;
  int chunk = argc>2?atoi(argv[2]):ns;
  if(chunk>ns) chunk=ns;
  char self[0x800]; memset(self,0,sizeof(self));
  ((Ctor)(base+(0x100f5c5e0ULL-0x100000000ULL)))(self);
  int a1=argc>3?atoi(argv[3]):44100;
  // exact arguments from the real call site (Analyzer::create, 0x100f32b10)
  int cok=((Create)(base+(0x100f5c78cULL-0x100000000ULL)))(self,a1,0,ns,8,6);
  fprintf(stderr,"create=%d\n",cok);
  std::vector<BA> out(chunk+64);
  V vec={out.data(), out.data(), out.data()+out.size()};
  f(self, a, chunk, NULL);
  long n=vec.e-vec.b;
  printf("chunk=%d peaks=%ld\n",chunk,n);
  for(long i=0;i<n && i<8;i++) printf("  %ld: pos=%d iv=%.4f a=%d b=%d f=%d\n",i,out[i].pos,out[i].iv,out[i].a,out[i].b,out[i].flag);
  return 0;
}
