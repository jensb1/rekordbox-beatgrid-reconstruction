#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mach-o/dyld.h>
struct BA { int pos; int pad; float iv; int bar; int x; int flag; };
typedef void (*Fn)(void*, void*, int, int, int);
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Fn f=(Fn)(base+(0x100f6a5fcULL-0x100000000ULL));
  int sr = argc>1?atoi(argv[1]):48000;
  int N=6;
  std::vector<BA> v(N); memset(v.data(),0,N*sizeof(BA));
  for(int i=0;i<N;i++){ v[i].pos=536+i*21688; v[i].iv=491.8f; v[i].bar=(i%4)+1; v[i].x=-1; }
  struct Vec { BA* b; BA* e; BA* c; } vec={v.data(), v.data()+N, v.data()+N};
  char self[0x40]; memset(self,0,sizeof(self));
  *(int*)(self+0)=1024; *(int*)(self+4)=sr;
  f(self,&vec,0,0,-1);
  int cnt=(int)(vec.e-vec.b);
  printf("sr=%d after: count=%d\n",sr,cnt);
  for(int i=0;i<cnt && i<8;i++)
    printf("  %d: pos=%d iv=%.6f bar=%d x=%d flag=%d\n", i, v[i].pos, v[i].iv, v[i].bar, v[i].x, v[i].flag);
  return 0;
}
