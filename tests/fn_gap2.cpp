#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mach-o/dyld.h>
struct BA { int pos; int pad; float iv; int a; int b; unsigned char flag,c1,c2,c3; };
typedef void (*Fn)(void*, void*, int, int, int);
struct V { BA* b; BA* e; BA* c; };
int main(int argc,char**argv){
  int seed=atoi(argv[1]); srand(seed);
  int jit=argc>2?atoi(argv[2]):50;
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Fn f=(Fn)(base+(0x100f6a5fcULL-0x100000000ULL));
  int N=24;
  std::vector<BA> v(N); memset(v.data(),0,N*sizeof(BA));
  int p=536;
  printf("in  :");
  for(int i=0;i<N;i++){
    p += 21688 + (rand()%(2*jit+1)) - jit;
    v[i].pos=p; v[i].iv=491.791382f; v[i].flag=0;
    if(i<6) printf(" %d",p);
  }
  printf("\n");
  V vec={v.data(), v.data()+N, v.data()+N};
  char self[0x40]; memset(self,0,sizeof(self)); *(int*)(self+4)=44100;
  f(self,&vec,0,0,-1);
  int cnt=(int)(vec.e-vec.b);
  printf("out (count=%d):",cnt); for(int i=0;i<cnt && i<6;i++) printf(" %d",v[i].pos); printf("\n");
  return 0;
}
