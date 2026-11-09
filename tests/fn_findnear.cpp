#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mach-o/dyld.h>
struct BA { int pos; int pad; float iv; int bar; int x; int flag; };
typedef int (*Fn)(void*, void*, int, int, int*);
int main(int argc,char**argv){
  int seed=atoi(argv[1]); srand(seed);
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Fn f=(Fn)(base+(0x100f5d00cULL-0x100000000ULL));
  for(int t=0;t<200;t++){
    int N=1+rand()%12;
    std::vector<BA> v(N); memset(v.data(),0,N*sizeof(BA));
    int p=rand()%200;
    for(int i=0;i<N;i++){ p+=1+rand()%400; v[i].pos=p; v[i].iv=491.8f; v[i].bar=(i%4)+1; v[i].x=-1; }
    int target=rand()%(p+400);
    int tol=rand()%300;
    struct Vec { BA* b; BA* e; BA* c; } vec={v.data(), v.data()+N, v.data()+N};
    int out=-999; int r=f(NULL,&vec,target,tol,&out);
    printf("%d %d %d",N,target,tol);
    for(int i=0;i<N;i++) printf(" %d",v[i].pos);
    printf(" | %d %d\n", r, out);
  }
  return 0;
}
