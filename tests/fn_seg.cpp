#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mach-o/dyld.h>
struct BA { int pos; int pad; float iv; int bar; int x; int flag; };
typedef void (*Fn)(void*, void*, void*, int);
struct V { BA* b; BA* e; BA* c; };
int main(int argc,char**argv){
  void* h=dlopen("/tmp/rb2.dylib",RTLD_NOW|RTLD_LOCAL);
  if(!h){fprintf(stderr,"dlopen %s\n",dlerror());return 1;}
  uint64_t base=0;
  for(uint32_t i=0;i<_dyld_image_count();i++){const char*nm=_dyld_get_image_name(i);
    if(nm&&strstr(nm,"rb2.dylib")){base=(uint64_t)_dyld_get_image_header(i);break;}}
  Fn f=(Fn)(base+(0x100f6a008ULL-0x100000000ULL));
  int sr=argc>1?atoi(argv[1]):44100;
  int N=6;
  std::vector<BA> v(N), cand(N);
  memset(v.data(),0,N*sizeof(BA)); memset(cand.data(),0,N*sizeof(BA));
  for(int i=0;i<N;i++){
    v[i].pos=536+i*21688; v[i].iv=491.791382f; v[i].bar=(i%4)+1; v[i].x=-1;
    cand[i].pos=536+i*21688; cand[i].iv=491.791382f; cand[i].bar=(i%4)+1; cand[i].x=-1;
  }
  V vec={v.data(), v.data()+N, v.data()+N};
  V cvec={cand.data(), cand.data()+N, cand.data()+N};
  char self[0x40]; memset(self,0,sizeof(self));
  *(int*)(self+0)=1024; *(int*)(self+4)=sr;
  char info[0x40]; memset(info,0,sizeof(info));
  *(void**)(info+8)=&cvec;                // info+8 = POINTER to the candidate vector
  *(int*)(info+0x14)=sr;                  // caller: info+0x14 = [Analyzer+0x14]
  *(int*)(info+0x18)=4;                   // caller: info+0x18 = 4
  f(self, info, &vec, 21689344);
  int cnt=(int)(vec.e-vec.b);
  printf("sr=%d count=%d\n",sr,cnt);
  for(int i=0;i<cnt && i<8;i++)
    printf("  %d: pos=%d iv=%.6f bar=%d x=%d flag=%d\n",i,v[i].pos,v[i].iv,v[i].bar,v[i].x,v[i].flag);
  return 0;
}
