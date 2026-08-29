import Foundation
import Metal

let kernelSource = """
#include <metal_stdlib>
using namespace metal;

kernel void roundtrip(device const uchar *input [[buffer(0)]],
                      device uchar *output [[buffer(1)]],
                      uint id [[thread_position_in_grid]]) {
    output[id] = input[id];
}

struct KeyParams {
    uint sx;
    uint sy;
    uint sz;
    uint n;
};

kernel void neighborhood_keys_3x3_at(device const uchar *image [[buffer(0)]],
                                     device const uint *coords [[buffer(1)]],
                                     device uint *keys [[buffer(2)]],
                                     constant KeyParams &params [[buffer(3)]],
                                     uint id [[thread_position_in_grid]]) {
    if (id >= params.n) {
        return;
    }
    uint x = coords[id * 3 + 0];
    uint y = coords[id * 3 + 1];
    uint z = coords[id * 3 + 2];
    uint key = 0;
    uint bit = 0;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint nx = uint(int(x) + dx);
                uint ny = uint(int(y) + dy);
                uint nz = uint(int(z) + dz);
                uint offset = (nx * params.sy + ny) * params.sz + nz;
                if (image[offset] != 0) {
                    key |= (1u << bit);
                }
                bit += 1;
            }
        }
    }
    keys[id] = key;
}

struct VolumeParams {
    uint sx;
    uint sy;
    uint sz;
    uint psx;
    uint psy;
    uint psz;
    uint total;
};

struct ThinParams {
    uint pointType;
    uint px;
    uint py;
    uint pz;
};

inline uint idx3(uint x, uint y, uint z, uint sy, uint sz) {
    return (x * sy + y) * sz + z;
}

inline uchar getv(device const uchar *data, constant VolumeParams &p, int x, int y, int z) {
    if (x < 0 || y < 0 || z < 0 || x >= int(p.psx) || y >= int(p.psy) || z >= int(p.psz)) {
        return 0;
    }
    return data[idx3(uint(x), uint(y), uint(z), p.psy, p.psz)] != 0;
}

inline void c5Offset(uint oneBased, thread int &dx, thread int &dy, thread int &dz) {
    uint i = oneBased - 1;
    dx = int(i % 5) - 2;
    dy = int((i / 5) % 5) - 2;
    dz = int(i / 25) - 2;
}

inline uchar c5(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z, uint oneBased) {
    int dx, dy, dz;
    c5Offset(oneBased, dx, dy, dz);
    return getv(data, p, int(x) + dx, int(y) + dy, int(z) + dz);
}

inline uchar borderPointType(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uint sopen[6] = {38, 58, 62, 64, 68, 88};
    for (uint i = 0; i < 6; i++) {
        if (!c5(data, p, x, y, z, sopen[i])) return 1;
    }
    if (
        (!c5(data,p,x,y,z,33) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,53)) ||
        (!c5(data,p,x,y,z,37) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,39) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,43) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,73)) ||
        (!c5(data,p,x,y,z,57) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,59) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,67) && c5(data,p,x,y,z,61) && c5(data,p,x,y,z,73)) ||
        (!c5(data,p,x,y,z,69) && c5(data,p,x,y,z,73) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,83) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,53)) ||
        (!c5(data,p,x,y,z,87) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,89) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,93) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,73))
    ) return 2;
    if (
        (!c5(data,p,x,y,z,32) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,34) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,42) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,73) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,44) && c5(data,p,x,y,z,13) && c5(data,p,x,y,z,73) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,82) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,84) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,53) && c5(data,p,x,y,z,65)) ||
        (!c5(data,p,x,y,z,92) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,73) && c5(data,p,x,y,z,61)) ||
        (!c5(data,p,x,y,z,94) && c5(data,p,x,y,z,113) && c5(data,p,x,y,z,73) && c5(data,p,x,y,z,65))
    ) return 3;
    return 0;
}

inline uchar endpoint(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uint count = 0;
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
                count += getv(data, p, int(x)+dx, int(y)+dy, int(z)+dz) ? 1 : 0;
    return count <= 2;
}

inline uchar cfg5(thread uchar *cfg, uint oneBased) {
    uint i = oneBased - 1;
    uint x = i % 5;
    uint y = (i / 5) % 5;
    uint z = i / 25;
    return cfg[(x * 5 + y) * 5 + z] != 0;
}

inline void rotate2(thread uchar *in, thread uchar *out) {
    for (uint x=0; x<5; x++) for (uint y=0; y<5; y++) for (uint z=0; z<5; z++)
        out[((4 - z) * 5 + y) * 5 + x] = in[(x * 5 + y) * 5 + z];
}

inline void rotate3(thread uchar *in, thread uchar *out) {
    for (uint x=0; x<5; x++) for (uint y=0; y<5; y++) for (uint z=0; z<5; z++)
        out[((4 - y) * 5 + x) * 5 + z] = in[(x * 5 + y) * 5 + z];
}

inline uchar encircles4(thread uchar *cfg) {
    int sets[8][12] = {
        {1,2,3,4,5,8,9,12,13,14,15,16},
        {1,2,3,5,7,8,9,12,13,14,15,16},
        {1,2,3,5,7,8,9,10,12,14,15,16},
        {5,6,7,8,9,12,13,14,15,16,0,0},
        {1,2,3,5,7,8,9,12,13,14,15,16},
        {2,3,4,5,6,8,9,12,13,14,15,16},
        {6,7,8,10,12,14,15,16,0,0,0,0},
        {2,3,4,6,8,10,12,14,15,16,0,0}
    };
    for (uint s=0; s<8; s++) {
        uchar anyValue = 0;
        for (uint j=0; j<12; j++) {
            int idx = sets[s][j];
            if (idx == 0) continue;
            int zero = idx - 1;
            int y = zero % 4;
            int z = zero / 4;
            if (cfg[(2 * 5 + uint(y)) * 5 + uint(z)]) {
                anyValue = 1;
                break;
            }
        }
        if (!anyValue) return 1;
    }
    return 0;
}

inline uchar c12(thread uchar *cfg) {
    uint nsum = 0, n0 = 0, n2 = 0;
    for (uint y=1; y<=3; y++) for (uint z=1; z<=3; z++) {
        uchar anyx = 0;
        for (uint x=1; x<=3; x++) {
            uchar v = cfg[(x * 5 + y) * 5 + z] != 0;
            anyx |= v;
            if (x == 1 && v) n0++;
            if (x == 3 && v) n2++;
        }
        if (anyx) nsum++;
    }
    uchar cond1 = encircles4(cfg) && n0 > 0 && n2 > 0;
    uchar cond2 = (!cfg5(cfg,62)) && ((!cfg5(cfg,64)) || (!cfg5(cfg,65))) && nsum == 9;
    return cond1 || cond2;
}

inline uchar shapePreserving(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uchar cfg[125], r2[125], r3[125];
    uint i=0;
    for (int dx=-2; dx<=2; dx++) for (int dy=-2; dy<=2; dy++) for (int dz=-2; dz<=2; dz++)
        cfg[i++] = getv(data,p,int(x)+dx,int(y)+dy,int(z)+dz);
    rotate2(cfg,r2); rotate3(cfg,r3);
    return c12(cfg) || c12(r2) || c12(r3);
}

inline uint count2d8(thread uchar *active) {
    uchar labels[9] = {0,0,0,0,0,0,0,0,0};
    uint component = 0;
    uint queue[9];
    for (uint start=0; start<9; start++) {
        if (!active[start] || labels[start]) continue;
        component++;
        uint qh=0, qt=0; queue[qt++]=start; labels[start]=uchar(component);
        while (qh < qt) {
            uint item = queue[qh++]; int ix=int(item%3), iy=int(item/3);
            for (uint other=0; other<9; other++) {
                if (!active[other] || labels[other]) continue;
                int ox=int(other%3), oy=int(other/3);
                int cheb = max(abs(ix-ox), abs(iy-oy));
                if (cheb == 1) { labels[other]=uchar(component); queue[qt++]=other; }
            }
        }
    }
    return component;
}

inline uchar c3one(thread uchar *cfg) {
    uchar allE = cfg5(cfg,33) && cfg5(cfg,43) && cfg5(cfg,83) && cfg5(cfg,93);
    uchar plane[9]; uint i=0;
    for (uint y=1; y<=3; y++) for (uint z=1; z<=3; z++) plane[i++] = cfg[(2*5+y)*5+z];
    plane[4] = 0;
    uchar single = count2d8(plane) == 1;
    uchar noTunnel = !(cfg5(cfg,38) && cfg5(cfg,58) && cfg5(cfg,68) && cfg5(cfg,88));
    return (!allE) || (single && noTunnel);
}

inline uchar tunnelPreservingE(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uchar cfg[125], r2[125], r3[125]; uint i=0;
    for (int dx=-2; dx<=2; dx++) for (int dy=-2; dy<=2; dy++) for (int dz=-2; dz<=2; dz++)
        cfg[i++] = getv(data,p,int(x)+dx,int(y)+dy,int(z)+dz);
    rotate2(cfg,r2); rotate3(cfg,r3);
    return c3one(cfg) && c3one(r2) && c3one(r3);
}

inline void c456one(thread uchar *cfg, thread uchar &cond, thread uchar &mbep, thread uchar &mcfp) {
    uchar bp[9], cp[9]; uint i=0;
    for (uint x=1; x<=3; x++) for (uint z=1; z<=3; z++) bp[i++] = cfg[(x*5+2)*5+z];
    i=0;
    for (uint x=1; x<=3; x++) for (uint y=1; y<=3; y++) cp[i++] = cfg[(x*5+y)*5+2];
    bp[4] = 0; cp[4] = 0;
    mbep = (count2d8(bp) == 1) && ((uint)cfg5(cfg,38)+(uint)cfg5(cfg,62)+(uint)cfg5(cfg,64)+(uint)cfg5(cfg,88) != 4);
    mcfp = (count2d8(cp) == 1) && ((uint)cfg5(cfg,58)+(uint)cfg5(cfg,62)+(uint)cfg5(cfg,64)+(uint)cfg5(cfg,68) != 4);
    cond = (!cfg5(cfg,62)) && (!cfg5(cfg,65)) && cfg5(cfg,58) && cfg5(cfg,38) && cfg5(cfg,64);
}

inline uchar finalErosion(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uchar cfg[125], r2[125], r3[125]; uint i=0;
    for (int dx=-2; dx<=2; dx++) for (int dy=-2; dy<=2; dy++) for (int dz=-2; dz<=2; dz++)
        cfg[i++] = getv(data,p,int(x)+dx,int(y)+dy,int(z)+dz);
    rotate2(cfg,r2); rotate3(cfg,r3);
    uchar cond, mbep, mcfp; uint condCount=0, mbepCount=0, mcfpCount=0;
    c456one(cfg,cond,mbep,mcfp); if (cond) { condCount++; mbepCount += mbep; mcfpCount += mcfp; }
    c456one(r2,cond,mbep,mcfp); if (cond) { condCount++; mbepCount += mbep; mcfpCount += mcfp; }
    c456one(r3,cond,mbep,mcfp); if (cond) { condCount++; mbepCount += mbep; mcfpCount += mcfp; }
    if (condCount == 1) return mcfpCount > 0 && mbepCount > 0;
    if (condCount == 2) return mcfpCount > 1 || mbepCount > 1;
    return condCount == 3;
}

inline uint count3d(thread uchar *active, bool use26, thread uchar *labelsOut) {
    uchar labels[27];
    for (uint i=0; i<27; i++) labels[i]=0;
    uint component=0, queue[27];
    for (uint start=0; start<27; start++) {
        if (!active[start] || labels[start]) continue;
        component++; uint qh=0, qt=0; queue[qt++]=start; labels[start]=uchar(component);
        while (qh<qt) {
            uint item=queue[qh++]; int ix=int(item%3), iy=int((item/3)%3), iz=int(item/9);
            for (uint other=0; other<27; other++) {
                if (!active[other] || labels[other]) continue;
                int ox=int(other%3), oy=int((other/3)%3), oz=int(other/9);
                int dist=abs(ix-ox)+abs(iy-oy)+abs(iz-oz);
                int cheb=max(max(abs(ix-ox),abs(iy-oy)),abs(iz-oz));
                bool connected = use26 ? (cheb == 1) : (dist == 1);
                if (connected) { labels[other]=uchar(component); queue[qt++]=other; }
            }
        }
    }
    if (labelsOut != nullptr) for (uint i=0; i<27; i++) labelsOut[i]=labels[i];
    return component;
}

inline uchar classifySpoints(thread uchar *cfg) {
    uint sp[6] = {4,10,12,16,14,22};
    uint num=0; for (uint i=0; i<6; i++) num += cfg[sp[i]] ? 1 : 0;
    uint opposite=0;
    if (cfg[sp[0]] && cfg[sp[5]]) opposite += 2;
    if (cfg[sp[2]] && cfg[sp[4]]) opposite += 2;
    if (cfg[sp[1]] && cfg[sp[3]]) opposite += 2;
    uint adjacent = num - opposite;
    if (num == 6) return 0;
    if (num == 5) return 1;
    if (opposite == 4) return 2;
    if (opposite == 2 && adjacent == 2) return 3;
    if (opposite == 2 && adjacent == 1) return 4;
    if (adjacent == 3) return 5;
    if (opposite == 2) return 6;
    if (adjacent == 2) return 7;
    if (adjacent == 1) return 8;
    return 9;
}

inline uchar simplePoint(device const uchar *data, constant VolumeParams &p, uint x, uint y, uint z) {
    uchar cfg[27]; uint i=0;
    for (int dz=-1; dz<=1; dz++) for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++)
        cfg[i++] = getv(data,p,int(x)+dx,int(y)+dy,int(z)+dz);
    if (classifySpoints(cfg) == 0) return 0;
    uchar black[27]; for (uint k=0; k<27; k++) black[k]=cfg[k]; black[13]=0;
    uint epsilon = count3d(black, true, nullptr);
    uchar bg[27]; for (uint k=0; k<27; k++) bg[k]=!cfg[k]; bg[13]=0;
    uint corners[8] = {0,6,2,8,18,20,24,26};
    for (uint k=0; k<8; k++) bg[corners[k]]=0;
    uchar labels[27]; count3d(bg, false, labels);
    uint sp[6] = {4,10,12,16,14,22};
    uchar seen[28]; for (uint k=0; k<28; k++) seen[k]=0;
    uint intersecting=0;
    for (uint k=0; k<6; k++) { uchar label=labels[sp[k]]; if (label && !seen[label]) { seen[label]=1; intersecting++; } }
    int mu = int(intersecting) - 1;
    return epsilon == 1 && mu == 0;
}

kernel void pad_input(device const uchar *input [[buffer(0)]],
                      device uchar *current [[buffer(1)]],
                      constant VolumeParams &p [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    if (id >= p.sx * p.sy * p.sz) return;
    uint x = id / (p.sy * p.sz);
    uint rem = id - x * p.sy * p.sz;
    uint y = rem / p.sz;
    uint z = rem - y * p.sz;
    current[idx3(x+2, y+2, z+2, p.psy, p.psz)] = input[id] != 0;
}

kernel void copy_buffer(device const uchar *src [[buffer(0)]],
                        device uchar *dst [[buffer(1)]],
                        constant VolumeParams &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id < p.total) dst[id] = src[id];
}

kernel void clear_u8(device uchar *dst [[buffer(0)]],
                     constant VolumeParams &p [[buffer(1)]],
                     uint id [[thread_position_in_grid]]) {
    if (id < p.total) dst[id] = 0;
}

kernel void clear_counter(device atomic_uint *counter [[buffer(0)]]) {
    atomic_store_explicit(counter, 0, memory_order_relaxed);
}

kernel void classify_candidates(device const uchar *start [[buffer(0)]],
                                device const uchar *protectedVoxels [[buffer(1)]],
                                device uchar *pointTypes [[buffer(2)]],
                                device atomic_uint *counter [[buffer(3)]],
                                constant VolumeParams &p [[buffer(4)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= p.total) return;
    uint x = id / (p.psy * p.psz);
    uint rem = id - x * p.psy * p.psz;
    uint y = rem / p.psz;
    uint z = rem - y * p.psz;
    if (x < 2 || y < 2 || z < 2 || x >= p.psx-2 || y >= p.psy-2 || z >= p.psz-2) return;
    if (!start[id] || protectedVoxels[id]) return;
    uint n=0;
    for (int dx=-1; dx<=1; dx++) for (int dy=-1; dy<=1; dy++) for (int dz=-1; dz<=1; dz++)
        n += getv(start,p,int(x)+dx,int(y)+dy,int(z)+dz) ? 1 : 0;
    if (n >= 27) return;
    uchar t = borderPointType(start,p,x,y,z);
    if (t) { pointTypes[id] = t; atomic_fetch_add_explicit(counter, 1, memory_order_relaxed); }
}

kernel void thin_subfield(device uchar *current [[buffer(0)]],
                          device const uchar *start [[buffer(1)]],
                          device uchar *protectedVoxels [[buffer(2)]],
                          device const uchar *pointTypes [[buffer(3)]],
                          device atomic_uint *removed [[buffer(4)]],
                          constant VolumeParams &p [[buffer(5)]],
                          constant ThinParams &thin [[buffer(6)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= p.total) return;
    uint x = id / (p.psy * p.psz);
    uint rem = id - x * p.psy * p.psz;
    uint y = rem / p.psz;
    uint z = rem - y * p.psz;
    if (x < 2 || y < 2 || z < 2 || x >= p.psx-2 || y >= p.psy-2 || z >= p.psz-2) return;
    if ((x & 1) != thin.px || (y & 1) != thin.py || (z & 1) != thin.pz) return;
    if (pointTypes[id] != thin.pointType || !current[id] || protectedVoxels[id]) return;
    if (endpoint(current,p,x,y,z)) { protectedVoxels[id] = 1; return; }
    if (shapePreserving(start,p,x,y,z)) { protectedVoxels[id] = 1; return; }
    if (thin.pointType == 2) {
        if (tunnelPreservingE(start,p,x,y,z)) { current[id] = 0; atomic_fetch_add_explicit(removed, 1, memory_order_relaxed); }
    } else if (simplePoint(current,p,x,y,z)) {
        current[id] = 0; atomic_fetch_add_explicit(removed, 1, memory_order_relaxed);
    }
}

kernel void mark_final_erosion(device const uchar *current [[buffer(0)]],
                               device uchar *deleteMask [[buffer(1)]],
                               device atomic_uint *counter [[buffer(2)]],
                               constant VolumeParams &p [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= p.total) return;
    uint x = id / (p.psy * p.psz);
    uint rem = id - x * p.psy * p.psz;
    uint y = rem / p.psz;
    uint z = rem - y * p.psz;
    if (x < 2 || y < 2 || z < 2 || x >= p.psx-2 || y >= p.psy-2 || z >= p.psz-2) return;
    if (current[id] && finalErosion(current,p,x,y,z) && simplePoint(current,p,x,y,z)) {
        deleteMask[id] = 1;
        atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
    }
}

kernel void apply_delete(device uchar *current [[buffer(0)]],
                         device const uchar *deleteMask [[buffer(1)]],
                         constant VolumeParams &p [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    if (id < p.total && deleteMask[id]) current[id] = 0;
}

kernel void crop_output(device const uchar *current [[buffer(0)]],
                        device uchar *output [[buffer(1)]],
                        constant VolumeParams &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= p.sx * p.sy * p.sz) return;
    uint x = id / (p.sy * p.sz);
    uint rem = id - x * p.sy * p.sz;
    uint y = rem / p.sz;
    uint z = rem - y * p.sz;
    output[id] = current[idx3(x+2, y+2, z+2, p.psy, p.psz)] != 0;
}
"""

func json(_ values: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: values, options: [.sortedKeys])
    print(String(data: data, encoding: .utf8)!)
}

struct KeyParams {
    var sx: UInt32
    var sy: UInt32
    var sz: UInt32
    var n: UInt32
}

struct VolumeParams {
    var sx: UInt32
    var sy: UInt32
    var sz: UInt32
    var psx: UInt32
    var psy: UInt32
    var psz: UInt32
    var total: UInt32
}

struct ThinParams {
    var pointType: UInt32
    var px: UInt32
    var py: UInt32
    var pz: UInt32
}

func fail(_ message: String, _ code: Int32 = 1) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(code)
}

func makeDeviceAndLibrary() -> (MTLDevice, MTLLibrary) {
    guard let device = MTLCreateSystemDefaultDevice() else {
        fail("No default Metal device is available.")
    }
    do {
        let library = try device.makeLibrary(source: kernelSource, options: nil)
        return (device, library)
    } catch {
        fail("Metal runtime shader compilation failed: \(error)")
    }
}

func runProbe() {
    guard let device = MTLCreateSystemDefaultDevice() else {
        json([
            "available": false,
            "device": NSNull(),
            "reason": "No default Metal device is available."
        ])
        exit(0)
    }
    do {
        _ = try device.makeLibrary(source: kernelSource, options: nil)
        json([
            "available": true,
            "device": device.name,
            "reason": "Metal device and runtime shader compiler are available."
        ])
    } catch {
        json([
            "available": false,
            "device": device.name,
            "reason": "Metal runtime shader compilation failed: \(error)"
        ])
    }
}

func runKeys(_ arguments: [String]) {
    if arguments.count != 9 {
        fail("usage: PlateRodMetal --keys image.u8 coords.u32 keys.u32 sx sy sz n")
    }
    let imageURL = URL(fileURLWithPath: arguments[2])
    let coordsURL = URL(fileURLWithPath: arguments[3])
    let outputURL = URL(fileURLWithPath: arguments[4])
    guard
        let sx = UInt32(arguments[5]),
        let sy = UInt32(arguments[6]),
        let sz = UInt32(arguments[7]),
        let n = UInt32(arguments[8])
    else {
        fail("invalid dimensions")
    }
    let imageData: Data
    let coordsData: Data
    do {
        imageData = try Data(contentsOf: imageURL)
        coordsData = try Data(contentsOf: coordsURL)
    } catch {
        fail("failed to read input: \(error)")
    }
    let expectedImageBytes = Int(sx) * Int(sy) * Int(sz)
    let expectedCoordsBytes = Int(n) * 3 * MemoryLayout<UInt32>.stride
    if imageData.count != expectedImageBytes {
        fail("image byte count mismatch")
    }
    if coordsData.count != expectedCoordsBytes {
        fail("coords byte count mismatch")
    }

    let (device, library) = makeDeviceAndLibrary()
    guard let function = library.makeFunction(name: "neighborhood_keys_3x3_at") else {
        fail("missing neighborhood_keys_3x3_at kernel")
    }
    let pipeline: MTLComputePipelineState
    do {
        pipeline = try device.makeComputePipelineState(function: function)
    } catch {
        fail("failed to build compute pipeline: \(error)")
    }
    guard let queue = device.makeCommandQueue() else {
        fail("failed to create Metal command queue")
    }
    guard
        let imageBuffer = device.makeBuffer(bytes: Array(imageData), length: imageData.count, options: .storageModeShared),
        let coordsBuffer = device.makeBuffer(bytes: Array(coordsData), length: coordsData.count, options: .storageModeShared),
        let keysBuffer = device.makeBuffer(length: Int(n) * MemoryLayout<UInt32>.stride, options: .storageModeShared),
        let commandBuffer = queue.makeCommandBuffer(),
        let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
        fail("failed to allocate Metal resources")
    }

    var params = KeyParams(sx: sx, sy: sy, sz: sz, n: n)
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(imageBuffer, offset: 0, index: 0)
    encoder.setBuffer(coordsBuffer, offset: 0, index: 1)
    encoder.setBuffer(keysBuffer, offset: 0, index: 2)
    encoder.setBytes(&params, length: MemoryLayout<KeyParams>.stride, index: 3)
    let width = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256))
    encoder.dispatchThreads(
        MTLSize(width: Int(n), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        fail("Metal command failed: \(error)")
    }

    let output = Data(bytes: keysBuffer.contents(), count: Int(n) * MemoryLayout<UInt32>.stride)
    do {
        try output.write(to: outputURL)
    } catch {
        fail("failed to write output: \(error)")
    }
}

func pipeline(_ library: MTLLibrary, _ device: MTLDevice, _ name: String) -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
        fail("missing \(name) kernel")
    }
    do {
        return try device.makeComputePipelineState(function: function)
    } catch {
        fail("failed to build \(name) pipeline: \(error)")
    }
}

func runKernel(
    _ pipeline: MTLComputePipelineState,
    _ queue: MTLCommandQueue,
    _ total: Int,
    _ bind: (MTLComputeCommandEncoder) -> Void
) {
    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
        fail("failed to allocate Metal command encoder")
    }
    encoder.setComputePipelineState(pipeline)
    bind(encoder)
    let width = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256))
    encoder.dispatchThreads(
        MTLSize(width: max(total, 1), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    if let error = commandBuffer.error {
        fail("Metal command failed: \(error)")
    }
}

func counterValue(_ buffer: MTLBuffer) -> UInt32 {
    return buffer.contents().assumingMemoryBound(to: UInt32.self).pointee
}

func runSkeletonize(_ arguments: [String]) {
    if arguments.count != 8 {
        fail("usage: PlateRodMetal --skeletonize image.u8 output.u8 sx sy sz max_iterations")
    }
    let inputURL = URL(fileURLWithPath: arguments[2])
    let outputURL = URL(fileURLWithPath: arguments[3])
    guard
        let sx = UInt32(arguments[4]),
        let sy = UInt32(arguments[5]),
        let sz = UInt32(arguments[6]),
        let maxIterations = Int(arguments[7])
    else {
        fail("invalid dimensions")
    }
    let inputData: Data
    do {
        inputData = try Data(contentsOf: inputURL)
    } catch {
        fail("failed to read input: \(error)")
    }
    let voxelCount = Int(sx) * Int(sy) * Int(sz)
    if inputData.count != voxelCount {
        fail("image byte count mismatch")
    }

    let (device, library) = makeDeviceAndLibrary()
    guard let queue = device.makeCommandQueue() else {
        fail("failed to create Metal command queue")
    }
    let psx = sx + 4
    let psy = sy + 4
    let psz = sz + 4
    let paddedTotal = Int(psx) * Int(psy) * Int(psz)
    var params = VolumeParams(sx: sx, sy: sy, sz: sz, psx: psx, psy: psy, psz: psz, total: UInt32(paddedTotal))

    guard
        let inputBuffer = device.makeBuffer(bytes: Array(inputData), length: inputData.count, options: .storageModeShared),
        let outputBuffer = device.makeBuffer(length: voxelCount, options: .storageModeShared),
        let currentBuffer = device.makeBuffer(length: paddedTotal, options: .storageModeShared),
        let startBuffer = device.makeBuffer(length: paddedTotal, options: .storageModeShared),
        let protectedBuffer = device.makeBuffer(length: paddedTotal, options: .storageModeShared),
        let pointTypesBuffer = device.makeBuffer(length: paddedTotal, options: .storageModeShared),
        let counterBuffer = device.makeBuffer(length: MemoryLayout<UInt32>.stride, options: .storageModeShared)
    else {
        fail("failed to allocate Metal buffers")
    }

    let clear = pipeline(library, device, "clear_u8")
    let clearCounter = pipeline(library, device, "clear_counter")
    let pad = pipeline(library, device, "pad_input")
    let copy = pipeline(library, device, "copy_buffer")
    let classify = pipeline(library, device, "classify_candidates")
    let thin = pipeline(library, device, "thin_subfield")
    let markFinal = pipeline(library, device, "mark_final_erosion")
    let applyDelete = pipeline(library, device, "apply_delete")
    let crop = pipeline(library, device, "crop_output")

    runKernel(clear, queue, paddedTotal) {
        $0.setBuffer(currentBuffer, offset: 0, index: 0)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 1)
    }
    runKernel(clear, queue, paddedTotal) {
        $0.setBuffer(protectedBuffer, offset: 0, index: 0)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 1)
    }
    runKernel(pad, queue, voxelCount) {
        $0.setBuffer(inputBuffer, offset: 0, index: 0)
        $0.setBuffer(currentBuffer, offset: 0, index: 1)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 2)
    }

    if maxIterations > 0 {
        for _ in 0..<maxIterations {
            runKernel(copy, queue, paddedTotal) {
                $0.setBuffer(currentBuffer, offset: 0, index: 0)
                $0.setBuffer(startBuffer, offset: 0, index: 1)
                $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 2)
            }
            runKernel(clear, queue, paddedTotal) {
                $0.setBuffer(pointTypesBuffer, offset: 0, index: 0)
                $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 1)
            }
            runKernel(clearCounter, queue, 1) {
                $0.setBuffer(counterBuffer, offset: 0, index: 0)
            }
            runKernel(classify, queue, paddedTotal) {
                $0.setBuffer(startBuffer, offset: 0, index: 0)
                $0.setBuffer(protectedBuffer, offset: 0, index: 1)
                $0.setBuffer(pointTypesBuffer, offset: 0, index: 2)
                $0.setBuffer(counterBuffer, offset: 0, index: 3)
                $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 4)
            }
            if counterValue(counterBuffer) == 0 {
                break
            }

            runKernel(clearCounter, queue, 1) {
                $0.setBuffer(counterBuffer, offset: 0, index: 0)
            }
            for pointType in UInt32(1)...UInt32(3) {
                for px in UInt32(0)...UInt32(1) {
                    for py in UInt32(0)...UInt32(1) {
                        for pz in UInt32(0)...UInt32(1) {
                            var thinParams = ThinParams(pointType: pointType, px: px, py: py, pz: pz)
                            runKernel(thin, queue, paddedTotal) {
                                $0.setBuffer(currentBuffer, offset: 0, index: 0)
                                $0.setBuffer(startBuffer, offset: 0, index: 1)
                                $0.setBuffer(protectedBuffer, offset: 0, index: 2)
                                $0.setBuffer(pointTypesBuffer, offset: 0, index: 3)
                                $0.setBuffer(counterBuffer, offset: 0, index: 4)
                                $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 5)
                                $0.setBytes(&thinParams, length: MemoryLayout<ThinParams>.stride, index: 6)
                            }
                        }
                    }
                }
            }
            if counterValue(counterBuffer) == 0 {
                break
            }
        }
    }

    runKernel(clear, queue, paddedTotal) {
        $0.setBuffer(pointTypesBuffer, offset: 0, index: 0)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 1)
    }
    runKernel(clearCounter, queue, 1) {
        $0.setBuffer(counterBuffer, offset: 0, index: 0)
    }
    runKernel(markFinal, queue, paddedTotal) {
        $0.setBuffer(currentBuffer, offset: 0, index: 0)
        $0.setBuffer(pointTypesBuffer, offset: 0, index: 1)
        $0.setBuffer(counterBuffer, offset: 0, index: 2)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 3)
    }
    if counterValue(counterBuffer) > 0 {
        runKernel(applyDelete, queue, paddedTotal) {
            $0.setBuffer(currentBuffer, offset: 0, index: 0)
            $0.setBuffer(pointTypesBuffer, offset: 0, index: 1)
            $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 2)
        }
    }
    runKernel(crop, queue, voxelCount) {
        $0.setBuffer(currentBuffer, offset: 0, index: 0)
        $0.setBuffer(outputBuffer, offset: 0, index: 1)
        $0.setBytes(&params, length: MemoryLayout<VolumeParams>.stride, index: 2)
    }

    let output = Data(bytes: outputBuffer.contents(), count: voxelCount)
    do {
        try output.write(to: outputURL)
    } catch {
        fail("failed to write output: \(error)")
    }
}

if CommandLine.arguments.contains("--probe") {
    runProbe()
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--keys" {
    runKeys(CommandLine.arguments)
} else if CommandLine.arguments.count > 1 && CommandLine.arguments[1] == "--skeletonize" {
    runSkeletonize(CommandLine.arguments)
} else {
    json([
        "available": false,
        "device": NSNull(),
        "reason": "Run with --probe."
    ])
}
