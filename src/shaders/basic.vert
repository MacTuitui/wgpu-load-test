// shader.vert
#version 450
//input
layout(location=0) in vec3 a_position;
layout(location=1) in vec2 a_tex_coords;
layout(location=2) in vec3 a_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 world;
    mat4 view;
    mat4 proj;
    vec4 light_pos;
    vec4 colors[30];
    float time;
    float costime;
    float sintime;
} uniforms;

//output
layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_tex_coords;
layout(location = 2) out vec4 v_color;
layout(location = 3) out vec4 v_pos;
layout(location = 4) out vec4 v_light_pos;

#define TAU 6.283185307179586
//	Simplex 4D Noise
//	by Ian McEwan, Ashima Arts
//
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
float permute(float x){return floor(mod(((x*34.0)+1.0)*x, 289.0));}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}
float taylorInvSqrt(float r){return 1.79284291400159 - 0.85373472095314 * r;}

vec4 grad4(float j, vec4 ip){
  const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
  vec4 p,s;

  p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
  p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
  s = vec4(lessThan(p, vec4(0.0)));
  p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;

  return p;
}

float snoise(vec4 v){
  const vec2  C = vec2( 0.138196601125010504,  // (5 - sqrt(5))/20  G4
                        0.309016994374947451); // (sqrt(5) - 1)/4   F4
// First corner
  vec4 i  = floor(v + dot(v, C.yyyy) );
  vec4 x0 = v -   i + dot(i, C.xxxx);

// Other corners

// Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
  vec4 i0;

  vec3 isX = step( x0.yzw, x0.xxx );
  vec3 isYZ = step( x0.zww, x0.yyz );
//  i0.x = dot( isX, vec3( 1.0 ) );
  i0.x = isX.x + isX.y + isX.z;
  i0.yzw = 1.0 - isX;

//  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
  i0.y += isYZ.x + isYZ.y;
  i0.zw += 1.0 - isYZ.xy;

  i0.z += isYZ.z;
  i0.w += 1.0 - isYZ.z;

  // i0 now contains the unique values 0,1,2,3 in each channel
  vec4 i3 = clamp( i0, 0.0, 1.0 );
  vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
  vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );

  //  x0 = x0 - 0.0 + 0.0 * C
  vec4 x1 = x0 - i1 + 1.0 * C.xxxx;
  vec4 x2 = x0 - i2 + 2.0 * C.xxxx;
  vec4 x3 = x0 - i3 + 3.0 * C.xxxx;
  vec4 x4 = x0 - 1.0 + 4.0 * C.xxxx;

// Permutations
  i = mod(i, 289.0);
  float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
  vec4 j1 = permute( permute( permute( permute (
             i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
           + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
           + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
           + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));
// Gradients
// ( 7*7*6 points uniformly over a cube, mapped onto a 4-octahedron.)
// 7*7*6 = 294, which is close to the ring size 17*17 = 289.

  vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

  vec4 p0 = grad4(j0,   ip);
  vec4 p1 = grad4(j1.x, ip);
  vec4 p2 = grad4(j1.y, ip);
  vec4 p3 = grad4(j1.z, ip);
  vec4 p4 = grad4(j1.w, ip);

// Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;
  p4 *= taylorInvSqrt(dot(p4,p4));

// Mix contributions from the five corners
  vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
  vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
  m0 = m0 * m0;
  m1 = m1 * m1;
  return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
               + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;

}


float random (in vec2 _st) {
    return fract(sin(dot(_st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}
float smoothfloor(float x)
{
    float F = floor(x),
          f = clamp( 1.-(1.-fract(x))/0.2, 0.,1.);
       return F + smoothstep(0.,1., f) ;                    // C1   NB: 3.x^2 - 2.x^3
}

float random11(float n){return fract(sin(n) * 42502.9504);}

float noise11(float p){
    float fl = floor(p);
    float fc = fract(p);
    return mix(random11(fl), random11(fl + 1.0), fc);
}


vec2 bigf(vec3 p) {

    //float n_s = n(p, 0.1,1.0)*0.5+0.5;
    //float n_s = snoise(vec4(p*0.1,1.0))*0.5+0.5;
    float n_o = snoise(vec4(p,uniforms.costime*0.1))*0.5+0.5;
    float stripe_noise = floor(n_o*10.0)/10.0;
    stripe_noise = random11(stripe_noise)*0.4;
    float n_o2 = snoise(vec4(p* 0.3,uniforms.sintime))*0.5+0.5;
    float stripe_noise2 = pow(n_o2,2.0)*0.5;//floor(n_o2*3.0)/3.0;
    float d = (snoise(vec4(p*10.0,0.0))*0.5+0.5)*0.03;
    return vec2(d+stripe_noise+stripe_noise2, (stripe_noise)*5.0);
}

void main() {
    mat4 worldview = uniforms.view * uniforms.world;
    
    v_tex_coords = a_tex_coords;
    v_light_pos = uniforms.view * uniforms.world * uniforms.light_pos;

    vec2 pr = bigf(a_position);
    float orig_r = length(a_position);//should be 1
    v_pos = vec4(a_position/ orig_r,1.0);
    float phy = acos(v_pos.z);
    float theta = atan(v_pos.y, v_pos.x);
    vec3 pos_o = a_position*(0.8+0.2*pr.x);
    vec3 op = vec3(cos(theta)*sin(phy+0.002), sin(theta)*sin(phy+0.002), cos(phy+0.002))*orig_r;
    vec3 ot = vec3(cos(theta+0.002)*sin(phy), sin(theta+0.002)*sin(phy), cos(phy))*orig_r;
    vec3 pos_op = op*(0.8+0.2*bigf(op).x)-pos_o;
    vec3 pos_ot = ot*(0.8+0.2*bigf(ot).x)-pos_o;
    vec3 normal = normalize(cross(pos_op,pos_ot));

    v_normal = transpose(inverse(mat3(worldview))) * normal;

    v_color = uniforms.colors[int(pr.y*40.0)%30];
    v_color.a = pr.y;

    v_pos = worldview * vec4(pos_o,1.0);
    gl_Position = uniforms.proj*v_pos;
}
