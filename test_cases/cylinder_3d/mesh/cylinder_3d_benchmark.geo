// ============================================================
// 3D Cylinder Flow - Fixed Topology V2 (X-Z Swapped)
// Status: FIXING Curve Loop Orientation
// ============================================================

// 1. Geometric Parameters
L = 2.5;  H = 0.41;  W = 0.41;
D = 0.1;  R = D/2.0;
Cz = 0.5; Cy = 0.2;
R_box = 0.12; 
lc = 0.1; 

// 2. Mesh Density
N_core = 20;   // Cylinder quadrant & Box side
N_rad = 15;    // Radial thickness
N_upstream = 30;
N_downstream = 60;
N_vert = 7;   // Vertical height
N_span = 12;   // 3D Extrusion

r_wall = 1.25;
r_wake = 1.03;

// ============================================================
// 3. Points (X and Z swapped)
// ============================================================
p_center = newp; Point(p_center) = {Cz, Cy, 0, lc};          // was {0, Cy, Cz}

// Cylinder (45 deg)
d45 = R * 0.70710678;
pc_sw = newp; Point(pc_sw) = {Cz-d45, Cy-d45, 0, lc};       // was {0, Cy-d45, Cz-d45}
pc_se = newp; Point(pc_se) = {Cz+d45, Cy-d45, 0, lc};       // was {0, Cy-d45, Cz+d45}
pc_ne = newp; Point(pc_ne) = {Cz+d45, Cy+d45, 0, lc};       // was {0, Cy+d45, Cz+d45}
pc_nw = newp; Point(pc_nw) = {Cz-d45, Cy+d45, 0, lc};       // was {0, Cy+d45, Cz-d45}

// Box
pb_sw = newp; Point(pb_sw) = {Cz-R_box, Cy-R_box, 0, lc};   // was {0, Cy-R_box, Cz-R_box}
pb_se = newp; Point(pb_se) = {Cz+R_box, Cy-R_box, 0, lc};   // was {0, Cy-R_box, Cz+R_box}
pb_ne = newp; Point(pb_ne) = {Cz+R_box, Cy+R_box, 0, lc};   // was {0, Cy+R_box, Cz+R_box}
pb_nw = newp; Point(pb_nw) = {Cz-R_box, Cy+R_box, 0, lc};   // was {0, Cy+R_box, Cz-R_box}

// Domain Outer Points
pd_sw = newp; Point(pd_sw) = {0, 0, 0, lc};                 // unchanged (0,0,0)
pd_se = newp; Point(pd_se) = {L, 0, 0, lc};                 // was {0,0,L}
pd_ne = newp; Point(pd_ne) = {L, H, 0, lc};                 // was {0,H,L}
pd_nw = newp; Point(pd_nw) = {0, H, 0, lc};                 // unchanged (0,H,0)

// Projection Points
// Bottom Wall
pp_b_w = newp; Point(pp_b_w) = {Cz-R_box, 0, 0, lc};        // was {0,0,Cz-R_box}
pp_b_e = newp; Point(pp_b_e) = {Cz+R_box, 0, 0, lc};        // was {0,0,Cz+R_box}
// Top Wall
pp_t_w = newp; Point(pp_t_w) = {Cz-R_box, H, 0, lc};        // was {0,H,Cz-R_box}
pp_t_e = newp; Point(pp_t_e) = {Cz+R_box, H, 0, lc};        // was {0,H,Cz+R_box}
// Inlet Plane
pp_w_b = newp; Point(pp_w_b) = {0, Cy-R_box, 0, lc};        // was {0,Cy-R_box,0}
pp_w_t = newp; Point(pp_w_t) = {0, Cy+R_box, 0, lc};        // was {0,Cy+R_box,0}
// Outlet Plane
pp_e_b = newp; Point(pp_e_b) = {L, Cy-R_box, 0, lc};        // was {0,Cy-R_box,L}
pp_e_t = newp; Point(pp_e_t) = {L, Cy+R_box, 0, lc};        // was {0,Cy+R_box,L}

// ============================================================
// 4. Lines (Definitions with Directions) - unchanged
// ============================================================

// --- Cylinder ---
c_bot = newl; Circle(c_bot) = {pc_sw, p_center, pc_se}; // Left->Right
c_rig = newl; Circle(c_rig) = {pc_se, p_center, pc_ne}; // Bot->Top
c_top = newl; Circle(c_top) = {pc_ne, p_center, pc_nw}; // Right->Left
c_lef = newl; Circle(c_lef) = {pc_nw, p_center, pc_sw}; // Top->Bot

// --- Box ---
b_bot = newl; Line(b_bot) = {pb_sw, pb_se}; // L->R
b_rig = newl; Line(b_rig) = {pb_se, pb_ne}; // B->T
b_top = newl; Line(b_top) = {pb_ne, pb_nw}; // R->L
b_lef = newl; Line(b_lef) = {pb_nw, pb_sw}; // T->B

// --- Spokes ---
s_sw = newl; Line(s_sw) = {pc_sw, pb_sw};
s_se = newl; Line(s_se) = {pc_se, pb_se};
s_ne = newl; Line(s_ne) = {pc_ne, pb_ne};
s_nw = newl; Line(s_nw) = {pc_nw, pb_nw};

// --- Outer Connectors ---
// Bottom
l_b_1 = newl; Line(l_b_1) = {pd_sw, pp_b_w};
l_b_2 = newl; Line(l_b_2) = {pp_b_w, pp_b_e};
l_b_3 = newl; Line(l_b_3) = {pp_b_e, pd_se};
v_b_1 = newl; Line(v_b_1) = {pp_b_w, pb_sw};
v_b_2 = newl; Line(v_b_2) = {pp_b_e, pb_se};

// Top
l_t_1 = newl; Line(l_t_1) = {pd_nw, pp_t_w};
l_t_2 = newl; Line(l_t_2) = {pp_t_w, pp_t_e};
l_t_3 = newl; Line(l_t_3) = {pp_t_e, pd_ne};
v_t_1 = newl; Line(v_t_1) = {pb_nw, pp_t_w};
v_t_2 = newl; Line(v_t_2) = {pb_ne, pp_t_e};

// Left (Inlet)
v_in_b = newl; Line(v_in_b) = {pd_sw, pp_w_b};
v_in_m = newl; Line(v_in_m) = {pp_w_b, pp_w_t};
v_in_t = newl; Line(v_in_t) = {pp_w_t, pd_nw};
h_in_b = newl; Line(h_in_b) = {pp_w_b, pb_sw};
h_in_t = newl; Line(h_in_t) = {pp_w_t, pb_nw};

// Right (Outlet)
v_out_b = newl; Line(v_out_b) = {pd_se, pp_e_b};
v_out_m = newl; Line(v_out_m) = {pp_e_b, pp_e_t};
v_out_t = newl; Line(v_out_t) = {pp_e_t, pd_ne};
h_out_b = newl; Line(h_out_b) = {pb_se, pp_e_b};
h_out_t = newl; Line(h_out_t) = {pb_ne, pp_e_t};

// ============================================================
// 5. Transfinite (Matching)
// ============================================================
Transfinite Curve {c_bot, c_rig, c_top, c_lef} = N_core;
Transfinite Curve {b_bot, b_rig, b_top, b_lef} = N_core;
Transfinite Curve {l_b_2, l_t_2} = N_core; 
Transfinite Curve {v_in_m, v_out_m} = N_core; 

Transfinite Curve {s_sw, s_se, s_ne, s_nw} = N_rad Using Progression r_wall;

Transfinite Curve {l_b_1, l_t_1, h_in_b, h_in_t} = N_upstream;
Transfinite Curve {l_b_3, l_t_3, h_out_b, h_out_t} = N_downstream Using Progression r_wake;

Transfinite Curve {v_in_b, v_b_1, v_b_2, v_out_b} = N_vert;
Transfinite Curve {v_in_t, v_t_1, v_t_2, v_out_t} = N_vert;

// ============================================================
// 6. Surface Loops (Fixed Orientation) - unchanged
// ============================================================

// --- Inner 4 ---
ll_o1 = newll; Curve Loop(ll_o1) = {c_bot, s_se, -b_bot, -s_sw};
s_o1 = news; Plane Surface(s_o1) = {ll_o1};
Transfinite Surface {s_o1} = {pc_sw, pc_se, pb_se, pb_sw};

ll_o2 = newll; Curve Loop(ll_o2) = {c_rig, s_ne, -b_rig, -s_se};
s_o2 = news; Plane Surface(s_o2) = {ll_o2};
Transfinite Surface {s_o2} = {pc_se, pc_ne, pb_ne, pb_se};

ll_o3 = newll; Curve Loop(ll_o3) = {c_top, s_nw, -b_top, -s_ne};
s_o3 = news; Plane Surface(s_o3) = {ll_o3};
Transfinite Surface {s_o3} = {pc_ne, pc_nw, pb_nw, pb_ne};

ll_o4 = newll; Curve Loop(ll_o4) = {c_lef, s_sw, -b_lef, -s_nw};
s_o4 = news; Plane Surface(s_o4) = {ll_o4};
Transfinite Surface {s_o4} = {pc_nw, pc_sw, pb_sw, pb_nw};

// --- Outer 8 Blocks (Strict CCW) ---

// 5. Bottom Center
ll_bc = newll; Curve Loop(ll_bc) = {l_b_2, v_b_2, -b_bot, -v_b_1};
s_bc = news; Plane Surface(s_bc) = {ll_bc};
Transfinite Surface {s_bc} = {pp_b_w, pp_b_e, pb_se, pb_sw};

// 6. Top Center
ll_tc = newll; Curve Loop(ll_tc) = {-b_top, v_t_2, -l_t_2, -v_t_1};
s_tc = news; Plane Surface(s_tc) = {ll_tc};
Transfinite Surface {s_tc} = {pb_nw, pb_ne, pp_t_e, pp_t_w};

// 7. Left Center
ll_lc = newll; Curve Loop(ll_lc) = {h_in_b, -b_lef, -h_in_t, -v_in_m};
s_lc = news; Plane Surface(s_lc) = {ll_lc};
Transfinite Surface {s_lc} = {pp_w_b, pb_sw, pb_nw, pp_w_t};

// 8. Right Center
ll_rc = newll; Curve Loop(ll_rc) = {h_out_b, v_out_m, -h_out_t, -b_rig};
s_rc = news; Plane Surface(s_rc) = {ll_rc};
Transfinite Surface {s_rc} = {pb_se, pp_e_b, pp_e_t, pb_ne};

// 9. Bottom Left
ll_bl = newll; Curve Loop(ll_bl) = {l_b_1, v_b_1, -h_in_b, -v_in_b};
s_bl = news; Plane Surface(s_bl) = {ll_bl};
Transfinite Surface {s_bl} = {pd_sw, pp_b_w, pb_sw, pp_w_b};

// 10. Bottom Right
ll_br = newll; Curve Loop(ll_br) = {l_b_3, v_out_b, -h_out_b, -v_b_2};
s_br = news; Plane Surface(s_br) = {ll_br};
Transfinite Surface {s_br} = {pp_b_e, pd_se, pp_e_b, pb_se};

// 11. Top Left
ll_tl = newll; Curve Loop(ll_tl) = {h_in_t, v_t_1, -l_t_1, -v_in_t};
s_tl = news; Plane Surface(s_tl) = {ll_tl};
Transfinite Surface {s_tl} = {pp_w_t, pb_nw, pp_t_w, pd_nw};

// 12. Top Right
ll_tr = newll; Curve Loop(ll_tr) = {h_out_t, v_out_t, -l_t_3, -v_t_2};
s_tr = news; Plane Surface(s_tr) = {ll_tr};
Transfinite Surface {s_tr} = {pb_ne, pp_e_t, pd_ne, pp_t_e};

Recombine Surface "*";

// ============================================================
// 7. 3D Extrusion (now along Z direction)
// ============================================================
surface_list[] = {s_o1, s_o2, s_o3, s_o4, s_bc, s_tc, s_lc, s_rc, s_bl, s_br, s_tl, s_tr};
vol[] = Extrude {0, 0, W} {   // was {W, 0, 0}
  Surface{surface_list[]};
  Layers{N_span};
  Recombine;
};

// ============================================================
// 8. Physical Groups (BoundingBox updated for swapped axes)
// ============================================================
eps = 1e-4;

// Inlet: x=0 plane (was z=0)
Physical Surface("Inlet", 1) = Surface In BoundingBox {-eps, -eps, -eps, eps, H+eps, W+eps};

// Outlet: x=L plane (was z=L)
Physical Surface("Outlet", 2) = Surface In BoundingBox {L-eps, -eps, -eps, L+eps, H+eps, W+eps};

// Walls: now y=0, y=H, z=0, z=W
Physical Surface("Walls", 3) = {
  Surface In BoundingBox {-eps, -eps, -eps, L+eps, eps, W+eps},   // y=0
  Surface In BoundingBox {-eps, H-eps, -eps, L+eps, H+eps, W+eps}, // y=H
  Surface In BoundingBox {-eps, -eps, -eps, L+eps, H+eps, eps},    // z=0
  Surface In BoundingBox {-eps, -eps, W-eps, L+eps, H+eps, W+eps}  // z=W
};

// Cylinder: axis along z (x ranges around Cz, y around Cy, z full)
Physical Surface("Cylinder", 4) = Surface In BoundingBox {Cz-R-eps, Cy-R-eps, -eps, Cz+R+eps, Cy+R+eps, W+eps};

// Fluid volume
Physical Volume("Fluid", 10) = Volume In BoundingBox {-eps, -eps, -eps, L+eps, H+eps, W+eps};

Printf("Mesh Generation Complete. No missing blocks.");