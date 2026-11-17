// cylinder_quad_fixed.geo
// DFG 2D Benchmark - Structured Quadrilateral Mesh
// Fixed the curve-loop orientation bug in the mid-right block

// ==========================================
// 1. Parameters
// ==========================================
L = 2.2;
H = 0.41;
R = 0.05;
Cx = 0.2;
Cy = 0.2;

// O-grid box extents
Bx_min = 0.1; Bx_max = 0.3;
By_min = 0.1; By_max = 0.3;

// --- Mesh point counts ---
n_circ = 30;   // Circumferential direction
n_radial = 20; // Boundary-layer radial direction
n_inlet = 15;  // Upstream length
n_wake = 80;   // Wake length
n_vert = 15;   // Vertical branch

// --- Grading ratios ---
prog_radial = 1.15; // Boundary-layer refinement
prog_wake   = 1.03; // Wake refinement

// ==========================================
// 2. Point definitions
// ==========================================
p0 = newp; Point(p0) = {Cx, Cy, 0};

// A. Cylinder points
dR = 0.70710678 * R;
pc_ne = newp; Point(pc_ne) = {Cx+dR, Cy+dR, 0}; 
pc_nw = newp; Point(pc_nw) = {Cx-dR, Cy+dR, 0};
pc_sw = newp; Point(pc_sw) = {Cx-dR, Cy-dR, 0};
pc_se = newp; Point(pc_se) = {Cx+dR, Cy-dR, 0};

// B. O-grid box corners
pb_ne = newp; Point(pb_ne) = {Bx_max, By_max, 0};
pb_nw = newp; Point(pb_nw) = {Bx_min, By_max, 0};
pb_sw = newp; Point(pb_sw) = {Bx_min, By_min, 0};
pb_se = newp; Point(pb_se) = {Bx_max, By_min, 0};

// C. Split points on the outer boundary
pi_nw = newp; Point(pi_nw) = {0, By_max, 0};
pi_sw = newp; Point(pi_sw) = {0, By_min, 0};
po_ne = newp; Point(po_ne) = {L, By_max, 0};
po_se = newp; Point(po_se) = {L, By_min, 0};
pt_nw = newp; Point(pt_nw) = {Bx_min, H, 0};
pt_ne = newp; Point(pt_ne) = {Bx_max, H, 0};
pd_sw = newp; Point(pd_sw) = {Bx_min, 0, 0};
pd_se = newp; Point(pd_se) = {Bx_max, 0, 0};

// D. Domain corners
corn_sw = newp; Point(corn_sw) = {0, 0, 0};
corn_nw = newp; Point(corn_nw) = {0, H, 0};
corn_se = newp; Point(corn_se) = {L, 0, 0};
corn_ne = newp; Point(corn_ne) = {L, H, 0};

// ==========================================
// 3. Line definitions
// ==========================================

// --- Inner circle (counter-clockwise) ---
c_right = newc; Circle(c_right) = {pc_se, p0, pc_ne};
c_top   = newc; Circle(c_top)   = {pc_ne, p0, pc_nw};
c_left  = newc; Circle(c_left)  = {pc_nw, p0, pc_sw};
c_bot   = newc; Circle(c_bot)   = {pc_sw, p0, pc_se};

// --- Spokes ---
l_s_ne = newl; Line(l_s_ne) = {pc_ne, pb_ne};
l_s_nw = newl; Line(l_s_nw) = {pc_nw, pb_nw};
l_s_sw = newl; Line(l_s_sw) = {pc_sw, pb_sw};
l_s_se = newl; Line(l_s_se) = {pc_se, pb_se};

// --- Box edges (directions chosen for loop orientation) ---
l_b_right = newl; Line(l_b_right) = {pb_se, pb_ne}; // Up
l_b_top   = newl; Line(l_b_top)   = {pb_nw, pb_ne}; // Left -> Right
// Keep the top edge in its original direction and use a negative sign
// in curve loops whenever the opposite orientation is required.

l_b_left  = newl; Line(l_b_left)  = {pb_sw, pb_nw}; // Up
l_b_bot   = newl; Line(l_b_bot)   = {pb_sw, pb_se}; // Left -> Right

// --- Outer Horizontal Lines ---
// Top group (Left -> Right)
l_h_top_l = newl; Line(l_h_top_l) = {corn_nw, pt_nw};
l_h_top_c = newl; Line(l_h_top_c) = {pt_nw, pt_ne};
l_h_top_r = newl; Line(l_h_top_r) = {pt_ne, corn_ne};
// Mid Top group (Left -> Right)
l_h_mid_tl = newl; Line(l_h_mid_tl) = {pi_nw, pb_nw};
l_h_mid_tr = newl; Line(l_h_mid_tr) = {pb_ne, po_ne};
// Mid Bot group (Left -> Right)
l_h_mid_bl = newl; Line(l_h_mid_bl) = {pi_sw, pb_sw};
l_h_mid_br = newl; Line(l_h_mid_br) = {pb_se, po_se};
// Bot group (Left -> Right)
l_h_bot_l = newl; Line(l_h_bot_l) = {corn_sw, pd_sw};
l_h_bot_c = newl; Line(l_h_bot_c) = {pd_sw, pd_se};
l_h_bot_r = newl; Line(l_h_bot_r) = {pd_se, corn_se};

// --- Outer Vertical Lines (Bottom -> Top) ---
// Inlet side
l_v_in_top = newl; Line(l_v_in_top) = {pi_nw, corn_nw};
l_v_in_mid = newl; Line(l_v_in_mid) = {pi_sw, pi_nw};
l_v_in_bot = newl; Line(l_v_in_bot) = {corn_sw, pi_sw};
// Mid Left side
l_v_ml_top = newl; Line(l_v_ml_top) = {pb_nw, pt_nw};
l_v_ml_bot = newl; Line(l_v_ml_bot) = {pd_sw, pb_sw};
// Mid Right side
l_v_mr_top = newl; Line(l_v_mr_top) = {pb_ne, pt_ne};
l_v_mr_bot = newl; Line(l_v_mr_bot) = {pd_se, pb_se};
// Outlet side
l_v_out_top = newl; Line(l_v_out_top) = {po_ne, corn_ne};
l_v_out_mid = newl; Line(l_v_out_mid) = {po_se, po_ne};
l_v_out_bot = newl; Line(l_v_out_bot) = {corn_se, po_se};

// ==========================================
// 4. Curve loops
// ==========================================

// --- Inner Blocks ---
// Right: Circle(Up) -> Spoke(Out) -> Box(Down) -> Spoke(In)
ll_in_r = newll; Curve Loop(ll_in_r) = {c_right, l_s_ne, -l_b_right, -l_s_se};
s_in_r = news; Plane Surface(s_in_r) = {ll_in_r};

// Top: Circle(Left) -> Spoke(Out) -> Box(Right) -> Spoke(In)
// l_b_top is pb_nw -> pb_ne. We need pb_ne -> pb_nw. So -l_b_top.
ll_in_t = newll; Curve Loop(ll_in_t) = {c_top, l_s_nw, l_b_top, -l_s_ne}; 
// Wait, l_b_top is pb_nw -> pb_ne. 
// Path: pc_ne -> pc_nw -> pb_nw -> pb_ne -> pc_ne
//       c_top    l_s_nw   l_b_top   -l_s_ne
s_in_t = news; Plane Surface(s_in_t) = {ll_in_t};

// Left: Circle(Down) -> Spoke(Out) -> Box(Up) -> Spoke(In)
ll_in_l = newll; Curve Loop(ll_in_l) = {c_left, l_s_sw, l_b_left, -l_s_nw};
s_in_l = news; Plane Surface(s_in_l) = {ll_in_l};

// Bottom: Circle(Right) -> Spoke(Out) -> Box(Left) -> Spoke(In)
ll_in_b = newll; Curve Loop(ll_in_b) = {c_bot, l_s_se, -l_b_bot, -l_s_sw};
s_in_b = news; Plane Surface(s_in_b) = {ll_in_b};


// --- Outer Blocks ---
// 1. Top Left
ll_o_tl = newll; Curve Loop(ll_o_tl) = {l_h_top_l, -l_v_ml_top, -l_h_mid_tl, l_v_in_top};
s_o_tl = news; Plane Surface(s_o_tl) = {ll_o_tl};

// 2. Mid Left
ll_o_ml = newll; Curve Loop(ll_o_ml) = {l_h_mid_tl, -l_b_left, -l_h_mid_bl, l_v_in_mid};
s_o_ml = news; Plane Surface(s_o_ml) = {ll_o_ml};

// 3. Bot Left
ll_o_bl = newll; Curve Loop(ll_o_bl) = {l_h_mid_bl, -l_v_ml_bot, -l_h_bot_l, l_v_in_bot};
s_o_bl = news; Plane Surface(s_o_bl) = {ll_o_bl};

// 4. Center Top
// Path: pt_nw -> pt_ne -> pb_ne -> pb_nw -> pt_nw
// Lines: l_h_top_c, -l_v_mr_top, -l_b_top, l_v_ml_top
// l_b_top is pb_nw -> pb_ne. So -l_b_top is pb_ne -> pb_nw. Correct.
ll_o_ct = newll; Curve Loop(ll_o_ct) = {l_h_top_c, -l_v_mr_top, -l_b_top, l_v_ml_top};
s_o_ct = news; Plane Surface(s_o_ct) = {ll_o_ct};

// 5. Center Bot
ll_o_cb = newll; Curve Loop(ll_o_cb) = {l_b_bot, -l_v_mr_bot, -l_h_bot_c, l_v_ml_bot};
s_o_cb = news; Plane Surface(s_o_cb) = {ll_o_cb};

// 6. Top Right
ll_o_tr = newll; Curve Loop(ll_o_tr) = {l_h_top_r, -l_v_out_top, -l_h_mid_tr, l_v_mr_top};
s_o_tr = news; Plane Surface(s_o_tr) = {ll_o_tr};

// 7. Mid right (the loop-orientation fix is here)
// Path: pb_ne -> po_ne -> po_se -> pb_se -> pb_ne
// Lines: l_h_mid_tr (pb_ne->po_ne) 
//        -l_v_out_mid (po_ne->po_se)
//        -l_h_mid_br (po_se->pb_se)
//        l_b_right (pb_se->pb_ne) -- previously this used -l_b_right, which reversed the loop
ll_o_mr = newll; Curve Loop(ll_o_mr) = {l_h_mid_tr, -l_v_out_mid, -l_h_mid_br, l_b_right};
s_o_mr = news; Plane Surface(s_o_mr) = {ll_o_mr};

// 8. Bot Right
ll_o_br = newll; Curve Loop(ll_o_br) = {l_h_mid_br, -l_v_out_bot, -l_h_bot_r, l_v_mr_bot};
s_o_br = news; Plane Surface(s_o_br) = {ll_o_br};


// ==========================================
// 5. Mesh control
// ==========================================
Transfinite Curve {l_s_ne, l_s_nw, l_s_sw, l_s_se} = n_radial Using Progression prog_radial;
Transfinite Curve {c_right, c_top, c_left, c_bot} = n_circ;
Transfinite Curve {l_b_right, l_b_top, l_b_left, l_b_bot} = n_circ;
Transfinite Curve {l_h_top_c, l_h_bot_c} = n_circ;
Transfinite Curve {l_v_in_mid, l_v_out_mid} = n_circ;

Transfinite Curve {l_h_top_l, l_h_mid_tl, l_h_mid_bl, l_h_bot_l} = n_inlet;
Transfinite Curve {l_h_top_r, l_h_mid_tr, l_h_mid_br, l_h_bot_r} = n_wake Using Progression prog_wake;
Transfinite Curve {l_v_in_top, l_v_ml_top, l_v_mr_top, l_v_out_top} = n_vert;
Transfinite Curve {l_v_in_bot, l_v_ml_bot, l_v_mr_bot, l_v_out_bot} = n_vert;


// ==========================================
// 6. Explicit corner ordering
// ==========================================

// Inner
Transfinite Surface {s_in_r} = {pc_se, pc_ne, pb_ne, pb_se};
Transfinite Surface {s_in_t} = {pc_ne, pc_nw, pb_nw, pb_ne};
Transfinite Surface {s_in_l} = {pc_nw, pc_sw, pb_sw, pb_nw};
Transfinite Surface {s_in_b} = {pc_sw, pc_se, pb_se, pb_sw};

// Outer
Transfinite Surface {s_o_tl} = {corn_nw, pt_nw, pb_nw, pi_nw};
Transfinite Surface {s_o_ml} = {pi_nw, pb_nw, pb_sw, pi_sw};
Transfinite Surface {s_o_bl} = {pi_sw, pb_sw, pd_sw, corn_sw};

Transfinite Surface {s_o_ct} = {pt_nw, pt_ne, pb_ne, pb_nw};
Transfinite Surface {s_o_cb} = {pb_sw, pb_se, pd_se, pd_sw};

Transfinite Surface {s_o_tr} = {pt_ne, corn_ne, po_ne, pb_ne};
Transfinite Surface {s_o_mr} = {pb_ne, po_ne, po_se, pb_se}; // Mid Right
Transfinite Surface {s_o_br} = {pb_se, po_se, corn_se, pd_se};

Recombine Surface "*";

// ==========================================
// 7. Physical groups
// ==========================================
Physical Curve("Inlet", 1) = {l_v_in_top, l_v_in_mid, l_v_in_bot};
Physical Curve("Outlet", 2) = {l_v_out_top, l_v_out_mid, l_v_out_bot};
Physical Curve("Walls", 3) = {l_h_top_l, l_h_top_c, l_h_top_r, l_h_bot_l, l_h_bot_c, l_h_bot_r};
Physical Curve("Cylinder", 4) = {c_right, c_top, c_left, c_bot};

Physical Surface("Domain", 100) = {s_in_r, s_in_t, s_in_l, s_in_b, 
                                   s_o_tl, s_o_ml, s_o_bl, 
                                   s_o_ct, s_o_cb, 
                                   s_o_tr, s_o_mr, s_o_br};