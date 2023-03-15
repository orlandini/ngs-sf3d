from ngsolve import *
from numpy import random
import netgen.gui
from netgen.occ import *
import os
from geo import GenMesh
from modal import ModalAnalysis

ngsglobals.msg_level = 1

MESH_FILE_NAME = "sf3d.vol"
# all dimensions are in microns

wl = 1.5  # wavelength (in microns)

urvals = 1  # relative magnetic permeability
# refractive indices
nclad = 1.4378
ncore = 1.4457

r_cyl = 8  # core radius
# distance from center to end of cladding region(inner box)
d_box = r_cyl + 3.5 * wl/nclad
l_domain = 2*wl
d_pml = 1.5*wl/nclad  # pml width
elsize = 0.8
# element sizes are different in cladding or core
el_clad = elsize*wl/nclad  # el size in cladding
el_core = elsize*wl/ncore  # el size in core
p_modal = 2  # polynomial order of hcurl space in modal analysis
p_scatt = 2  # polynomial order of scattering analysis

gen_mesh = True
if gen_mesh or not os.path.isfile(MESH_FILE_NAME):
    GenMesh(d_box, l_domain, r_cyl, d_pml, el_core, el_clad, MESH_FILE_NAME)

mesh = Mesh("sf3d.vol").Curve(3)

# setting up PMLs

alphapml = 1.2j/d_pml

boxmin = [-d_box, -d_box, -l_domain/2]
boxmax = [d_box, d_box, l_domain/2]
mesh.SetPML(
    pml.Cartesian(mins=boxmin, maxs=boxmax, alpha=alphapml),
    "pml_core_back|pml_clad_back|pml_core_front|pml_clad_front|pml_clad_2d")


# checking surface domains
surflist = {"core_2d": 1, "clad_2d": 2, "dirichlet_3d": 3, 'pml_clad_2d':4}
surf = mesh.BoundaryCF(surflist,-1)
gu = GridFunction(H1(mesh), name='surfs')
gu.Set(surf, definedon=~mesh.Boundaries(''))
Draw(gu)
# print("Checking 2D domains... press Enter to continue")
# input()

# #checking volume domains
# vollist = {"core":1, "clad":2, "pml_clad_back":3, "pml_core_back":4,
#            "pml_clad_front":5, "pml_core_front":6}
# vol = CoefficientFunction([vollist.get(m,0) for m in mesh.GetMaterials()])
# Draw(gu,mesh,"vols",draw_surf=False)
# print("Checking 3D domains... press Enter to continue")
# input()


# creating constitutive params for 2d problem
er2dlist = {"core_2d": ncore**2, "clad_2d": nclad**2, "pml_clad_2d":nclad**2}
er2d = mesh.BoundaryCF(er2dlist)

ur2dlist = {"core_2d": 1, "clad_2d": 1, "pml_clad_2d": 1}

ur2d = mesh.BoundaryCF(ur2dlist)
# here we could check if ur2d and er2d are correctly defined
# gu = GridFunction(H1(mesh),name='ur')
# gu.Set(ur2d, definedon=~mesh.Boundaries(''))
# Draw(gu)
# ge = GridFunction(H1(mesh),name='er')
# ge.Set(er2d, definedon=~mesh.Boundaries(''))
# Draw(ge)
# input()

"""
We are now going to perform the modal analysis of our step-fiber.
For such, we use the formulation taken from
Full-Wave Analysis of Dielectric Waveguides Using Tangential
Vector Finite Elements, by Jin-Fa Lee, Din-Kow Sun and Zoltan J. Cendes, 1991
doi: 10.1109/22.85399

In this approach, the Electric(or magnetic) field is decomposed into its
transverse and longitudinal component.
The analysis is performed over the cross section of the waveguide and
the resulting algebraic problem is a Generalised Eigenvalue Problem
with the propagation constant beta as the eigenvalue and
the transformed field components et, ez as the eigenvector.
The transformed field components are obtained by the following transformation
et = Beta Et
ez = -j Ez

A Hcurl(resp H1) conforming space is used for et(resp ez), and their
finite dimensional counterpart is chosen as to respect the de rham diagram
"""


# creating FEM spaces for 2D problem

V2d = HCurl(
    mesh, order=p_modal, complex=True, dirichlet_bbnd="dirichlet_2d",
    definedon=mesh.Boundaries("clad_2d|core_2d|pml_clad_2d"))
Q2d = H1(
    mesh, order=p_modal + 1, complex=True, dirichlet_bbnd="dirichlet_2d",
    definedon=mesh.Boundaries("clad_2d|core_2d|pml_clad_2d"))

fes2d = V2d*Q2d


# expected eigenvalues are around effective index * k0**2
kzero = 2*pi/wl
target = -ncore*ncore * kzero * kzero
ev, sol2d = ModalAnalysis(fes2d, wl, ur2d, er2d, target)
Draw(sol2d.components[0], mesh, "sol2d_hcurl", draw_surf=True, draw_vol=False)
Draw(sol2d.components[1], mesh, "sol2d_hone", draw_surf=True, draw_vol=False)

# now we go to the 3D problem

# defining ur and er for volume domains
erlist = {"core": ncore**2, "clad": nclad**2,
          "pml_clad_back": nclad**2, "pml_core_back": ncore**2,
          "pml_clad_front": nclad**2, "pml_core_front": ncore**2}

er = CoefficientFunction([erlist.get(m, 0) for m in mesh.GetMaterials()])
ur = CoefficientFunction(1)


fes3d = HCurl(mesh, order=p_scatt, complex=True,
              dirichlet="dirichlet_3d")


u, v = fes3d.TnT()

"""
In the 3D analysis, we approximate the full-wave equation for the electric field,
namely

curl(1./mu_r curl(E)) - k_0^2 e_r E = -j omega J

Instead of inserting our source as an equivalent current source, we estabilish that
at the cross-section Gamma (z = 0) in which the modal analysis was performed, our field
can be decomposed into E = E^s + E^inc, where E^inc is the incident field

Therefore, our volume integral reads as

(1./mu_r curl E) . (curl F) - k_0^2 e_r E . F dx = 0

and at Gamma we have

(n \cross (1./mu_r curl (E1-E2))). F ds

where n is the normal vector (+z), E1 is the incident field on the left side (negative z)
and E2 is the incident field on the right side (positive z).
The scattered fields are the same on both sides

both fields can be expressed as

E1 = Em(x,y) e^{+j beta z} + Es
E2 = Em(x,y) e^{-j beta z} + Es

where Em is the modal analysis obtained field, and Es is the scattered field.
Calling n1 the outward normal vector in the left side and n2 the normal vector in the right side,  we then have

n1 \cross curl(E1) =  z \cross (curl E1)
                   =  z \cross (curl Es) + grad_t Emz - j beta Emt
n2 \cross curl(E2) = -z \cross (curl E2)
                   = -z \cross (curl Es) - grad_t Emz - j beta Emt


Where Emt is the tangential component of the Em field and Emz is the longitudinal component.

Assuming now that the curl Es is continuous on the interface (can we assume that?),

we have

n1 \cross curl E1 + n2 \cross curl E2 =

-2 j beta (Emx, Emy, 0) . F ds = - 2 j beta Em.Trace() * F.Trace() ds
"""

kzero = 2*pi/wl
# we take the dominant mode
which = 0
beta = sqrt(-ev[which])
sol2d_hcurl = sol2d.components[0].MDComponent(which)
a = BilinearForm(fes3d, symmetric=True)
a += ((1./ur) * curl(u) * curl(v) - kzero**2 * er * u * v)*dx
c = Preconditioner(a, "bddc")

f = LinearForm(fes3d)
# recalling that sold2d_hcurl is actually Et * beta,
# we dont need to insert it in the linear form

#ps: i have ommited the ur since it is unitary and i wasnt sure
#if the coefficient function would play well on the 2d domain
f += (2j * sol2d_hcurl.Trace() * v.Trace())*ds("clad_2d|core_2d|pml_clad_2d")

with TaskManager():
    a.Assemble()
    f.Assemble()

gfu = GridFunction(fes3d)

res = f.vec.CreateVector()
# with TaskManager():
#     gfu.vec.data = a.mat.Inverse(
#         fes3d.FreeDofs(),
#         inverse="pardiso") * f.vec



print("solving system with ndofs {}".format(sum(fes3d.FreeDofs())))
with TaskManager():
    gmr = GMRESSolver(mat=a.mat, pre=c.mat, maxsteps=200,
                      precision=1e-13, printrates=True)
    gfu.vec.data = gmr * f.vec

bc_projector = Projector(fes3d.FreeDofs(), True)
res.data = bc_projector*(f.vec - a.mat * gfu.vec)
print("res norm {}".format(Norm(res)))

Draw(gfu, mesh, "sol3d")
