from viewOpt import loadView
from ngsolve import *
from numpy import random
import netgen.gui
from netgen.occ import *
from ngsolve.krylovspace import GMRes
import os
from geo import GenMeshRectangularWaveguide
from modal import ModalAnalysis
from mygmres import MyGMRes

SetNumThreads(7)
ngsglobals.msg_level = 1

MESH_FILE_NAME = "wr90.vol"
# all dimensions are in microns

c0 = 299792458
f0 = 25*10**9
wl =(c0/f0) * 1000  # wavelength (in mm)

ur = 1  # relative magnetic permeability
er = 1 # relative electric permittivity

width = 28.86
height = 10.16
l_domain = 2.0*wl
d_pml = 2.0*wl  # pml width
elsize = wl/10
# element sizes are different in cladding or core
p_modal = 2  # polynomial order of hcurl space in modal analysis
p_scatt = 2  # polynomial order of scattering analysis

custom_pml = True
gen_mesh = True
if gen_mesh or not os.path.isfile(MESH_FILE_NAME):
    GenMeshRectangularWaveguide(width,height,l_domain,d_pml,elsize,custom_pml,MESH_FILE_NAME)

mesh = Mesh(MESH_FILE_NAME)


# checking surface domains
surflist = {"air_2d": 1, "dirichlet_3d": 4}
surf = mesh.BoundaryCF(surflist, -1)
gu = GridFunction(H1(mesh), name='surfs')
gu.Set(surf, definedon=~mesh.Boundaries(''))
Draw(gu)
# print("Checking 2D domains... press Enter to continue")
# input()

#checking volume domains

pml_domains = 'pml_front|pml_back' if custom_pml else 'pml'


vollist = {"air":1} | {d:i+2 for i,d in enumerate(pml_domains.split('|'))}
vol = CoefficientFunction([vollist.get(m,0) for m in mesh.GetMaterials()])
Draw(vol,mesh,"vols",draw_surf=False)
# print("Checking 3D domains... press Enter to continue")
# input()


# creating constitutive params for 2d problem
er2d = mesh.BoundaryCF({"air_2d":er})
ur2d = mesh.BoundaryCF({"air_2d":ur})
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
domains_2d = "air_2d"

V2d = HCurl(
    mesh, order=p_modal, complex=True, dirichlet_bbnd="dirichlet_2d",
    definedon=mesh.Boundaries(domains_2d))
Q2d = H1(
    mesh, order=p_modal + 1, complex=True, dirichlet_bbnd="dirichlet_2d",
    definedon=mesh.Boundaries(domains_2d))

fes2d = V2d*Q2d


# expected eigenvalues are around effective index * k0**2
kzero = 2*pi/wl
target = -er * kzero * kzero
ev, sol2d = ModalAnalysis(fes2d, wl, ur2d, er2d, target,domains_2d)
Draw(sol2d.components[0], mesh, "sol2d_hcurl", draw_surf=True, draw_vol=False)
Draw(sol2d.components[1], mesh, "sol2d_hone", draw_surf=True, draw_vol=False)

# now we go to the 3D problem
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

# defining ur and er for volume domains
alpha_pml = .1j

alpha_pml = alpha_pml/d_pml if custom_pml else alpha_pml


if custom_pml:
    sz_d = {"air": 1,
            "pml_back":  1- alpha_pml * (z+l_domain/2)*(z+l_domain/2)/(d_pml*d_pml),
            "pml_front":  1- alpha_pml * (z-l_domain/2)*(z-l_domain/2)/(d_pml*d_pml)
            }

    sz = CoefficientFunction([sz_d[mat] for mat in mesh.GetMaterials()])

    er3d = CoefficientFunction((er*sz,0,0,0,er*sz,0,0,0,er/sz),dims=(3,3))
    urinv3d = CoefficientFunction((ur/sz,0,0,0,ur/sz,0,0,0,ur*sz),dims=(3,3))
else:


    boxmin = [-width/2, -height/2, -l_domain/2]
    boxmax = [width/2, height/2, l_domain/2]
    mesh.SetPML(
        pml.Cartesian(mins=boxmin, maxs=boxmax, alpha=alpha_pml),pml_domains)

    er3d = 1
    urinv3d = 1

kzero = 2*pi/wl
# we take the dominant mode
which = 0
beta = sqrt(-ev[which])
sol2d_hcurl = sol2d.components[0].MDComponent(which)
a = BilinearForm(fes3d, symmetric=False)
a += (urinv3d * curl(u) * curl(v) - kzero**2  * er3d * u * v)*dx
# a += ((urinv3d * curl(u)) * (curl(v)) - kzero**2  * (er3d * u) * (v))*dx("air")
# a += ((urinv3d * curl(u)) * (curl(v)) - kzero**2  * (er3d * u) * (v))*dx(pml_domains,intrules={TET:IntegrationRule(TET,15)})
c = Preconditioner(a, type="bddc")

f = LinearForm(fes3d)
# recalling that sold2d_hcurl is actually Et * beta,
# we dont need to insert it in the linear form

# ps: i have ommited the ur since it is unitary and i wasnt sure
# if the coefficient function would play well on the 2d domain
f += (2j * sol2d_hcurl.Trace() * v.Trace())*ds("air_2d")

with TaskManager():
    a.Assemble()
    f.Assemble()

gfu = GridFunction(fes3d)

res = f.vec.CreateVector()


print("solving system with ndofs {}".format(sum(fes3d.FreeDofs())))




# with TaskManager():
#     gfu.vec.data = a.mat.Inverse(
#         fes3d.FreeDofs(),
#         inverse="pardiso") * f.vec

with TaskManager():
    gfu.vec.data = MyGMRes(a.mat, f.vec, pre=c.mat,
                           maxsteps=300, tol=1e-15, printrates=True)

bc_projector = Projector(fes3d.FreeDofs(), True)
res.data = bc_projector*(f.vec - a.mat * gfu.vec)
print("res norm {}".format(Norm(res)))

Draw(gfu, mesh, "sol3d")


Draw(CF((gfu[0], gfu[1], 0)), mesh, "sol3dxy")


# loadView()
