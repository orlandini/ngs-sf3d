from ngsolve import *
from numpy import random
import netgen.gui
from netgen.occ import *
import os

from geo import GenMesh
from modal import ModalAnalysis

MESH_FILE_NAME = "sf3d.vol"
#all dimensions are in microns

wl = 1.5#wavelength (in microns)

urvals = 1;#relative magnetic permeability
#refractive indices
nclad = 1.4378
ncore = 1.4457

r_cyl = 8#core radius
d_box = r_cyl + 3.5 * wl/nclad#distance from center to end of cladding region(inner box)
l_domain = 6*wl
d_pml = 0.5*wl/nclad#pml width
elsize = 0.75
#element sizes are different in cladding or core
el_clad = elsize*wl/nclad#el size in cladding
el_core = elsize*wl/ncore#el size in core
p_modal = 1#polynomial order of hcurl space in modal analysis
p_scatt = 1#polynomial order of scattering analysis

gen_mesh = True
if gen_mesh or not os.path.isfile(MESH_FILE_NAME):
    GenMesh(d_box,l_domain,r_cyl,d_pml, el_core,el_clad,MESH_FILE_NAME)

mesh = Mesh("sf3d.vol").Curve(3)

#checking surface domains
surflist = {"core_2d":1, "clad_2d":2, "dirichlet_3d":3}
surf = mesh.BoundaryCF(surflist)
gu = GridFunction(H1(mesh),name='surfs')
gu.Set(surf, definedon=~mesh.Boundaries(''))
Draw(mesh)
print("Checking 2D domains... press Enter to continue")
# input()

# #checking volume domains
# vollist = {"core":1, "clad":2, "pml_clad_back":3, "pml_core_back":4,
#            "pml_clad_front":5, "pml_core_front":6}
# vol = CoefficientFunction([vollist.get(m,0) for m in mesh.GetMaterials()])
# Draw(gu,mesh,"vols",draw_surf=False)
# print("Checking 3D domains... press Enter to continue")
# input()


#creating constitutive params for 2d problem
er2dlist = {"core_2d":ncore**2, "clad_2d":nclad**2}
er2d = mesh.BoundaryCF(er2dlist)

ur2dlist = {"core_2d":1, "clad_2d":1}
             
ur2d = mesh.BoundaryCF(ur2dlist)
# here we could check if ur2d and er2d are correctly defined
# gu = GridFunction(H1(mesh),name='ur')
# gu.Set(ur2d, definedon=~mesh.Boundaries(''))
# Draw(gu)
# ge = GridFunction(H1(mesh),name='er')
# ge.Set(er2d, definedon=~mesh.Boundaries(''))
# Draw(ge)
# input()



#creating FEM spaces for 2D problem

V2d = HCurl(mesh, order = p_modal, complex=True,
          dirichlet_bbnd = "dirichlet_2d", definedon = mesh.Boundaries("clad_2d|core_2d"))
Q2d = H1(mesh, order = p_modal+1, complex=True,
          dirichlet_bbnd = "dirichlet_2d", definedon = mesh.Boundaries("clad_2d|core_2d"))

fes2d = V2d*Q2d


#expected eigenvalues are around effective index * k0**2
kzero = 2*pi/wl
target = -ncore*ncore * kzero * kzero
ev, sol2d = ModalAnalysis(fes2d,wl,ur2d,er2d, target)

Draw(sol2d.components[0],mesh,"sol2d_hcurl", draw_surf=True,draw_vol=False)
Draw(sol2d.components[1],mesh,"sol2d_hone", draw_surf=True,draw_vol=False)


#now we go for the 3d problem
alphapml = 0.5j

mesh.SetPML(pml.Cartesian(mins=[0,0,-l_domain/2],maxs=[0,0,-(l_domain/2+d_pml)],alpha=alphapml), "pml_core_back")
mesh.SetPML(pml.Cartesian(mins=[0,0,-l_domain/2],maxs=[0,0,-(l_domain/2+d_pml)],alpha=alphapml), "pml_clad_back")
mesh.SetPML(pml.Cartesian(mins=[0,0,l_domain/2],maxs=[0,0,(l_domain/2+d_pml)],alpha=alphapml), "pml_core_front")
mesh.SetPML(pml.Cartesian(mins=[0,0,l_domain/2],maxs=[0,0,(l_domain/2+d_pml)],alpha=alphapml), "pml_clad_front")


#defining ur and er for volume domains
erlist = {"core":ncore**2, "clad":nclad**2,
          "pml_clad_back":nclad**2, "pml_core_back":ncore**2,
          "pml_clad_front":nclad**2, "pml_core_front":ncore**2}
er = CoefficientFunction([erlist.get(m,0) for m in mesh.GetMaterials()])
ur = CoefficientFunction(1)

fes3d = HCurl(mesh, order = p_scatt, complex=True,
            dirichlet = "dirichlet_3d")

u, v = fes3d.TnT()

kzero = 2*pi/wl
#we take the dominant mode
beta = sqrt(-ev[0])
sol2d_hcurl = sol2d.components[0]
a = BilinearForm(fes3d,symmetric=True)
a += ((1./ur) * curl(u) * curl(v) - kzero**2 * er * u * v)*dx
a.Assemble()
c = Preconditioner(a, "bddc")

f = LinearForm(fes3d)
# f += (-2j * beta * (sol2d_hcurl[0] * v.Trace()[0] - sol2d_hcurl[1] * v.Trace()[1])*ds("clad_2d|core_2d"))
f += (-2j * sol2d_hcurl * v.Trace())*ds("clad_2d|core_2d")

with TaskManager():
    a.Assemble()
    f.Assemble()

gfu = GridFunction(fes3d)
with TaskManager():
    gmr = GMRESSolver(mat=a.mat, pre=c.mat, maxsteps=200, precision=1e-12,printrates=True)
    gfu.vec.data = gmr * f.vec

Draw(gfu,mesh,"sol3d")