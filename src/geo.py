from netgen.occ import *
from ngsolve import Mesh


def GenMesh(d_box, l_domain, r_cyl, d_pml, el_core, el_clad, filename):
    cube_back = Box(Pnt(-d_box, -d_box, -l_domain/2), Pnt(d_box, d_box, 0))
    cube_front = Box(Pnt(-d_box, -d_box, 0), Pnt(d_box, d_box, l_domain/2))
    cyl = Cylinder(Pnt(0, 0, -l_domain/2), Z, r=r_cyl, h=l_domain)

    clad_front = cube_front - cyl
    clad_front.mat('clad').maxh = el_clad
    clad_back = cube_back - cyl
    clad_back.mat('clad').maxh = el_clad
    core_front = cube_front * cyl
    core_front.mat('core').maxh = el_core
    core_back = cube_back * cyl
    core_back.mat('core').maxh = el_core

    clad_front.faces.name = 'default'
    clad_front.faces.Min(Z).name = 'clad_2d'
    core_front.faces.name = 'default'
    core_front.faces.Min(Z).name = 'core_2d'
    clad_front.faces.Min(Z).edges.Max(X).name = 'dirichlet_2d'
    clad_front.faces.Min(Z).edges.Max(Y).name = 'dirichlet_2d'
    clad_front.faces.Min(Z).edges.Min(X).name = 'dirichlet_2d'
    clad_front.faces.Min(Z).edges.Min(Y).name = 'dirichlet_2d'

    clad_front.faces.Max(X).name = 'dirichlet_3d'
    clad_front.faces.Max(Y).name = 'dirichlet_3d'
    clad_front.faces.Min(X).name = 'dirichlet_3d'
    clad_front.faces.Min(Y).name = 'dirichlet_3d'
    clad_back.faces.Max(X).name = 'dirichlet_3d'
    clad_back.faces.Max(Y).name = 'dirichlet_3d'
    clad_back.faces.Min(X).name = 'dirichlet_3d'
    clad_back.faces.Min(Y).name = 'dirichlet_3d'

    domain_list = [clad_front, clad_back, core_front, core_back]
    # pml front and back

    pml_cube_front = Box(
        Pnt(-d_box, -d_box, l_domain / 2),
        Pnt(d_box, d_box, (l_domain / 2 + d_pml)))
    pml_cube_front.faces.name = 'dirichlet_3d'
    pml_cube_front.faces.Min(Z).name = 'default'  # revert this one
    cyl_front = Cylinder(Pnt(0, 0, l_domain/2), Z, r=r_cyl, h=d_pml)
    pml_clad_front = pml_cube_front - cyl_front
    pml_core_front = pml_cube_front * cyl_front
    pml_clad_front.mat("pml_clad_front").maxh = el_clad
    pml_core_front.mat("pml_core_front").maxh = el_core

    pml_cube_back = Box(
        Pnt(-d_box, -d_box, -(l_domain / 2 + d_pml)),
        Pnt(d_box, d_box, -(l_domain / 2)))
    pml_cube_back.faces.name = 'dirichlet_3d'
    pml_cube_back.faces.Max(Z).name = 'default'  # revert this one
    cyl_back = Cylinder(Pnt(0, 0, -l_domain/2-d_pml), Z, r=r_cyl, h=d_pml)
    pml_clad_back = pml_cube_back - cyl_back
    pml_core_back = pml_cube_back * cyl_back
    pml_clad_back.mat("pml_clad_back").maxh = el_clad
    pml_core_back.mat("pml_core_back").maxh = el_core

    domain_list = domain_list + [pml_clad_back,
                                 pml_core_back, pml_clad_front, pml_core_front]

    geo = OCCGeometry(domain_list)
    mesh = Mesh(geo.GenerateMesh(maxh=el_core))
    mesh.ngmesh.Save(filename)
