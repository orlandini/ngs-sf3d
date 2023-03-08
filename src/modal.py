from ngsolve import *


def ModalAnalysis(fes,wl,ur,er, target):
    u, p = fes.TrialFunction()
    v, q = fes.TestFunction()

    kzero = 2*pi/wl
    a = BilinearForm(fes,symmetric=True)
    a += ((1./ur) * curl(u).Trace() * curl(v).Trace() - kzero**2 * er * u.Trace() * v.Trace())*ds("clad_2d|core_2d")

    a.Assemble()

    b = BilinearForm(fes,symmetric=True)
    b += (1./ur) * u.Trace() * v.Trace() * ds("clad_2d|core_2d")
    b += (1./ur) * (u.Trace() * grad(q).Trace() + v.Trace() * grad(p).Trace()) * ds("clad_2d|core_2d")
    b += ((1./ur) * grad(p).Trace() * grad(q).Trace() - kzero**2*er * p * q) * ds("clad_2d|core_2d")

    b.Assemble()

    gfu = GridFunction(fes,multidim=10,name="sol")
    with TaskManager():
        ev = ArnoldiSolver(a.mat, b.mat, freedofs=fes.FreeDofs(),
                           vecs=list(gfu.vecs), shift=target)
        beta = -1.*sqrt(complex(ev[0]))        
        print("\nev {} target {} ev/kzero {} ".format(ev[0], target, ev[0]/(kzero**2)))
        return ev, gfu