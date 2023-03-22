from ngsolve import *


def ModalAnalysis(fes, wl, ur, er, target, domains, nev):
    u, p = fes.TrialFunction()
    v, q = fes.TestFunction()
    kzero = 2*pi/wl
    a = BilinearForm(fes, symmetric=True)
    a += ((1./ur) * curl(u).Trace() * curl(v).Trace() - kzero **
          2 * er * u.Trace() * v.Trace())*ds(domains)

    a.Assemble()

    b = BilinearForm(fes, symmetric=True)
    b += (1./ur) * u.Trace() * v.Trace() * ds(domains)
    b += (1./ur) * (u.Trace() * grad(q).Trace() + v.Trace()
                    * grad(p).Trace()) * ds(domains)
    b += ((1./ur) * grad(p).Trace() * grad(q).Trace() - kzero **
          2*er * p * q) * ds(domains)

    b.Assemble()
    md = min(nev*4, 10)
    gfu = GridFunction(fes, multidim=md, name="sol")
    with TaskManager():
        ev = ArnoldiSolver(a.mat, b.mat, freedofs=fes.FreeDofs(),
                           vecs=list(gfu.vecs), shift=target)
        gfu.AddMultiDimComponent
        print("target {}".format(target))
        for i in range(nev):
            print(
                "ev {} ev/kzero {}".format(ev[i], ev[i]/(kzero**2)))
        return ev, gfu
